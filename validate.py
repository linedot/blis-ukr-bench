#!/usr/bin/env python3
"""
Calibrate model.py from goto_bench, then compare its predictions with what
goto_bench measures.

    validate.py detect-cache
    validate.py calibrate --bench build/goto_bench [--ghz G] -o calib.json
    validate.py run --bench build/goto_bench --calib calib.json \\
                    --cache 32K:64:8:128/64:1.0:16/8:5 1M:64:8:64/32:1.0:32/16:18 INF:::8:::150 \\
                    --size 2000 4000 --kc 128 256 --mc 96 --nc 4096 [--top jc ir ...] \\
                    [--pack A:ic,B:pc] [--prefetch ir:C:L2 ...] [--events-map events.json] \\
                    [-o results.csv]

detect-cache  prints a --cache spec from sysfs: sizes, lines and ways.  The
              bandwidths, latencies and buffer counts are not in sysfs, and
              have to come from microbenchmarks or vendor data.

calibrate     the micro-kernel at k small enough to stay in L1, fitted as
              t(k) = C + b*k: peak = 2*mr*nr/b and the per-call cost C, which
              the model charges as its epilogue.  Then one A_c and one B_c
              block packed in cache: --pack-rate.  Cycles come from the cycle
              counter if it is available, otherwise from ns * --ghz.

run           every combination of the sweep values, each measured by
              goto_bench and replayed through the model with the same
              blocking, layout, beta, packing placement and prefetches.  Two
              adjustments make the replay like for like:

              * packing placed above --top happens once, outside the timed
                region, so the model gets packing.outside for that operand;
              * a region below jc repeats on the same data, which in steady
                state lives in the smallest cache that holds it -- so that
                level is treated as memory (--no-steady to turn this off).

              Compared on cycles per region run, and on FLOP/cycle with the
              same 2*m*n*k work on both sides.  With --events-map (from
              probe_events.py) the measured counters are set against the
              model's predicted counts as well.

The model assumes BLIS's packing placement (A at ic, B at pc) when packing is
inside the region: its rule that a packed block is repacked in place depends
on it.  Other placements are measured, and flagged in the output.
"""

import argparse
import csv
import dataclasses
import glob
import itertools
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: E402

RUNGS = ['ukr', 'ir', 'jr', 'ic', 'pc', 'jc', 'pre']
EPI_FLOPS = {'general': 3, 'one': 2, 'zero': 1}


PIN = []          # --cpu N for goto_bench, from --cpu


def run_bench(bench, args):
    """Run goto_bench, return (metadata, row)."""
    out = subprocess.run([bench] + [str(a) for a in args] + PIN, capture_output=True,
                         text=True)
    if out.returncode != 0:
        raise RuntimeError(f"goto_bench {' '.join(map(str, args))} failed:\n{out.stderr}{out.stdout}")
    meta, lines = {}, []
    for line in out.stdout.splitlines():
        if line.startswith('#'):
            k, _, v = line[1:].partition(':')
            meta[k.strip()] = v.strip()
        elif line.strip():
            lines.append(line)
    rows = list(csv.DictReader(lines))
    for w in out.stderr.splitlines():
        if 'warning' in w and w not in _seen_warnings:
            _seen_warnings.add(w)
            print(f"  {w.strip()}  (shown once)", file=sys.stderr)
    return meta, rows[0]


_seen_warnings = set()


def cycles(row, ghz):
    c = float(row['cycles_min'])
    if c > 0:
        return c, 'cycle counter'
    if not ghz:
        raise SystemExit("no cycles in this run: the cycle counter could not be opened (see "
                         "goto_bench's warnings above), or this machine has none -- pass "
                         "--ghz to convert ns to cycles")
    return float(row['ns_min']) * ghz, f'ns * {ghz} GHz'


def sysfs_caches():
    """[(level, size_bytes, line, ways)] for data and unified caches."""
    out = []
    for d in sorted(glob.glob('/sys/devices/system/cpu/cpu0/cache/index*')):
        def rd(f):
            try:
                with open(os.path.join(d, f)) as fh:
                    return fh.read().strip()
            except OSError:
                return ''
        if rd('type') == 'Instruction':
            continue
        size = rd('size')
        mult = {'K': 1024, 'M': 1024**2, 'G': 1024**3}.get(size[-1:], 1)
        out.append((int(rd('level') or 0), int(size.rstrip('KMG') or 0) * mult,
                    int(rd('coherency_line_size') or 64), int(rd('ways_of_associativity') or 0)))
    return sorted(out)


def cmd_detect_cache(_):
    caches = sysfs_caches()
    if not caches:
        print("no cache information in sysfs")
        return 1
    specs = [f"{model._fmt_size(sz)}:{line}:{ways or ''}:-1" for _, sz, line, ways in caches]
    print("--cache " + ' '.join(specs) + " INF:::-1")
    print("# size:line:ways come from sysfs; replace each -1 with rd/wr bandwidth in B/cycle,")
    print("# and add usable:mshr/wbuf:latency -- these are not in sysfs (see model.py --help)")
    return 0


def theil_sen(xs, ys):
    """Line through (x, y): median pairwise slope, so one run off the line does not tilt it."""
    slopes = [(ys[j] - ys[i]) / (xs[j] - xs[i])
              for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i]]
    b = statistics.median(slopes)
    return statistics.median(y - b * x for x, y in zip(xs, ys)), b


def per_kstep_table(results, mr, nr):
    """
    Per call = fixed + per k-step x k_c, for cycles and every counter, over runs
    that differ only in k_c. The k-step part is the k-loop's; the fixed part is
    everything once per call -- the C update, the call itself, a prefetch hook.
    """
    groups = {}
    for r in results:
        key = (r['top'], r.get('size'), r['m'], r['n'], r['mc'], r['nc'])
        groups.setdefault(key, []).append(r)
    k_call = lambda r: min(r['kc'], r['k'])        # what each call runs
    for (top, size, m, n, mc, nc), rs in groups.items():
        ks = sorted({k_call(r) for r in rs})
        if len(ks) < 3:
            continue
        print(f"# per call = fixed + per k-step x k: top {top}, problem {size}, region "
              f"{m}x{n} (m_c {mc}, n_c {nc}), {len(rs)} runs, k {ks[0]}-{ks[-1]}")
        print(f"  {'':<30} {'per k-step':>10} {'per call':>10}   off the line by over 10%")
        cols = ['measured_cycles'] + [c for c in rs[0] if c.endswith('_measured')]
        for col in cols:
            pts = [(k_call(r), r[col] / r['calls']) for r in rs if col in r and r['calls']]
            if len(pts) < 3:
                continue
            c0, b = theil_sen([k for k, _ in pts], [y for _, y in pts])
            off = [f"{k} ({(y - c0 - b * k) / y:+.0%})" for k, y in pts
                   if y and abs(y - c0 - b * k) > 0.1 * abs(y) and abs(y - c0 - b * k) > 5]
            name = 'cycles' if col == 'measured_cycles' else col[:-len('_measured')]
            note = (f"   = {2 * mr * nr / b:.2f} FLOP/cycle in the k-loop"
                    if col == 'measured_cycles' and b > 0 else '')
            print(f"  {name:<30} {b:10.2f} {c0:10.0f}   {', '.join(off)}{note}")
            if col == 'measured_cycles' and c0 < 0:
                print("  # a negative per-call part: these k span two regimes (the region "
                      "outgrows a cache between them); split them to compare")


def cmd_calibrate(a):
    meta, _ = run_bench(a.bench, ['--size', 64, 64, 64, '--kc', 16, '--mc', 64, '--nc', 64,
                                  '--top', 'ukr', '--reps', 3, '--beta', a.beta,
                                  '--c-layout', a.c_layout])
    mr, nr = (int(x) for x in meta['ukr size'].split()[0].split('x'))
    l1 = a.l1 or next((sz for lv, sz, _, _ in sysfs_caches() if lv == 1), 32 * 1024)
    kmax = max(8, (l1 // 2 - 8 * mr * nr) // (8 * (mr + nr)))
    ks = sorted({max(4, int(kmax * f) // 4 * 4)
                 for f in (1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2, 3 / 4, 1)})
    print(f"# micro-kernel {mr}x{nr} ({meta.get('ukr implementation', '?')}), "
          f"k up to {kmax} keeps it in L1 ({model._fmt_size(l1)})")
    pts, src = [], ''
    for k in ks:
        _, row = run_bench(a.bench, ['--size', mr, nr, k, '--kc', k, '--mc', mr, '--nc', nr,
                                     '--top', 'ukr', '--min-time', a.min_time, '--beta', a.beta,
                                     '--c-layout', a.c_layout] + a.extra)
        cyc, src = cycles(row, a.ghz)
        pts.append((k, cyc))
        print(f"  k={k:>5}: {cyc:9.1f} cycles per call")
    # the smallest k overlaps the per-call work with its short k-loop
    # differently, which bends the line: fit k >= 8 when that leaves three.
    # One bad point bends it too -- a peak above the hardware's is the sign
    # -- so the worst is dropped while it misses the line by over 2%
    fit = [(k, c) for k, c in pts if k >= 8]
    if len(fit) < 3:
        fit = pts

    def line(points):
        n = len(points)
        mk, mc = sum(k for k, _ in points) / n, sum(c for _, c in points) / n
        if a.peak:                               # slope given: median intercept
            b = 2 * mr * nr / a.peak
            c0 = sorted(c - b * k for k, c in points)[n // 2]
        else:
            b = (sum((k - mk) * (c - mc) for k, c in points)
                 / sum((k - mk) ** 2 for k, _ in points))
            c0 = mc - b * mk
        res = {k: abs(c0 + b * k - c) / c for k, c in points}
        return b, c0, res

    b, c0, res = line(fit)
    while len(fit) > 3 and max(res.values()) > 0.02:
        bad = max(res, key=res.get)
        print(f"  (left out k={bad}: {100 * res[bad]:.1f}% off the line)")
        fit = [p for p in fit if p[0] != bad]
        b, c0, res = line(fit)
    worst = max(res.values())
    peak = 2 * mr * nr / b
    print(f"  fit t(k) = {c0:.1f} + {b:.3f}*k over k = {[k for k, _ in fit]}"
          f"   (worst residual {100 * worst:.1f}%)")
    print(f"  -> peak {peak:.2f} FLOP/cycle" + (" (given)" if a.peak else "")
          + f", {c0:.1f} cycles per call outside the k-loop")
    if worst > 0.02:
        print("  # warning: the points do not lie on a line -- rerun with a longer "
              "--min-time, or give --peak")

    # Pack rates with source and packed copy in L1, in L2, and the source from
    # memory (4x the last level, at most 64 MB). The L1 rate is the pack
    # kernel's own cost, which the model takes: its memory terms add where the
    # source lives. The others show what each level costs on top -- on an
    # in-order core, every miss not prefetched stalls for its full latency.
    caches = sysfs_caches()
    l2 = a.l2 or next((sz for lv, sz, _, _ in caches if lv == 2), 1 << 20)
    llc = max([sz for _, sz, _, _ in caches] + [l2])
    places = (('L1', l1 // 24), ('L2', l2 // 24), ('memory', min(4 * llc, 64 << 20) // 8))
    rates, by_level = {}, {}
    print("# pack kernels, elements/cycle with source and packed copy in:")
    for op, top, w in (('A', 'packA', mr), ('B', 'packB', nr)):
        by_level[op] = {}
        for where, elems in places:
            # 240 and an odd panel count: power-of-two leading dimensions alias in
            # the cache and measure that instead (a 4-way L1 shows it first)
            kc = min(240, max(8, elems // w // 8 * 8))
            width = max(w, elems // kc // w * w)
            if width > w and width & (width - 1) == 0:
                width += w
            size, blk = (((width, 64, kc), (kc, width, 64)) if op == 'A' else
                         ((64, width, kc), (kc, 64, width)))
            _, row = run_bench(a.bench, ['--size', *size, '--kc', blk[0], '--mc', blk[1],
                                         '--nc', blk[2], '--top', top,
                                         '--min-time', a.min_time] + a.extra)
            cyc, src = cycles(row, a.ghz)
            by_level[op][where] = float(row['work']) / cyc
        rates[op] = by_level[op]['L1']
        lv = by_level[op]
        print(f"  pack {op}: " + ', '.join(f"{lv[w]:.3g} {w}" for w, _ in places)
              + (f"   (from memory {lv['L1'] / lv['memory']:.0f}x slower than L1: "
                 "latency exposed)" if lv['memory'] < 0.25 * lv['L1'] else ""))

    calib = {'mr': mr, 'nr': nr, 'impl': meta.get('ukr implementation', ''), 'beta': a.beta,
             'c_layout': a.c_layout, 'pack_rates': by_level,
             'peak': peak, 'epi_cycles': c0, 'ukr_points': pts, 'fit_worst_residual': worst,
             'pack_rate_a': rates['A'], 'pack_rate_b': rates['B'], 'cycles_source': src,
             'ghz': a.ghz}
    calib.update(calibrate_in_nest(a, mr, nr, l1))
    with open(a.output, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f"# wrote {a.output}: --peak {peak:.2f} --epi-peak "
          f"{EPI_FLOPS[a.beta] * mr * nr / c0:.2f} --pack-rate {min(rates.values()):.2f}")
    return 0


def calibrate_in_nest(a, mr, nr, l1):
    """
    Per-call costs as the nest sees them: one ir loop of 4 calls, everything
    resident in L1, at two k.  With --events-map, loads and stores per call
    against k: the intercepts, less the C tile's, are --call-uops, the slope
    is loads per k-step.  With --prefetch, the same with goto_bench's
    prefetch hook running: the extra cycles, loads and stores per call.
    """
    k_hi = 4
    while (4 * mr * (k_hi + 4) + (k_hi + 4) * nr + 4 * mr * nr) * 8 <= l1 // 2:
        k_hi += 4
    ks, out = (max(4, k_hi // 2 // 4 * 4), k_hi), {}
    emap = json.load(open(a.events_map)) if a.events_map else {}
    want = {c: emap[c] for c in ('L1D_CACHE_RD', 'L1D_CACHE_WR') if c in emap}
    ev_args = []
    if want:
        f = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
        f.write('\n'.join(want.values()) + '\n')
        f.close()
        ev_args = ['--events-file', f.name, '--group', a.group]

    def per_call(k, pf):
        args = ['--size', 4 * mr, nr, k, '--kc', k, '--mc', 4 * mr, '--nc', nr, '--top', 'ir',
                '--beta', a.beta, '--c-layout', a.c_layout,
                '--min-time', a.min_time] + ev_args + a.extra
        if pf:
            args += ['--prefetch'] + pf
        _, row = run_bench(a.bench, args)
        calls = float(row['calls'])
        res = {'cycles': cycles(row, a.ghz)[0] / calls}
        for c, spec in want.items():
            if f"{spec}_min" in row:
                res[c] = float(row[f"{spec}_min"]) / calls
        return res

    base = {k: per_call(k, []) for k in ks}
    print(f"# in the nest (--top ir, 4 calls, in L1): k={ks[0]}: {base[ks[0]]['cycles']:.1f}, "
          f"k={ks[1]}: {base[ks[1]]['cycles']:.1f} cycles per call")
    c_ops = mr * nr * 8 / a.vl
    if all('L1D_CACHE_RD' in base[k] and 'L1D_CACHE_WR' in base[k] for k in ks):
        slope = (base[ks[1]]['L1D_CACHE_RD'] - base[ks[0]]['L1D_CACHE_RD']) / (ks[1] - ks[0])
        ld = base[ks[0]]['L1D_CACHE_RD'] - slope * ks[0] - (c_ops if a.beta != 'zero' else 0)
        st = sum(base[k]['L1D_CACHE_WR'] for k in ks) / 2 - c_ops
        out.update(call_ld=ld, call_st=st, loads_per_kstep=slope)
        vec = (mr + nr) * 8 / a.vl
        shape = min((abs(slope - x), name) for x, name in
                    ((vec, 'vectors'), (mr + nr * 8 / a.vl, '--bcast A'),
                     (mr * 8 / a.vl + nr, '--bcast B'), (mr + nr, '--bcast AB')))[1]
        print(f"  {slope:.2f} loads per k-step (closest: {shape}); per call outside the k-loop "
              f"and the C tile: {ld:.0f} loads, {st:.0f} stores  -> --call-uops {ld:.0f}/{st:.0f}")
    if a.prefetch:
        key = ' '.join(sorted(a.prefetch))
        hook = {k: per_call(k, a.prefetch) for k in ks}
        d = {c: sum(hook[k][c] - base[k][c] for k in ks) / 2
             for c in ('cycles', 'L1D_CACHE_RD', 'L1D_CACHE_WR') if all(c in hook[k] for k in ks)}
        out['prefetch_hook'] = {key: {'cycles': d['cycles'], 'ld': d.get('L1D_CACHE_RD'),
                                      'st': d.get('L1D_CACHE_WR')}}
        print(f"  prefetch hook ({key}): {d['cycles']:.1f} cycles per call"
              + (f", {d['L1D_CACHE_RD']:.0f} loads and {d['L1D_CACHE_WR']:.0f} stores"
                 if 'L1D_CACHE_RD' in d else ''))
    if ev_args:
        os.unlink(f.name)
    return out


def kernel_from(calib, beta, bcast='', call_uops='0'):
    epi = EPI_FLOPS.get(beta, 3)
    mr, nr = calib['mr'], calib['nr']
    if call_uops in ('', '0') and 'call_ld' in calib:      # calibrated
        ld, st = str(calib['call_ld']), str(calib['call_st'])
    else:
        ld, _, st = call_uops.partition('/')
    return model.ukernel(mr, nr, 8, calib['peak'], epi * mr * nr / calib['epi_cycles'], epi,
                         beta != 'zero', bcast=bcast.upper(),
                         call_ld=float(ld or 0), call_st=float(st or 0))


def kernel_prefetches(specs, k_c):
    """
    --kernel-prefetch specs that apply at this k_c.  A spec may end in
    @KMIN, for a prefetch the kernel only issues when k_c >= KMIN.
    """
    out = []
    for s in specs:
        spec, _, kmin = s.partition('@')
        if kmin and k_c < int(kmin):
            continue
        f = spec.split(':')
        if len(f) == 4 and f[3] not in ('', 'keep', 'strm'):
            f = f[:3] + ['keep', f[3]]            # ir:C:L1:2kc
        if len(f) > 4 and f[4]:                  # N k-steps, or Nkc: N calls
            f[4] = str(max(0, round(parse_lead(f[4], k_c))))
        out.append(model.parse_prefetch(':'.join(f)))
    return out


def steady_levels(levels, footprint):
    """Treat the smallest cache that holds the region's data as its memory."""
    for i, L in enumerate(levels[:-1]):
        if footprint <= L.cap:
            return levels[:i] + [dataclasses.replace(L, size=float('inf'))], L.name
    return levels, None


def to_model_prefetch(spec, levels, u, kc_mean):
    """goto_bench's loop:op:Lx[:hint[:lead in calls]] -> model.sw_prefetch."""
    f = spec.split(':')
    if int(f[2][1:]) >= len(levels):
        return None         # a steady state that ends above the target: nothing to fetch
    target = levels[int(f[2][1:]) - 1].name
    hint = f[3] if len(f) > 3 and f[3] else 'keep'
    lead = int(f[4]) if len(f) > 4 and f[4] else 0
    t_k = 2 * u.m_r * u.n_r / u.peak
    dist = math.ceil(lead * u.time(kc_mean) / t_k) if lead else 0
    return model.sw_prefetch(f[0], f[1], target, hint, dist)


def parse_block(text, mr, nr):
    """A block size: N, or a multiple of the micro-tile: Nmr, Nnr (4mr, 200nr, mr)."""
    m = re.fullmatch(r'(\d*)\s*(mr|nr)', str(text).strip(), re.I)
    if m:
        return int(m.group(1) or 1) * (mr if m.group(2).lower() == 'mr' else nr)
    try:
        return int(text)
    except ValueError:
        raise SystemExit(f"block size {text!r}: give N, Nmr or Nnr") from None


def parse_lead(text, k_c=None):
    """
    A prefetch lead: N in the spec's own unit, or Nkc / Nk -- N calls ahead
    (k_c None: goto_bench counts calls), or N*k_c k-steps (the kernel's).
    """
    m = re.fullmatch(r'(\d+(?:\.\d+)?|\.\d+)?\s*(kc|k)?', text.strip(), re.I)
    if not m or not (m.group(1) or m.group(2)):
        raise SystemExit(f"prefetch lead {text!r}: give N, or Nkc for N calls ahead")
    n = float(m.group(1)) if m.group(1) else 1.0
    return n if not m.group(2) or k_c is None else n * k_c


def norm_prefetch(spec):
    """
    goto_bench's loop:op:Lx[:hint[:lead]], the lead in calls as N or Nkc --
    also in the hint's place: ir:C:L2:2kc is ir:C:L2:keep:2.
    """
    f = spec.split(':')
    if len(f) == 4 and f[3] not in ('', 'keep', 'strm'):
        f = f[:3] + ['keep', f[3]]
    if len(f) > 4 and f[4]:
        n = parse_lead(f[4])
        if n != int(n):
            raise SystemExit(f"prefetch {spec}: goto_bench prefetches whole tiles, so "
                             "its lead is a whole number of calls")
        f[4] = str(int(n))
    return ':'.join(f)


def c_aliasing(ld, count, L1):
    """
    How many of a C tile's `count` columns (rows, for row-major C), `ld`
    elements apart, land in each L1 set: above the associativity they evict
    each other on every call.  0 when they fit.
    """
    if not L1.assoc or not math.isfinite(L1.size):
        return 0
    way = int(L1.size) // L1.assoc
    per_set = math.ceil(count / (way // math.gcd(8 * ld, way)))
    return per_set if per_set > L1.assoc else 0


def parse_size(s):
    p = [int(x) for x in s.replace('x', ',').split(',')]
    return tuple(p * 3) if len(p) == 1 else tuple(p)


def cmd_run(a):
    with open(a.calib) as f:
        calib = json.load(f)
    if calib.get('c_layout', 'rm') != a.c_layout:
        print(f"# warning: calibrated with --c-layout {calib.get('c_layout', 'rm')}, running "
              f"{a.c_layout}: the per-call cost may differ")
    if calib.get('beta') != a.beta:
        print(f"# warning: calibrated with --beta {calib.get('beta')}, running with {a.beta}; "
              f"the per-call cost may differ")
    def levels_at(ghz):
        lv = model.parse_hierarchy(a.cache, ghz)
        if a.level_names:
            lv = model.name_levels(lv, a.level_names.split(','))
        return sorted(lv, key=lambda L: L.size)

    # a spec with ns or GB/s fields is re-read at each run's own clock
    clocked = model.has_clock_units(a.cache)
    levels = None if clocked else levels_at(a.ghz)
    emap, events_file = {}, None
    if a.events_map:
        with open(a.events_map) as f:
            emap = json.load(f)
        events_file = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
        events_file.write('\n'.join(emap.values()) + '\n')
        events_file.close()
    u = kernel_from(calib, a.beta, a.bcast, a.call_uops)
    hook = calib.get('prefetch_hook', {}).get(' '.join(sorted(a.prefetch)))
    if hook and a.prefetch:
        u = dataclasses.replace(u, call_extra=hook['cycles'])
        print(f"# prefetch hook from the calibration: {hook['cycles']:.0f} cycles per call")
    elif a.prefetch:
        print(f"# note: no calibrated prefetch hook for {' '.join(a.prefetch)}: "
              f"calibrate with the same --prefetch to charge its cost")
    pack = dict(kv.split(':') for kv in a.pack.split(','))
    blocks = [[parse_block(x, calib['mr'], calib['nr']) for x in xs]
              for xs in (a.kc, a.mc, a.nc)]
    if len(a.size) > 1:
        print("# problems: " + ', '.join('x'.join(map(str, parse_size(s))) for s in a.size)
              + "  (one MxNxK problem is written 512x14x96)")
    results = []
    print(f"{'top':>4} {'size':>16} {'kc':>5} {'mc':>5} {'nc':>6}  {'measured':>9} {'model':>9} "
          f"{'ratio':>6} {'GHz':>5}  model bound")
    for size, kc, mc, nc, top in itertools.product(a.size, *blocks, a.top):
        m, n, k = parse_size(size)
        args = ['--size', m, n, k, '--kc', kc, '--mc', mc, '--nc', nc, '--top', top,
                '--pack', a.pack, '--c-layout', a.c_layout, '--beta', a.beta,
                '--min-time', a.min_time] + a.extra
        if a.prefetch:
            args += ['--prefetch'] + a.prefetch
        if events_file:
            args += ['--events-file', events_file.name, '--group', a.group]
        _, row = run_bench(a.bench, args)
        meas, src = cycles(row, calib.get('ghz') or a.ghz)
        # the clock this run ran at: cycles over time, or the one given
        ghz_run = (float(row['cycles_min']) / float(row['ns_min'])
                   if float(row['cycles_min']) > 0 and float(row['ns_min']) > 0
                   else (calib.get('ghz') or a.ghz))
        if clocked and not ghz_run:
            raise SystemExit("--cache has ns or GB/s fields, but this run has no cycle "
                             "counter to tell its clock: pass --ghz")
        lev = levels if not clocked else levels_at(ghz_run)
        me, ne, ke = int(row['m_exec']), int(row['n_exec']), int(row['k_exec'])
        work = float(row['work'])

        out = ''.join(op for op in 'AB' if RUNGS.index(pack[op]) > RUNGS.index(top))
        flags = []
        if pack['A'] != 'ic' and 'A' not in out or pack['B'] != 'pc' and 'B' not in out:
            flags.append('pack placement differs from BLIS: model assumes in-place repacking')
        pk = model.packing(calib['pack_rate_a'], calib['pack_rate_b'], outside=out)
        lv, mem = lev, None
        if top != 'jc' and not a.no_steady:
            foot = 8 * (me * ke * ('A' not in out and 2 or 1) + ke * ne * ('B' not in out and 2 or 1)
                        + me * ne)
            lv, mem = steady_levels(lev, foot)
            if mem:
                flags.append(f'steady state: {mem} as memory')
        ld, count = (m, u.n_r) if a.c_layout == 'cm' else (n, u.m_r)
        per_set = c_aliasing(ld, count, lev[0])
        if per_set:
            flags.append(f"C's leading dimension {ld} puts {per_set} of a tile's {count} "
                         f"{'columns' if a.c_layout == 'cm' else 'rows'} in each L1 set "
                         f"({lev[0].assoc}-way): conflict misses on every call")
        b = model.blocking(kc, mc, nc)
        p = model.problem(me, ne, ke)
        kc_mean = ke / math.ceil(ke / kc)
        sw = [x for x in (to_model_prefetch(s, lv, u, kc_mean) for s in a.prefetch) if x]
        if len(sw) < len(a.prefetch):
            flags.append(f"a prefetch below {mem}, the steady state's memory: not modelled")
        n_harness = len(sw)
        sw += kernel_prefetches(a.kernel_prefetch, kc)
        hw = model.hw_prefetcher(a.pf_streams, a.pf_min_run, a.pf_level.upper())
        an = model.analyse(u, b, p, lv, pk, True, a.core_mshr, hw,
                           a.c_layout, sw, a.lookahead, a.sw_pf_holds_inner)
        if hook and hook.get('ld') is not None:
            # the hook's own loads and stores; its prefetch instructions the
            # model already counts
            ks_all = p.m * p.n * p.k / (u.m_r * u.n_r)
            pf_n = sum(ev.instr_per_kstep for ev in an.evals[:n_harness]) * ks_all / an.calls
            an.kernel = dataclasses.replace(u, call_ld=u.call_ld + max(0.0, hook['ld'] - pf_n),
                                            call_st=u.call_st + hook['st'])
        pred = model.predict(an)
        fm, fp = work / meas, work / pred.time
        res = {'top': top, 'm': me, 'n': ne, 'k': ke, 'kc': kc, 'mc': mc, 'nc': nc,
               'measured_cycles': meas, 'model_cycles': pred.time,
               'measured_flop_per_cycle': fm, 'model_flop_per_cycle': fp,
               'ratio': fm / fp, 'model_bound': pred.bound, 'cycles_from': src,
               'calls': an.calls, 'size': f"{m}x{n}x{k}",
               'flags': '; '.join(flags)}
        if emap:
            ev = model.counters(an, a.vl).events
            for canon, spec in emap.items():
                col = f"{spec}_min"
                if col in row:
                    res[f"{canon}_measured"] = float(row[col])
                    if ev.get(canon, 0) > 0:
                        res[f"{canon}_model"] = ev[canon]
            occ = [res.get(f"L1D_MISS_OCCUPANCY_{w}") for w in ('measured', 'model')]
            ref = [res.get(f"L1D_CACHE_REFILL_{w}") for w in ('measured', 'model')]
            for w, o, r, t in zip(('measured', 'model'), occ, ref, (meas, pred.time)):
                if o and r:
                    res[f"l1d_miss_latency_{w}"] = o / r       # Little's law
                    res[f"l1d_misses_in_flight_{w}"] = o / t
        results.append(res)
        res['ghz'] = ghz_run
        print(f"{top:>4} {f'{me}x{ne}x{ke}':>16} {kc:>5} {mc:>5} {nc:>6}  {fm:9.2f} {fp:9.2f} "
              f"{fm / fp:6.2f} {f'{ghz_run:.2f}' if ghz_run else '-':>5}  {pred.bound}"
              + (f"   [{res['flags']}]" if flags else ''))
        for canon in emap:
            mv, pv = res.get(f"{canon}_measured"), res.get(f"{canon}_model")
            if mv is not None and pv:
                print(f"       {canon:<22} measured {mv:14.0f}  model {pv:14.0f}  ratio {mv / pv:6.2f}")
            elif mv is not None:
                print(f"       {canon:<22} measured {mv:14.0f}  (not modelled)")
        for what, unit, key in (("mean L1D miss latency", "cy", "l1d_miss_latency"),
                                ("L1D misses in flight", "  ", "l1d_misses_in_flight")):
            mv, pv = res.get(f"{key}_measured"), res.get(f"{key}_model")
            if mv is not None:
                print(f"       {what:<22} measured {mv:11.2f} {unit}"
                      + (f"  model {pv:11.2f} {unit}  ratio {mv / pv:6.2f}" if pv else ""))
    if results:
        r = [x['ratio'] for x in results]
        g = math.exp(sum(math.log(x) for x in r) / len(r))
        print(f"# measured/model FLOP/cycle: geometric mean {g:.3f}, range {min(r):.3f}-{max(r):.3f}"
              f" over {len(r)} runs")
    if a.per_kstep and results:
        per_kstep_table(results, calib['mr'], calib['nr'])
    if a.output and results:
        keys = sorted({k for x in results for k in x}, key=lambda k: list(results[0]).index(k)
                      if k in results[0] else 999)
        with open(a.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"# wrote {a.output}")
    if events_file:
        os.unlink(events_file.name)
    return 0


def main(argv=None):
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)    # quiet under | head
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('detect-cache', help='print a --cache spec from sysfs')
    for name in ('calibrate', 'run'):
        s = sub.add_parser(name)
        s.add_argument('--bench', required=True, help='path to goto_bench')
        s.add_argument('--ghz', type=float, help='clock, if there is no cycle counter')
        s.add_argument('--beta', default='general', choices=list(EPI_FLOPS))
        s.add_argument('--min-time', type=float, default=0.3)
        s.add_argument('--extra', nargs='*', default=[], help='more goto_bench options, e.g. --cpu 2')
        s.add_argument('--cpu', type=int,
                       help="pin goto_bench to this CPU (on a desktop, keep its SMT "
                            "sibling idle; on a server, avoid the CPU that takes the "
                            "interrupts, often 0)")
        s.add_argument('--c-layout', default='rm', choices=['rm', 'cm'],
                       help="C's layout: a kernel given the one it does not prefer may "
                            "move C element by element (goto_bench prints the "
                            "preference); calibrate and run with the same")
    c = sub.choices['calibrate']
    c.add_argument('-o', '--output', default='calib.json')
    c.add_argument('--prefetch', nargs='*', default=[],
                   help="also measure goto_bench's prefetch hook with these specs: its "
                        "cycles, loads and stores per call ('run' with the same "
                        "--prefetch then charges them)")
    c.add_argument('--events-map', help='events.json from probe_events.py: measure '
                                        'loads and stores per call (--call-uops)')
    c.add_argument('--vl', type=int, default=64,
                   help='vector length in bytes, as run (NEON 16, AVX2 or 256-bit RVV 32)')
    c.add_argument('--peak', type=float,
                   help="the micro-kernel's peak, FLOP/cycle, if known: fixes the "
                        "slope, so only the per-call cost is fitted")
    c.add_argument('--group', type=int, default=4)
    c.add_argument('--l1', type=int, help='L1 data cache bytes (default: from sysfs)')
    c.add_argument('--l2', type=int, help='L2 cache bytes (default: from sysfs)')
    r = sub.choices['run']
    r.add_argument('--calib', required=True)
    r.add_argument('--cache', nargs='+', required=True, metavar='LEVEL',
                   help="model.py's --cache spec for this machine")
    r.add_argument('--level-names', default='')
    r.add_argument('--size', nargs='+', required=True,
                   help='problems: N for a cube, MxNxK (or M,N,K) otherwise -- '
                        'three numbers are three cubes')
    r.add_argument('--kc', nargs='+', required=True, help='N, or Nmr / Nnr')
    r.add_argument('--mc', nargs='+', required=True, help='N, or a multiple of the tile: 4mr')
    r.add_argument('--nc', nargs='+', required=True, help='N, or a multiple of the tile: 200nr')
    r.add_argument('--top', nargs='+', default=['jc'], choices=RUNGS[:6])
    r.add_argument('--pack', default='A:ic,B:pc')
    r.add_argument('--prefetch', nargs='*', default=[],
                   help="prefetches goto_bench issues between calls (and the model gets)")
    r.add_argument('--kernel-prefetch', nargs='*', default=[], metavar='PF',
                   help="prefetches inside the micro-kernel, for the model only, in "
                        "model.py's syntax (lead in k-steps, or k for the whole "
                        "k-loop: issued at the call's start); end one in @KMIN if the "
                        "kernel issues it only when k_c >= KMIN.  AOCL's 8x24 dgemm "
                        "kernel: ir:C:L1:keep:100@128; BLIS's armv8a 8x6: "
                        "ir:C:L1:keep:k")
    r.add_argument('--pf-streams', type=int, default=0,
                   help="hardware prefetcher streams, as model.py (0: not modelled)")
    r.add_argument('--pf-min-run', type=int, default=4,
                   help="lines before the hardware prefetcher detects a stream")
    r.add_argument('--pf-level', default='',
                   help="level the hardware prefetcher fills, e.g. L1 or L2")
    r.add_argument('--core-mshr', type=int, default=0)
    r.add_argument('--bcast', default='', metavar='OPERANDS',
                   help="operands the kernel's k-loop loads as broadcasts (A for "
                        "AOCL's 8x24 dgemm kernel), for the L1D_CACHE uop counts")
    r.add_argument('--sw-pf-holds-inner', action='store_true',
                   help="software prefetches into an outer level hold the inner "
                        "levels' fill buffers until they land, as model.py")
    r.add_argument('--call-uops', default='0', metavar='LD[/ST]',
                   help="memory uops per call outside the k-loop and the C tile, "
                        "as model.py --call-uops (AOCL's 8x24 kernel in goto_bench "
                        "on Zen 5: about 90/47)")
    r.add_argument('--lookahead', type=float, default=0.0, metavar='CYCLES',
                   help="cycles before the epilogue the core can issue its C loads, "
                        "as model.py --lookahead")
    r.add_argument('--vl', type=int, default=64,
                   help="vector length in bytes, for the model's L1D_CACHE uop counts: "
                        "64 for SME or AVX-512, 32 for AVX2 or 256-bit RVV, 16 for NEON")
    r.add_argument('--events-map', help='events.json from probe_events.py')
    r.add_argument('--group', type=int, default=4)
    r.add_argument('--no-steady', action='store_true',
                   help='model regions below jc against the full hierarchy')
    r.add_argument('--per-kstep', action='store_true',
                   help='split cycles and every counter into a per-call and a '
                        'per-k-step part, fitted over runs that differ only in k_c')
    r.add_argument('-o', '--output')
    a = p.parse_args(argv)
    global PIN
    PIN = ['--cpu', str(a.cpu)] if getattr(a, 'cpu', None) is not None else []
    if getattr(a, 'prefetch', None):
        a.prefetch = [norm_prefetch(s) for s in a.prefetch]
    return {'detect-cache': cmd_detect_cache, 'calibrate': cmd_calibrate, 'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
