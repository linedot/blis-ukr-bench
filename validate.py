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
import signal
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: E402

RUNGS = ['ukr', 'ir', 'jr', 'ic', 'pc', 'jc', 'pre']
EPI_FLOPS = {'general': 3, 'one': 2, 'zero': 1}


def run_bench(bench, args):
    """Run goto_bench, return (metadata, row)."""
    out = subprocess.run([bench] + [str(a) for a in args], capture_output=True, text=True)
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
        raise SystemExit("no cycle counter on this machine: pass --ghz to convert ns to cycles")
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


def cmd_calibrate(a):
    meta, _ = run_bench(a.bench, ['--size', 64, 64, 64, '--kc', 16, '--mc', 64, '--nc', 64,
                                  '--top', 'ukr', '--reps', 3, '--beta', a.beta])
    mr, nr = (int(x) for x in meta['ukr size'].split()[0].split('x'))
    l1 = a.l1 or next((sz for lv, sz, _, _ in sysfs_caches() if lv == 1), 32 * 1024)
    kmax = max(8, (l1 // 2 - 8 * mr * nr) // (8 * (mr + nr)))
    ks = sorted({max(4, int(kmax * f) // 4 * 4) for f in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1)})
    print(f"# micro-kernel {mr}x{nr} ({meta.get('ukr implementation', '?')}), "
          f"k up to {kmax} keeps it in L1 ({model._fmt_size(l1)})")
    pts, src = [], ''
    for k in ks:
        _, row = run_bench(a.bench, ['--size', mr, nr, k, '--kc', k, '--mc', mr, '--nc', nr,
                                     '--top', 'ukr', '--min-time', a.min_time, '--beta', a.beta]
                           + a.extra)
        cyc, src = cycles(row, a.ghz)
        pts.append((k, cyc))
        print(f"  k={k:>5}: {cyc:9.1f} cycles per call")
    n = len(pts)
    mk, mc = sum(k for k, _ in pts) / n, sum(c for _, c in pts) / n
    b = (sum((k - mk) * (c - mc) for k, c in pts) / sum((k - mk) ** 2 for k, _ in pts))
    c0 = mc - b * mk
    worst = max(abs(c0 + b * k - c) / c for k, c in pts)
    peak = 2 * mr * nr / b
    print(f"  fit t(k) = {c0:.1f} + {b:.3f}*k   (worst residual {100 * worst:.1f}%)")
    print(f"  -> peak {peak:.2f} FLOP/cycle, {c0:.1f} cycles per call outside the k-loop")

    rates = {}
    for op, top, size, blk in (('A', 'packA', (8 * mr, 64, 256), (256, 8 * mr, 64)),
                               ('B', 'packB', (64, 32 * nr, 256), (256, 64, 32 * nr))):
        _, row = run_bench(a.bench, ['--size', *size, '--kc', blk[0], '--mc', blk[1], '--nc', blk[2],
                                     '--top', top, '--min-time', a.min_time])
        cyc, src = cycles(row, a.ghz)
        rates[op] = float(row['work']) / cyc
        print(f"  pack {op}: {rates[op]:.2f} elements/cycle  ({int(float(row['work']))} elements in cache)")

    calib = {'mr': mr, 'nr': nr, 'impl': meta.get('ukr implementation', ''), 'beta': a.beta,
             'peak': peak, 'epi_cycles': c0, 'ukr_points': pts, 'fit_worst_residual': worst,
             'pack_rate_a': rates['A'], 'pack_rate_b': rates['B'], 'cycles_source': src,
             'ghz': a.ghz}
    with open(a.output, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f"# wrote {a.output}: --peak {peak:.2f} --epi-peak "
          f"{EPI_FLOPS[a.beta] * mr * nr / c0:.2f} --pack-rate {min(rates.values()):.2f}")
    return 0


def kernel_from(calib, beta, bcast='', call_uops='0'):
    epi = EPI_FLOPS.get(beta, 3)
    mr, nr = calib['mr'], calib['nr']
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
        if not kmin or k_c >= int(kmin):
            out.append(model.parse_prefetch(spec))
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
    target = levels[int(f[2][1:]) - 1].name
    hint = f[3] if len(f) > 3 and f[3] else 'keep'
    lead = int(f[4]) if len(f) > 4 and f[4] else 0
    t_k = 2 * u.m_r * u.n_r / u.peak
    dist = math.ceil(lead * u.time(kc_mean) / t_k) if lead else 0
    return model.sw_prefetch(f[0], f[1], target, hint, dist)


def parse_size(s):
    p = [int(x) for x in s.replace('x', ',').split(',')]
    return tuple(p * 3) if len(p) == 1 else tuple(p)


def cmd_run(a):
    with open(a.calib) as f:
        calib = json.load(f)
    if calib.get('beta') != a.beta:
        print(f"# warning: calibrated with --beta {calib.get('beta')}, running with {a.beta}; "
              f"the per-call cost may differ")
    levels = model.parse_hierarchy(a.cache)
    if a.level_names:
        levels = model.name_levels(levels, a.level_names.split(','))
    levels = sorted(levels, key=lambda L: L.size)
    emap, events_file = {}, None
    if a.events_map:
        with open(a.events_map) as f:
            emap = json.load(f)
        events_file = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
        events_file.write('\n'.join(emap.values()) + '\n')
        events_file.close()
    u = kernel_from(calib, a.beta, a.bcast, a.call_uops)
    pack = dict(kv.split(':') for kv in a.pack.split(','))
    results = []
    print(f"{'top':>4} {'size':>16} {'kc':>5} {'mc':>5} {'nc':>6}  {'measured':>9} {'model':>9} "
          f"{'ratio':>6}  model bound")
    for size, kc, mc, nc, top in itertools.product(a.size, a.kc, a.mc, a.nc, a.top):
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
        me, ne, ke = int(row['m_exec']), int(row['n_exec']), int(row['k_exec'])
        work = float(row['work'])

        out = ''.join(op for op in 'AB' if RUNGS.index(pack[op]) > RUNGS.index(top))
        flags = []
        if pack['A'] != 'ic' and 'A' not in out or pack['B'] != 'pc' and 'B' not in out:
            flags.append('pack placement differs from BLIS: model assumes in-place repacking')
        pk = model.packing(calib['pack_rate_a'], calib['pack_rate_b'], outside=out)
        lv, mem = levels, None
        if top != 'jc' and not a.no_steady:
            foot = 8 * (me * ke * ('A' not in out and 2 or 1) + ke * ne * ('B' not in out and 2 or 1)
                        + me * ne)
            lv, mem = steady_levels(levels, foot)
            if mem:
                flags.append(f'steady state: {mem} as memory')
        b = model.blocking(kc, mc, nc)
        p = model.problem(me, ne, ke)
        kc_mean = ke / math.ceil(ke / kc)
        sw = [to_model_prefetch(s, lv, u, kc_mean) for s in a.prefetch]
        sw += kernel_prefetches(a.kernel_prefetch, kc)
        hw = model.hw_prefetcher(a.pf_streams, a.pf_min_run, a.pf_level.upper())
        an = model.analyse(u, b, p, lv, pk, True, a.core_mshr, hw,
                           a.c_layout, sw, a.lookahead)
        pred = model.predict(an)
        fm, fp = work / meas, work / pred.time
        res = {'top': top, 'm': me, 'n': ne, 'k': ke, 'kc': kc, 'mc': mc, 'nc': nc,
               'measured_cycles': meas, 'model_cycles': pred.time,
               'measured_flop_per_cycle': fm, 'model_flop_per_cycle': fp,
               'ratio': fm / fp, 'model_bound': pred.bound, 'cycles_from': src,
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
        print(f"{top:>4} {f'{me}x{ne}x{ke}':>16} {kc:>5} {mc:>5} {nc:>6}  {fm:9.2f} {fp:9.2f} "
              f"{fm / fp:6.2f}  {pred.bound}" + (f"   [{res['flags']}]" if flags else ''))
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
    c = sub.choices['calibrate']
    c.add_argument('-o', '--output', default='calib.json')
    c.add_argument('--l1', type=int, help='L1 data cache bytes (default: from sysfs)')
    r = sub.choices['run']
    r.add_argument('--calib', required=True)
    r.add_argument('--cache', nargs='+', required=True, metavar='LEVEL',
                   help="model.py's --cache spec for this machine")
    r.add_argument('--level-names', default='')
    r.add_argument('--size', nargs='+', required=True, help='N for a cube, or M,N,K')
    r.add_argument('--kc', nargs='+', type=int, required=True)
    r.add_argument('--mc', nargs='+', type=int, required=True)
    r.add_argument('--nc', nargs='+', type=int, required=True)
    r.add_argument('--top', nargs='+', default=['jc'], choices=RUNGS[:6])
    r.add_argument('--pack', default='A:ic,B:pc')
    r.add_argument('--c-layout', default='rm', choices=['rm', 'cm'])
    r.add_argument('--prefetch', nargs='*', default=[],
                   help="prefetches goto_bench issues between calls (and the model gets)")
    r.add_argument('--kernel-prefetch', nargs='*', default=[], metavar='PF',
                   help="prefetches inside the micro-kernel, for the model only, in "
                        "model.py's syntax (lead in k-steps); end one in @KMIN if the "
                        "kernel issues it only when k_c >= KMIN.  AOCL's 8x24 dgemm "
                        "kernel prefetches its C tile 100-128 k-steps ahead, and only "
                        "for k_c >= 128: ir:C:L1:keep:100@128")
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
    r.add_argument('-o', '--output')
    a = p.parse_args(argv)
    return {'detect-cache': cmd_detect_cache, 'calibrate': cmd_calibrate, 'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
