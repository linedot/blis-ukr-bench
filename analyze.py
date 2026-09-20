#!/usr/bin/env python3
"""Piecewise-linear fit of BLIS microkernel scaling data.

Reads the CSV written by the ukr benchmark, e.g.

    # ukr implementation: optimzd
    # ukr size:            16x32
    # threads:             1
    Size[Byte],[GFLOP/s],FLOPS,cycles/iter,l1d_read_miss/iter
    4864,20.8372,3584,266,0
    ...

mr/nr/thread count are taken from the '#' header when present, so k can be
recovered from the working-set size:

    size = (mr*k + nr*k + mr*nr) * sizeof(double)

The cost of one microkernel call is modelled as

    t(k) = C + b*k

with C the fixed per-call overhead (C load/store, prologue) and b the cost of
one rank-1 update: mr*nr FMAs over (mr+nr) freshly streamed doubles.  b jumps
whenever the A/B panels stop fitting in a cache level, so t(k) is piecewise
linear in k and the slope of each piece is the interesting number: it gives
FLOP/cycle while the panels are in cache, and effective bandwidth once they
are not.

The segment boundaries are found automatically (dynamic programming over all
contiguous splits, fewest segments that keep every point within --tol of its
segment's line), so no size thresholds have to be hardcoded per machine.

Every b comes with its standard error, because a window that does not reach far
enough in k cannot pin down a slope -- there b trades off against C and the fit
will happily return a value the hardware cannot reach.  Pass --peak (FLOP/cycle)
to have such fits flagged outright.

If the file has no cycle counter (only Size and GFLOP/s), t(k) is recovered from
GFLOP/s in nanoseconds; --ghz converts that to cycles per call.

Only numpy is required; matplotlib is needed just for --plot.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

DOUBLE = 8  # sizeof(double), the benchmark's element type


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

COLUMN_ALIASES = {
    "size": "size", "size[byte]": "size", "sizebyte": "size", "bytes": "size",
    "gflop/s": "gflops", "gflops": "gflops", "gflop": "gflops",
    "flops": "flops", "flop": "flops",
    "cycles/iter": "cycles", "cycles": "cycles", "cyclesiter": "cycles",
    "l1d_read_miss/iter": "l1d", "l1d_read_miss": "l1d", "l1d": "l1d",
}

# positional fallback when the file has no header row at all
POSITIONAL = ["size", "gflops", "flops", "cycles", "l1d"]


def canonical(name: str) -> str | None:
    key = name.strip().lower().strip("[]")
    key = re.sub(r"[\[\]\s]", "", key)
    return COLUMN_ALIASES.get(key)


@dataclass
class Bench:
    """One benchmark run: metadata plus the columns we care about."""
    path: str
    meta: dict[str, str]
    cols: dict[str, np.ndarray]
    mr: int
    nr: int
    threads: int
    threads_assumed: bool
    k: np.ndarray
    y: np.ndarray
    unit: str          # "cycles" or "ns"
    freq_ghz: float | None
    source: str        # how t(k) was obtained

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def title(self) -> str:
        impl = self.meta.get("ukr implementation")
        return f"{self.name} ({impl})" if impl else self.name

    # per-k work, used to turn a slope into something physical
    @property
    def flops_per_k(self) -> float:
        return 2.0 * self.mr * self.nr

    @property
    def bytes_per_k(self) -> float:
        return (self.mr + self.nr) * DOUBLE


def read_table(path: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Parse '#'-commented metadata and the data table (comma or whitespace)."""
    meta: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[float]] = []
    sep: str | None = None

    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if ":" in body:
                    key, val = body.split(":", 1)
                    meta[key.strip().lower()] = val.strip()
                continue

            if header is None and sep is None:
                sep = "," if "," in line else None

            fields = [f.strip() for f in line.split(sep) if f.strip() != ""]
            try:
                values = [float(f) for f in fields]
            except ValueError:
                if header is None:
                    header = fields
                    continue
                print(f"{path}:{lineno}: skipping unparseable line", file=sys.stderr)
                continue
            rows.append(values)

    if not rows:
        raise SystemExit(f"{path}: no data rows found")

    names = header if header is not None else POSITIONAL
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows], dtype=float)

    cols: dict[str, np.ndarray] = {}
    for idx in range(width):
        label = names[idx] if idx < len(names) else ""
        key = canonical(label) or (POSITIONAL[idx] if header is None and idx < len(POSITIONAL) else None)
        if key and key not in cols:
            cols[key] = data[:, idx]

    if "size" not in cols:
        raise SystemExit(f"{path}: no 'Size[Byte]' column found (got {names})")
    return meta, cols


def parse_cache_spec(text: str) -> list[tuple[float, int]]:
    """'48K,1M,32M:8' -> [(49152,1), (1048576,1), (33554432,8)].

    The ':N' suffix says the level is shared by N threads.  A private L1 and L2
    are per core, but an L3 shared by a whole CCD gives each thread only its
    slice, so with one thread the capacity is the whole cache and with sixteen
    it is an eighth of it -- the same spec string is right for both runs.
    """
    out = []
    for item in text.split(","):
        cap, _, sh = item.partition(":")
        out.append((parse_size(cap), int(sh) if sh.strip() else 1))
    return sorted(out)


def effective_cap(cap: float, sharers: int, threads: int) -> float:
    return cap / max(1, min(threads, sharers))


def parse_size(text: str) -> float:
    """'512K', '1.5M', '32768' -> bytes."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([kKmMgG]?)[iI]?[bB]?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"cannot parse size {text!r}")
    return float(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m.group(2).lower()]


def load(path: str, args) -> Bench:
    meta, cols = read_table(path)

    mr, nr = args.mr, args.nr
    if (mr is None or nr is None) and "ukr size" in meta:
        m = re.match(r"\s*(\d+)\s*x\s*(\d+)", meta["ukr size"])
        if m:
            mr = mr or int(m.group(1))
            nr = nr or int(m.group(2))
    if mr is None or nr is None:
        raise SystemExit(
            f"{path}: microkernel size unknown -- no '# ukr size: MRxNR' header; pass --mr/--nr")

    threads_assumed = "threads" not in meta and args.threads is None
    threads = args.threads or int(meta.get("threads", 1))

    size = cols["size"]
    order = np.argsort(size)
    for key in cols:
        cols[key] = cols[key][order]
    size = cols["size"]

    keep = np.ones_like(size, dtype=bool)
    if args.min_size is not None:
        keep &= size >= args.min_size
    if args.max_size is not None:
        keep &= size <= args.max_size
    if keep.sum() < 3:
        raise SystemExit(f"{path}: fewer than 3 usable data points "
                         f"(after any --min-size/--max-size filtering)")
    for key in cols:
        cols[key] = cols[key][keep]
    size = cols["size"]

    # size = (mr*k + nr*k + mr*nr) * 8   ->   k
    k = (size - mr * nr * DOUBLE) / ((mr + nr) * DOUBLE)
    if np.any(k <= 0):
        raise SystemExit(f"{path}: derived k <= 0 -- are --mr/--nr ({mr}x{nr}) right?")
    off = np.abs(k - np.round(k))
    if np.max(off) > 1e-6 * np.max(k):
        print(f"{path}: warning: derived k is not integral (worst off-by {off.max():.3g}); "
              f"check --mr/--nr and the element size", file=sys.stderr)
    k = np.round(k)

    # per-call time: prefer measured cycles, else derive nanoseconds from GFLOP/s
    freq = None
    total_flops = cols.get("flops")
    if total_flops is None:
        total_flops = (2.0 * mr * nr * k + 3.0 * mr * nr) * threads
    ns = total_flops / cols["gflops"] if "gflops" in cols else None

    have_cycles = "cycles" in cols and np.all(cols["cycles"] > 0)
    if args.y == "cycles" and not have_cycles:
        raise SystemExit(f"{path}: no usable 'cycles/iter' column; use --y ns")
    if args.y == "ns" and ns is None:
        raise SystemExit(f"{path}: no '[GFLOP/s]' column to derive time from; use --y cycles")

    if args.y == "cycles" or (args.y == "auto" and have_cycles):
        y, unit = cols["cycles"], "cycles"
    elif ns is not None:
        y, unit = ns, "ns"
    else:
        raise SystemExit(f"{path}: need either 'cycles/iter' or '[GFLOP/s]'")

    if have_cycles and ns is not None:
        cols["ns"] = ns
        cols["ghz"] = cols["cycles"] / ns          # per point, not just a median
        freq = float(np.median(cols["ghz"]))

    source = "measured cycles" if unit == "cycles" else "derived from GFLOP/s"
    if args.ghz:
        if unit == "ns":
            # no usable cycle counter -- convert at a clock the caller supplies
            y = y * args.ghz
            unit = "cycles"
            source = f"derived from GFLOP/s at {args.ghz} GHz"
        freq = args.ghz

    return Bench(path, meta, cols, mr, nr, threads,
                 threads_assumed and "flops" not in cols, k, y, unit, freq, source)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

@dataclass
class Fit:
    lo: int            # index range [lo, hi) into the sorted points
    hi: int
    slope: float       # b   [unit per k]
    intercept: float   # C   [unit]
    rel_sse: float     # sum of squared relative residuals
    max_dev: float     # worst |relative residual|
    r2: float          # ordinary (unweighted) coefficient of determination
    span: float        # y_max / y_min over the segment
    se: float          # standard error of the slope
    se_c: float        # standard error of the intercept

    @property
    def npts(self) -> int:
        return self.hi - self.lo


def fit_affine(k: np.ndarray, y: np.ndarray, lo: int, hi: int) -> Fit:
    """Least squares y = C + b*k weighted by 1/y^2.

    The relative weighting matters: k spans four orders of magnitude here, so an
    unweighted fit would be decided entirely by the largest k and would ignore
    the small-k points that pin down the intercept.
    """
    ks, ys = k[lo:hi], y[lo:hi]
    w = 1.0 / np.square(ys)
    sw = w.sum()
    swk = (w * ks).sum()
    swy = (w * ys).sum()
    swkk = (w * ks * ks).sum()
    swky = (w * ks * ys).sum()
    den = sw * swkk - swk * swk
    if den <= 0:
        b, c = 0.0, float(np.mean(ys))
    else:
        b = (sw * swky - swk * swy) / den
        c = (swy - b * swk) / sw

    pred = c + b * ks
    rel = (pred - ys) / ys
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    span = float(ys.max() / ys.min()) if ys.min() > 0 else float("inf")
    # weighted RSS is exactly sum(rel^2) for w = 1/y^2, so the slope's standard
    # error falls out directly.  It is the honest answer to "is this b the same
    # as that b": a short window near k=0 has almost no leverage on b and says so.
    rel_sse = float(np.sum(rel * rel))
    dof = max(len(ks) - 2, 1)
    se = math.sqrt(rel_sse / dof * sw / den) if den > 0 else float("inf")
    se_c = math.sqrt(rel_sse / dof * swkk / den) if den > 0 else float("inf")
    return Fit(lo, hi, float(b), float(c), rel_sse,
               float(np.max(np.abs(rel))), r2, span, float(se), float(se_c))


def segment(k: np.ndarray, y: np.ndarray, *, min_pts: int, max_segs: int,
            tol: float, min_span: float = 1.0, force: int | None = None
            ) -> tuple[list[Fit], list[tuple[int, float, float, bool]]]:
    """Split the points into the fewest contiguous linear pieces that fit.

    Dynamic programming over every contiguous split: for each segment count S we
    find the split minimising the total relative SSE, optionally under the
    constraint that no point deviates from its line by more than `tol`.  The
    answer is the smallest S that satisfies the constraint; if no S does, the
    scan falls back to the S with the best BIC (with `tol` acting as the assumed
    noise floor) and says so.

    `min_span` additionally requires y to vary by that factor inside a segment.
    Without it the handful of smallest-k points always form a beautiful straight
    line -- they are all sitting on the fixed overhead C, the b*k term is noise
    there, and the resulting "region" carries no usable slope.
    """
    n = len(k)
    max_segs = max(1, min(max_segs, n // min_pts))

    cache: dict[tuple[int, int], Fit] = {}
    for lo in range(n):
        for hi in range(lo + min_pts, n + 1):
            cache[(lo, hi)] = fit_affine(k, y, lo, hi)

    def dp(nseg: int, enforce: bool, span: float):
        INF = float("inf")
        # best[s][i] = cost of covering the first i points with s segments
        best = [[INF] * (n + 1) for _ in range(nseg + 1)]
        back = [[-1] * (n + 1) for _ in range(nseg + 1)]
        best[0][0] = 0.0
        for s in range(1, nseg + 1):
            for i in range(s * min_pts, n + 1):
                for j in range((s - 1) * min_pts, i - min_pts + 1):
                    if best[s - 1][j] == INF:
                        continue
                    f = cache[(j, i)]
                    if f.span < span or (enforce and f.max_dev > tol):
                        continue
                    cost = best[s - 1][j] + f.rel_sse
                    if cost < best[s][i]:
                        best[s][i] = cost
                        back[s][i] = j
        if best[nseg][n] == INF:
            return None
        out, i = [], n
        for s in range(nseg, 0, -1):
            j = back[s][i]
            out.append(cache[(j, i)])
            i = j
        return list(reversed(out))

    if dp(1, False, min_span) is None:  # too few points / too flat for the guard
        min_span = 1.0

    if force is not None:
        if force > max_segs:
            raise SystemExit(f"--segments {force} needs at least {force * min_pts} points, have {n}")
        out = dp(force, False, min_span)
        if out is None:
            raise SystemExit(f"--segments {force} is not possible under --min-points {min_pts} "
                             f"/ --min-span {min_span}")
        return out, []

    scan: list[tuple[int, float, float, bool]] = []
    chosen: list[Fit] | None = None
    for s in range(1, max_segs + 1):
        free = dp(s, False, min_span)
        if free is None:
            continue
        rel_sse = sum(f.rel_sse for f in free)
        worst = max(f.max_dev for f in free)
        strict = dp(s, True, min_span)
        scan.append((s, math.sqrt(rel_sse / n), worst, strict is not None))
        if chosen is None and strict is not None:
            chosen = strict

    if chosen is None:  # tolerance unreachable -- pick the best BIC instead
        floor = n * tol * tol
        best_bic, best_s = float("inf"), 1
        for s, rrmse, _worst, _ok in scan:
            rss = n * rrmse * rrmse
            bic = n * math.log((rss + floor) / n) + 3 * s * math.log(n)
            if bic < best_bic:
                best_bic, best_s = bic, s
        chosen = dp(best_s, False, min_span)
    return chosen, scan


def merge_similar(k: np.ndarray, y: np.ndarray, fits: list[Fit],
                  min_ratio: float) -> tuple[list[Fit], int]:
    """Drop breakpoints that do not change the slope.

    A split can always be made to fit better, but a boundary is only physically
    interesting if b actually changes across it -- that is what a cache level
    running out looks like.  Neighbours whose slopes agree to within `min_ratio`
    are the same region measured twice, so they get merged and re-fitted.
    """
    merged = 0
    while len(fits) > 1 and min_ratio > 1.0:
        ratios = []
        for a, b in zip(fits, fits[1:]):
            lo, hi = min(a.slope, b.slope), max(a.slope, b.slope)
            ratios.append(hi / lo if lo > 0 else float("inf"))
        i = int(np.argmin(ratios))
        if ratios[i] >= min_ratio:
            break
        fits = fits[:i] + [fit_affine(k, y, fits[i].lo, fits[i + 1].hi)] + fits[i + 2:]
        merged += 1
    return fits, merged


def fit_extra(x: np.ndarray, k: np.ndarray, f: Fit) -> float:
    """Ordinary slope of some other column over the same index range (for L1D misses)."""
    ks, xs = k[f.lo:f.hi], x[f.lo:f.hi]
    if len(ks) < 2 or np.allclose(xs, 0):
        return float("nan")
    return float(np.polyfit(ks, xs, 1)[0])


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def human_bytes(b: float) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= scale:
            return f"{b / scale:.3g} {unit}"
    return f"{b:.0f} B"


def label_segments(bench: Bench, fits: list[Fit], caches: list[float] | None) -> list[str]:
    if not caches:
        return [""] * len(fits)
    names = ["L1", "L2", "L3", "L4"]

    def level_of(size):
        for i, (cap, sh) in enumerate(sorted(caches)):
            if size <= effective_cap(cap, sh, bench.threads):
                return names[i] if i < len(names) else f"L{i + 1}"
        return "DRAM"

    out = []
    for f in fits:
        lo = level_of(bench.cols["size"][f.lo])
        hi = level_of(bench.cols["size"][f.hi - 1])
        out.append(hi if lo == hi else f"{lo}-{hi}")
    return out


def cache_limits(bench: Bench, caches: list[float], usable: float) -> list[str]:
    """Largest k that still fits a cache level, under two different criteria.

    In a real GEMM only the B micro-panel (k_c x n_r) has to live in L1; the A
    micro-panel streams past it from L2 and C sits in registers, so the blocking
    rule is k_c * n_r * sizeof < L1.  This benchmark is not that: it calls the
    ukr in a loop on one fixed A, B and C, so all three are re-read every call
    and the whole A+B+C footprint has to be resident.  Both are printed -- the
    B-panel column is the number to pick k_c from, the A+B+C column is the one
    that predicts where this measurement breaks.
    """
    names = ["L1", "L2", "L3", "L4"]
    out = []
    for i, (cap, sh) in enumerate(sorted(caches)):
        level = names[i] if i < len(names) else f"L{i + 1}"
        eff = effective_cap(cap, sh, bench.threads)
        avail = usable * eff
        shared = f" /{min(sh, bench.threads)} thr = {human_bytes(eff)}" if eff != cap else ""
        k_all = (avail - bench.mr * bench.nr * DOUBLE) / ((bench.mr + bench.nr) * DOUBLE)
        k_b = avail / (bench.nr * DOUBLE)
        all_txt = f"k <= {math.floor(k_all):.0f}" if k_all >= 1 else "does not fit"
        out.append(f"    {level} {human_bytes(cap):>9}{shared}:  A+B+C resident for {all_txt:<14} "
                   f"|  B panel alone for k <= {math.floor(k_b):.0f}")
    return out


def amortisation(bench: Bench, f: Fit, caches, usable: float) -> list[str]:
    """How large k has to be before the fixed per-call cost stops mattering.

    C/(C + b*k) = frac  ->  k = C*(1 - frac) / (b*frac).  Read against the cache
    limits this is the real blocking trade-off: the k that amortises the call may
    be larger than the k that still fits.
    """
    if f.slope <= 0 or f.intercept <= 0:
        return []
    ks = {frac: f.intercept * (1 - frac) / (f.slope * frac) for frac in (0.2, 0.1, 0.05)}
    out = [f"  overhead amortisation (region 1: C = {f.intercept:.1f}, b = {f.slope:.3f}):",
           "    C is " + ",  ".join(f"{frac * 100:.0f}% of t(k) at k = {k:.0f}"
                                    for frac, k in ks.items())]
    for cap, sh in sorted(caches or [])[:1]:
        avail = usable * effective_cap(cap, sh, bench.threads)
        klim = (avail - bench.mr * bench.nr * DOUBLE) / ((bench.mr + bench.nr) * DOUBLE)
        if klim >= 1:
            klim = math.floor(klim)
            frac = f.intercept / (f.intercept + f.slope * klim) * 100
            out.append(f"    at the L1 A+B+C limit k = {klim:.0f}, C is {frac:.1f}% of t(k)")
    return out


def unroll_note(k: np.ndarray, unroll: int) -> str:
    """Warn when the k sweep never varies its residue against the kernel's unroll."""
    res = sorted(set(int(v) % unroll for v in k))
    if len(res) == 1:
        return (f"every sampled k is {res[0]} mod {unroll}, so a {unroll}x k-unrolled kernel ran "
                f"its tail on every point: that cost is identical at every k, cancels in the "
                f"differences between points, and lands wholly in C -- b is unaffected, C is "
                f"inflated by the tail's excess over a normal body iteration")
    return (f"sampled k covers residues {res} mod {unroll}; points on the tail path carry an "
            f"extra fixed cost the others do not, which shows up as scatter in maxdev, not in b")


def intercepts(bench: Bench, fits: list[Fit]) -> list[str]:
    """Say what each region's C actually is.

    C is the fitted line at k=0, so it is a per-call overhead only for a region
    that reaches down to small k.  For a later region it is just that line's
    offset, and a negative one is informative rather than wrong: if the region
    before it was faster, the region starts out partly served at the faster rate
    and only becomes fully limited by the slower one as k grows.  Modelling that
    as t(k) = b_prev*k_res + b*(k - k_res) makes the intercept -(b - b_prev)*k_res,
    so k_res is how much of the panel still hits the faster level -- in practice a
    measurement of the last-level cache, and one that should not depend on which
    microkernel produced it.
    """
    out = []
    for i, f in enumerate(fits, 1):
        unit = bench.unit
        if i == 1:
            out.append(f"    region 1: C = {f.intercept:.1f} +/- {f.se_c:.1f} {unit} per call"
                       + (f" (fit starts at k = {bench.k[f.lo]:.0f}, so it does not describe "
                          f"smaller k)" if bench.k[f.lo] > 4 else ""))
            continue
        prev = fits[i - 2]
        if f.intercept < 0 and f.slope > prev.slope:
            k_res = -f.intercept / (f.slope - prev.slope)
            byt = (bench.mr + bench.nr) * DOUBLE * k_res + bench.mr * bench.nr * DOUBLE
            out.append(f"    region {i}: C = {f.intercept:.0f} +/- {f.se_c:.0f} {unit} is not a "
                       f"per-call cost -- it is the region starting out partly served at region "
                       f"{i-1}'s {prev.slope:.3f} {unit}/k")
            out.append(f"               -> k_res = {k_res:.0f} ({human_bytes(byt)} of A+B) still "
                       f"resident; compare against the last-level cache")
        else:
            out.append(f"    region {i}: C = {f.intercept:.1f} +/- {f.se_c:.1f} {unit}, an "
                       f"extrapolation from k >= {bench.k[f.lo]:.0f} back to 0, not a per-call cost")
    return out


def region_ghz(bench: Bench, f: Fit) -> float | None:
    """The clock this region actually ran at.

    A single median over the whole sweep is not good enough once the machine
    changes frequency with the working set: b is in cycles and is blind to that,
    so converting it to GB/s or GFLOP/s with the wrong clock quietly misstates
    the region that matters.
    """
    if "ghz" not in bench.cols:
        return None
    return float(np.median(bench.cols["ghz"][f.lo:f.hi]))


def report(bench: Bench, fits: list[Fit], scan, caches, show_scan: bool,
           notes: list[str], peak: float | None = None, usable: float = 1.0,
           unroll: int | None = None) -> None:
    size = bench.cols["size"]
    unit = bench.unit
    per_k = f"{unit}/k"
    rate = "FLOP/cycle" if unit == "cycles" else "GFLOP/s"
    bw = "B/cycle" if unit == "cycles" else "GB/s"

    print("=" * 96)
    print(bench.title)
    src = bench.source
    thr = f"{bench.threads}{' (assumed)' if bench.threads_assumed else ''}"
    print(f"  ukr {bench.mr}x{bench.nr}   threads {thr}   t(k) in {unit} per call ({src})"
          + (f"   ~{bench.freq_ghz:.3f} GHz" if bench.freq_ghz else ""))
    print(f"  {len(bench.k)} points   k {bench.k[0]:.0f}..{bench.k[-1]:.0f}   "
          f"working set {human_bytes(size[0])}..{human_bytes(size[-1])}")

    if caches:
        print(f"  cache limits (usable fraction {usable:g}):")
        for line in cache_limits(bench, caches, usable):
            print(line)

    if show_scan and scan:
        print("  segment scan (rel. RMSE / worst deviation / meets --tol):")
        for s, rrmse, worst, ok in scan:
            print(f"    {s} segment{'s' if s > 1 else ' '}: "
                  f"rel.RMSE {rrmse * 100:6.2f}%   worst {worst * 100:6.2f}%   {'yes' if ok else 'no'}")

    if unroll:
        notes.append(unroll_note(bench.k, unroll))
    if "ghz" in bench.cols:
        g = bench.cols["ghz"]
        if g.max() / g.min() > 1.03:
            lo_i = int(np.argmin(g))
            notes.append(
                f"the clock is not constant: {g.max():.3f} GHz at "
                f"{human_bytes(bench.cols['size'][int(np.argmax(g))])} down to {g.min():.3f} GHz "
                f"at {human_bytes(bench.cols['size'][lo_i])} ({100*(1-g.min()/g.max()):.1f}% lower). "
                f"b is in cycles and cannot see this -- compare GFLOP/s, not FLOP/cycle, "
                f"when deciding where to block")
    if peak:
        over = [i for i, f in enumerate(fits, 1) if bench.flops_per_k / f.slope > peak]
        for i in over:
            f = fits[i - 1]
            floor = bench.flops_per_k / peak
            notes.append(
                f"region {i}: b = {f.slope:.3f} is below the {floor:.3f} {per_k} floor implied by "
                f"--peak {peak:g} -- not physically reachable, so that window does not pin down b "
                f"(se {f.se:.3f}); it is usually the small-k end, where b*k barely moves t(k)")
    for note in notes:
        print(f"  note: {note}")
    labels = label_segments(bench, fits, caches)
    has_l1d = "l1d" in bench.cols and np.any(bench.cols["l1d"] > 0)
    head = (f"  {'seg':<7} {'pts':>4} {'k range':>16} {'working set':>20} "
            f"{'b [' + per_k + ']':>14} {'+/-':>7} {'C [' + unit + ']':>11} {'R^2':>8} "
            f"{'maxdev':>8} {rate:>11} {bw:>9}")
    has_ghz = "ghz" in bench.cols
    if has_ghz:
        head += f" {'GHz':>6}"
    if peak:
        head += f" {'%peak':>7}"
    if has_l1d:
        head += f" {'miss/k':>7}"
    print("-" * len(head))
    print(head)
    print("-" * len(head))
    for i, f in enumerate(fits, 1):
        ks = f"{bench.k[f.lo]:.0f}..{bench.k[f.hi - 1]:.0f}"
        ss = f"{human_bytes(size[f.lo])}..{human_bytes(size[f.hi - 1])}"
        tag = f"{i} {labels[i - 1]}".strip()
        row = (f"  {tag:<7} {f.npts:>4} {ks:>16} {ss:>20} "
               f"{f.slope:>14.3f} {f.se:>7.3f} {f.intercept:>11.1f} {f.r2:>8.5f} "
               f"{f.max_dev * 100:>7.2f}% "
               f"{bench.flops_per_k / f.slope:>11.2f} {bench.bytes_per_k / f.slope:>9.2f}")
        if has_ghz:
            row += f" {region_ghz(bench, f):>6.3f}"
        if peak:
            row += f" {bench.flops_per_k / f.slope / peak * 100:>6.1f}%"
        if has_l1d:
            miss = fit_extra(bench.cols["l1d"], bench.k, f)
            row += f" {miss:>7.2f}" if np.isfinite(miss) else f"{'-':>8}"
        print(row)
    print("-" * len(head))
    print("  intercepts:")
    for line in intercepts(bench, fits):
        print(line)
    base = next((f for f in fits
                 if not peak or bench.flops_per_k / f.slope <= peak), fits[0])
    for line in amortisation(bench, base, caches, usable):
        print(line)
    for i, f in enumerate(fits, 1):
        gbs = ""
        fghz = region_ghz(bench, f) or bench.freq_ghz
        if unit == "cycles" and fghz:
            per = bench.bytes_per_k / f.slope * fghz
            gbs = (f", {per:.1f} GB/s" if bench.threads == 1 else
                   f", {per:.1f} GB/s/thread (x{bench.threads} = {per * bench.threads:.0f} GB/s)")
            gbs += f" @ {fghz:.3f} GHz"
        print(f"  {i}: t(k) = {f.intercept:.2f} + {f.slope:.4f}*k {unit}"
              f"   ->  {bench.flops_per_k / f.slope:.1f} {rate}, "
              f"{bench.bytes_per_k / f.slope:.2f} {bw}{gbs}")
    print()


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

# dotted / dashed / dash-dot, in the order cache levels are given
CACHE_STYLES = [(0, (1, 2.5)), (0, (6, 3)), (0, (8, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1))]
# one linestyle per region, so the fits stay separable in grayscale too
REGION_STYLES = ["--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
BAND_FILL = ["none", "0.895"]


def region_bands(benches: list[Bench], all_fits: list[list[Fit]], key: str):
    """Band edges between regions, in k or in working-set bytes.

    Each boundary is put at the geometric mean of the last point of one region
    and the first of the next.  With several files the edges are averaged the
    same way; `spread` says how far the files disagree, so a caller can tell
    whether one set of bands honestly describes all of them.
    """
    nreg = min(len(f) for f in all_fits)
    edges, spread = [], 1.0
    for i in range(1, nreg):
        vals = []
        for b, fits in zip(benches, all_fits):
            arr = b.k if key == "k" else b.cols["size"]
            vals.append(math.sqrt(arr[fits[i].lo] * arr[fits[i].lo - 1]))
        edges.append(math.exp(sum(math.log(v) for v in vals) / len(vals)))
        spread = max(spread, max(vals) / min(vals))
    return edges, spread, nreg


def draw_bands(ax, edges, labels, lo, hi, annotate=False):
    bounds = [lo] + list(edges) + [hi]
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        if BAND_FILL[i % 2] != "none":
            ax.axvspan(a, b, color=BAND_FILL[i % 2], lw=0, zorder=0)
        if annotate and i < len(labels):
            ax.text(math.sqrt(a * b), 0.985, labels[i], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color=".35")


def plot(benches: list[Bench], all_fits: list[list[Fit]], caches, path: str | None,
         show: bool, peak: float | None = None, slope_scale: str = "auto") -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".8"})
    except ImportError:
        print("matplotlib or seaborn not available -- skipping plot", file=sys.stderr)
        return

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(7, 11.5), gridspec_kw=dict(height_ratios=[1, 1, 1, 0.85]))
    base_colors = sns.color_palette("tab10", max(len(benches), 3))

    # regions get the background, files get the hue -- one channel each
    k_edges, k_spread, nreg = region_bands(benches, all_fits, "k")
    s_edges, _, _ = region_bands(benches, all_fits, "size")
    names = label_segments(benches[0], all_fits[0], caches)
    labels = [n or f"region {i+1}" for i, n in enumerate(names[:nreg])]
    kall = np.concatenate([b.k for b in benches])
    sall = np.concatenate([b.cols["size"] for b in benches])
    draw_bands(ax1, k_edges, labels, kall.min() * 0.7, kall.max() * 1.4, annotate=True)
    draw_bands(ax2, k_edges, labels, kall.min() * 0.7, kall.max() * 1.4)
    draw_bands(ax3, s_edges, labels, sall.min() * 0.7, sall.max() * 1.4)
    draw_bands(ax4, k_edges, labels, kall.min() * 0.7, kall.max() * 1.4)
    if k_spread > 1.5:
        print(f"note: the files' region boundaries differ by up to {k_spread:.1f}x; "
              f"the shaded bands are their average", file=sys.stderr)

    for idx, (b, fits) in enumerate(zip(benches, all_fits)):
        base = base_colors[idx]

        ax1.plot(b.k, b.y, "o", color=base, ms=5, zorder=5, label=b.title)
        for j, f in enumerate(fits):
            kk = np.linspace(b.k[f.lo], b.k[f.hi - 1], 64)
            ax1.plot(kk, f.intercept + f.slope * kk, lw=2.0, color=base,
                     ls=REGION_STYLES[j % len(REGION_STYLES)],
                     label=f"    {labels[j] if j < len(labels) else j+1}: "
                           f"b={f.slope:.4g} {b.unit}/k")

        mid = np.sqrt(b.k[:-1] * b.k[1:])
        ax2.plot(mid, np.diff(b.y) / np.diff(b.k), "o-", color=base, ms=4, lw=1,
                 alpha=0.75, label=b.title)
        for j, f in enumerate(fits):
            ax2.hlines(f.slope, b.k[f.lo], b.k[f.hi - 1], color=base, lw=2.2,
                       ls=REGION_STYLES[j % len(REGION_STYLES)])

        if "gflops" in b.cols:
            ax3.plot(b.cols["size"], b.cols["gflops"], "o-", color=base, ms=4, lw=1.4,
                     label=b.title)

        # the fixed cost, on a scale where it is actually visible: what share of
        # each call it is.  C only means "per-call overhead" for the region that
        # reaches small k, so use that one.
        f = next((x for x in fits if not peak or b.flops_per_k / x.slope <= peak), fits[0])
        if f.intercept > 0 and f.slope > 0:
            kk = np.logspace(np.log10(b.k[f.lo]), np.log10(b.k[f.hi - 1]), 200)
            ax4.plot(kk, 100 * f.intercept / (f.intercept + f.slope * kk), color=base, lw=2,
                     label=f"{b.title}: C = {f.intercept:.1f} {b.unit}")
            kk_d = b.k[f.lo:f.hi]
            share = 100 * (b.y[f.lo:f.hi] - f.slope * kk_d) / b.y[f.lo:f.hi]
            model = 100 * f.intercept / (f.intercept + f.slope * kk_d)
            # past the point where C is a per-cent of the call, y - b*k is a small
            # difference of large numbers and measures slope drift, not overhead
            ok = (share > 0) & (model > 1.0)
            ax4.plot(kk_d[ok], share[ok], "o", color=base, ms=5, alpha=0.85)

    unit = benches[0].unit
    ghz = next((b.freq_ghz for b in benches if b.freq_ghz), None)

    # A DRAM region can be 25x the in-cache slope, which flattens every cache
    # plateau against the axis floor.  Decide from the spread rather than fixing
    # one scale: log only helps when the regions are far apart, and only works
    # when every local slope is positive.
    sl = np.concatenate([np.diff(b.y) / np.diff(b.k) for b in benches]
                        + [np.array([f.slope for f in fits]) for fits in all_fits])
    positive = bool((sl > 0).all())
    spread = sl.max() / sl.min() if positive else float("inf")
    if slope_scale == "auto":
        log_slope = positive and spread > 5
        if not positive and sl.max() / sl[sl > 0].min() > 5:
            print("note: slope panel kept linear -- some local slopes are <= 0 "
                  "(non-monotonic t(k)); --slope-scale log would drop them",
                  file=sys.stderr)
    else:
        log_slope = slope_scale == "log"

    ax1.set(xscale="log", yscale="log", xlabel="k (inner loop iterations)",
            ylabel=f"time per ukr call [{unit}]", title="t(k) = C + b*k, fitted per region")
    ax1.legend(fontsize=7.5, loc="upper left")

    if peak:
        floor = benches[0].flops_per_k / peak
        ax2.axhline(floor, color="k", ls="--", lw=1.6,
                    label=f"Peak ({floor:.4g} {unit}/k)")
    ax2.set(xscale="log", yscale="log" if log_slope else "linear", xlabel="k",
            ylabel=f"local slope d t/d k [{unit}/k]",
            title="local slope -- plateaus are the detected regions")
    ax2.legend(fontsize=8)

    nt0 = benches[0].threads
    for i, (cap, sh) in enumerate(sorted(caches or [])):
        eff = effective_cap(cap, sh, nt0)
        tag = f" /{min(sh, nt0)} thr" if eff != cap else ""
        ax3.axvline(eff, color=".35", lw=1.4, ls=CACHE_STYLES[i % len(CACHE_STYLES)],
                    label=f"L{i+1} Cache ({human_bytes(cap)}{tag})")
    if peak:
        # two references: the best clock the run ever reached, and the clock each
        # point actually ran at.  data-to-curve is per-cycle efficiency; curve-to-
        # line is what the machine gave away in frequency.
        if len({b.threads for b in benches}) > 1:
            print("note: the files use different thread counts; the peak line uses "
                  f"{benches[0].threads}", file=sys.stderr)
        nt = benches[0].threads
        best = max((b.cols["ghz"].max() for b in benches if "ghz" in b.cols),
                   default=ghz)
        if best:
            ax3.axhline(peak * best * nt, color="k", ls="--", lw=1.6,
                        label=f"Peak ({nt} thread{'s' if nt > 1 else ''} at {best:.2f} GHz)")
        for idx, b in enumerate(benches):
            if "ghz" not in b.cols or b.cols["ghz"].max() / b.cols["ghz"].min() < 1.01:
                continue
            g = b.cols["ghz"]
            ax3.plot(b.cols["size"], peak * g * b.threads, color=".35", lw=1.3,
                     ls=(0, (4, 2)), zorder=1,
                     label=f"Peak at measured clock ({g.max():.2f} -> {g.min():.2f} GHz)")
    ax3.set(xscale="log", xlabel="working set A+B+C [byte]", ylabel="GFLOP/s",
            title="throughput vs working set")
    ax3.legend(fontsize=8, ncol=2, loc="best")

    for guide in (50, 25, 10, 5, 1):
        ax4.axhline(guide, color=".45", ls=(0, (1, 3)), lw=1)
        ax4.text(0.997, guide, f"{guide}% ", transform=ax4.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7.5, color=".45")
    ax4.set(xscale="log", yscale="log", xlabel="k (inner loop iterations)",
            ylabel="C / t(k)  [%]",
            title="per-call overhead -- share of t(k) that is the fixed cost C")
    ax4.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")
    if show:
        plt.show()


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="benchmark output file(s)")
    p.add_argument("--mr", type=int, help="override microkernel MR (default: from '# ukr size')")
    p.add_argument("--nr", type=int, help="override microkernel NR")
    p.add_argument("--threads", type=int, help="override thread count (only used if the file has no FLOPS column)")
    p.add_argument("--y", choices=("auto", "cycles", "ns"), default="auto",
                   help="quantity to fit: measured cycles/iter, or ns/iter derived from GFLOP/s")
    p.add_argument("--ghz", type=float,
                   help="clock frequency; with no cycle counter in the file this converts the "
                        "derived ns into cycles, so b and C come out in cycles per call")
    p.add_argument("--peak", type=float, metavar="FLOP_PER_CYCLE",
                   help="theoretical peak of the microkernel, e.g. 128 for one 8x8 f64 FMOPA per "
                        "cycle; adds a %%peak column and flags any fit that beats it")
    p.add_argument("--tol", type=float, default=3.0,
                   help="max relative deviation inside a region, in %% (default: 3)")
    p.add_argument("--min-points", type=int, default=3, help="minimum points per region (default: 3)")
    p.add_argument("--min-slope-ratio", type=float, default=1.25,
                   help="merge neighbouring regions whose slopes differ by less than this "
                        "factor (default: 1.25); 1 keeps every breakpoint the fit finds")
    p.add_argument("--min-span", type=float, default=1.5,
                   help="a region must cover at least this factor in t(k) (default: 1.5); "
                        "stops the overhead-dominated small-k points from forming a bogus region")
    p.add_argument("--max-segments", type=int, default=6, help="upper bound on the number of regions")
    p.add_argument("--segments", type=int, help="force exactly this many regions")
    p.add_argument("--breaks", help="force boundaries at these working-set sizes, e.g. 512K,8M")
    p.add_argument("--caches",
                   help="cache capacities for labelling, e.g. 48K,1M,32M:8; also prints the "
                        "largest k that fits each level. ':N' marks a level shared by N "
                        "threads, so a multithreaded run gets its per-thread slice")
    p.add_argument("--unroll", type=int, metavar="N",
                   help="k-unroll factor of the microkernel; checks whether the sweep always "
                        "lands on the same residue mod N (then the tail cost hides in C)")
    p.add_argument("--usable", type=float, default=1.0,
                   help="fraction of each cache assumed available to the panels (default: 1); "
                        "0.75 leaves one way of a 4-way cache for the other streams")
    p.add_argument("--min-size", type=parse_size, help="ignore points below this working-set size")
    p.add_argument("--max-size", type=parse_size, help="ignore points above this working-set size")
    p.add_argument("--plot", nargs="?", const="", metavar="FILE",
                   help="draw the fits; with a filename, save instead of showing")
    p.add_argument("--slope-scale", choices=("auto", "linear", "log"), default="auto",
                   help="y scale of the local-slope panel; auto goes log when the regions "
                        "differ by more than 5x (default: auto)")
    p.add_argument("--no-scan", action="store_true", help="hide the segment-count scan")
    args = p.parse_args(argv)

    tol = args.tol / 100.0
    caches = parse_cache_spec(args.caches) if args.caches else None
    breaks = [parse_size(b) for b in args.breaks.split(",")] if args.breaks else None

    benches, all_fits = [], []
    for path in args.csv:
        b = load(path, args)
        n = len(b.k)
        notes: list[str] = []

        if breaks:
            cuts = sorted({int(np.searchsorted(b.cols["size"], s, side="left")) for s in breaks})
            bounds = [0] + [c for c in cuts if 0 < c < n] + [n]
            fits = [fit_affine(b.k, b.y, lo, hi) for lo, hi in zip(bounds, bounds[1:])]
            scan = []
        elif args.segments:
            fits, scan = segment(b.k, b.y, min_pts=args.min_points,
                                 max_segs=max(args.segments, args.max_segments),
                                 tol=tol, min_span=args.min_span, force=args.segments)
        else:
            fits, scan = segment(b.k, b.y, min_pts=args.min_points,
                                 max_segs=args.max_segments, tol=tol, min_span=args.min_span)
            if max(f.max_dev for f in fits) > tol:
                notes.append(f"no split keeps every point within --tol {args.tol}% "
                             f"(noisy data?); showing the best-scoring split instead")

        if not breaks and not args.segments:
            fits, nmerged = merge_similar(b.k, b.y, fits, args.min_slope_ratio)
            if nmerged:
                notes.append(f"merged {nmerged} boundar{'ies' if nmerged > 1 else 'y'} across which "
                             f"b changed by less than {args.min_slope_ratio}x -- same region either "
                             f"side (--min-slope-ratio 1 keeps them, which may exceed --tol)")
        report(b, fits, scan, caches, not args.no_scan, notes,
               args.peak, args.usable, args.unroll)
        benches.append(b)
        all_fits.append(fits)

    if args.plot is not None:
        plot(benches, all_fits, caches, args.plot or None, show=not args.plot,
             peak=args.peak, slope_scale=args.slope_scale)
    return 0


if __name__ == "__main__":
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # play nicely with | head
    except (ImportError, AttributeError, ValueError):
        pass
    sys.exit(main())
