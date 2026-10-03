#!/usr/bin/env python3
"""
Bottom-up traffic, bandwidth and concurrency model for a BLIS-style GEMM
loop nest.

The nest is  jc -> pc -> [pack B_c] -> ic -> [pack A_c] -> jr -> ir -> ukr,
and every loop has exactly one operand that is invariant across it:

    loop   trip count     invariant   footprint        reuse factor
    ir     m_c/m_r        B_r         k_c*n_r*s        m_c/m_r
    jr     n_c/n_r        A_c         m_c*k_c*s        n_c/n_r
    ic     ceil(m/m_c)    B_c         k_c*n_c*s        ceil(m/m_c)
    pc     ceil(k/k_c)    C panel     m*n_c*s          ceil(k/k_c)
    jc     ceil(n/n_c)    A (full)    m*k*s            ceil(n/n_c)

Blocks are clamped to their dimension and the cache-block loops count whole
trips, a partial last block included.  In the footprint column a block size
is the largest block, min(block, dim), since that is what has to fit; in the
ir and jr rows it is the mean realised block, dim / trips.  Micro-tiles are
still counted continuously (m/m_r): padding to multiples of m_r and n_r is
not modelled.

Traffic crossing a boundary is worked out per operand.  A rung's reuse applies
only to the operand it keeps invariant: each operand's core-side traffic is
divided by the product of the reuse factors of its own rungs held strictly
inside the boundary.  A rung's streamers get nothing from it -- they are the
bytes that stream past the invariant -- and only rungs of their own reduce
them.  So once jr:A is held in L2, A's traffic beyond L2 is divided by
n_c/n_r, and B's and C's are untouched.  Packing traffic is not divided by
reuse; see PACKING.  Reads and writes are tracked separately end to end.

Three things can limit a run, and all three are evaluated:

  BANDWIDTH   traffic across a boundary against that level's bus width.

  CONCURRENCY A read holds a buffer at every level it passes through, each
              from that level's own request until the data is back -- so for
              as long as the rest of the trip takes, which depends on where
              the line comes from.  A line from memory holds L2's buffer for
              the memory latency, L1's for L2 + memory, and the core's queue
              for all of it.  Each level's buffers are a separate pool with a
              separate constraint, on buffer-cycles:

                  reads   t >= ( N_rd(i)*L(i) + N_rd(i+1)*L(i+1) + ... ) / M_rd
                  writes  t >= N_wr(i)*L(i) / M_wr

              That chaining means a small fraction of deep misses can consume
              most of the inner buffers' cycles.  Writes do not chain: a dirty
              eviction frees the inner buffer as soon as the next level
              accepts it.

              The one exception is a prefetch into an outer level, which is
              taken to hold only that level's buffers: it passes the inner
              ones without allocating, which is why it can relieve them.
              That is implementation-dependent -- if the core tracks such a
              prefetch in an inner fill buffer until it lands, the relief is
              not there.

  COMPUTE     FLOP / peak, plus the packing kernel's element rate.

PHASES.  Packing and the micro-kernel do not overlap in single-threaded BLIS,
and pack traffic is not spread over the whole run -- it all happens inside
t_pack.  Each phase is therefore evaluated against its own time and the two
are summed, which typically makes packing an order of magnitude more expensive
than averaging it in would suggest.

PACKING is charged explicitly:

  * work:  A_c is packed once per (jc,pc,ic) -> ceil(n/n_c)*m*k elements
           B_c is packed once per (jc,pc)    -> k*n elements
  * time:  elements / pack_rate, in cycles
  * reads: amplified by `amp`, the bytes fetched per byte used.  Packing reads
           the *user's* matrix, so a kernel that walks the short stride touches
           a whole line per element: amp = line/s.  One that reads contiguous
           runs and transposes in registers gets amp = 1.
  * writes: one pass over the packed panel -- plus, if the stores are narrower
           than a line, a read-for-ownership pass as well (`rfo`).  A full-line
           vector store skips that; an element-at-a-time store cannot.
  * residency: BLIS packs every block into the same buffer.  If that buffer
           is resident in a cache, the micro-kernel reads it from there and
           the next pack overwrites it in place, so beyond that level neither
           the micro-kernel's pass over the operand nor the buffer's
           write-back exists: only the pack's read of the source crosses.
           A's source is reread on every jc trip, unless all of A is resident.
           A buffer that fits no cache is evicted dirty, and all of it is
           charged -- which on LX2 is B_c.

RESIDENCY is a hard cutoff: a rung either fits a level or it does not.  That
is pessimistic below the threshold (a footprint that overflows still gets some
reuse) and optimistic above it (one that fits still loses lines to streaming
interference).  It is therefore NOT a one-sided bound.  Use the `usable`
field to bracket it: run at 1.0 and at ~0.5 and report the interval.

Cache spec, innermost level first.  One --cache argument per level:

    size : line : assoc : rd[/wr] : usable : mshr[/wbuf] : latency

Any field may be left empty to take its default (line 64, associativity
unknown, usable 1.0, no buffers modelled, latency 0, bandwidth unknown):

    --cache "32K:64:8:128/64:1.0:16/8:5" "768K::12:64/32:1.0:32/16:18" \
            "INF:::8/4:::90"
    --cache "32K:::64;1M:::48;INF:::8"         one string, ';' between levels
    --cache generic                            a named preset

A single number in the rd[/wr] or mshr[/wbuf] field means one shared resource
-- one bus carrying both directions, or one buffer pool serving both -- and
the two are then checked together rather than separately.  -1 means unknown:
the level is reported but never blamed.  INF marks main memory: its capacity
is never consulted for residency, it only terminates the chain and supplies a
bandwidth and a latency.

Line size is per level and defaults to 64 B.  Most parts are uniform, but not
all: some use 256 B throughout, and some widen only the outermost cache.  Fill
counts, buffer occupancy and the predicted refill counters all use the line
size of the level that owns the buffer, not a single global value.

KNOWN LIMITATION: over-fetch across a line-size step is not modelled.  If an
outer level has a wider line than the one inside it, a miss there pulls the
wider line; for a sequential stream the rest gets used, but for scattered
access (short runs at a long stride, which is what C looks like) the extra
bytes are wasted.  The stream table prints each operand's run length so you
can see when a run is shorter than a level's line.

Usage:
    python3 model.py --cache "32K:64:8:128/64:1.0:16/8:4" \
                     "1M::8:48/24:1.0:16/8:20" "INF:::8/4:::150"
    python3 model.py --sw-prefetch ir:C:L2:keep k:A:L1:strm:4
    python3 model.py --pack scalar         # naive strided packing
    python3 model.py --cache generic --sweep kc
    python3 model.py --counters            # predicted PMU counts
    python3 model.py --selftest
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import argparse


# ---------------------------------------------------------------- inputs ---

@dataclass
class cache_level:
    """
    One level of the hierarchy.  rbw/wbw are what it delivers inward,
    toward the core (L1->core, L2->L1, MEM->L2), in bytes/cycle.  None means
    unknown: the level is reported but never blamed.  When `shared_bw` is set
    the two are one bus and reads+writes are checked together against rbw.
    Main memory is a cache_level with size=inf; its capacity is never
    consulted for residency, it only terminates the chain.

    :param name: Name of this Cache
    :param size: Size of the cache in bytes
    :param line: Size of the cache line in bytes
    :param assoc: Associativity; 0 if unknown.  Needed only to model
                  streaming insertion, where a streamer is charged one way.
    :param rbw: Max. read throughput in bytes/cycle
    :param wbw: Max. write throughput in bytes/cycle
    :param shared_bw: Whether read/write share one bus
    :param usable: Usable part of the Cache (to account for streaming effects/
                   associativity/etc... )
    :param mshr: Max. outstanding read requests (ARM calls them MSHRs, Intel -
                 Fill Buffers, ...)
    :param wbuf: Max. outstanding write requests
    :param shared_buf: whether reads and writes share buffers (like shared_bw
                       for bandwidth)
    :param lat: Load-to-use latency of the cache in cycles, as seen from lower
                levels

    """
    name: str
    size: float
    line: int = 64
    assoc: int = 0


    rbw: float = None
    wbw: float = None
    shared_bw: bool = True
    usable: float = 1.0
    mshr: int = 0
    wbuf: int = 0
    shared_buf: bool = True
    lat: float = 0.0

    @property
    def cap(self) -> float:
        """
        Gives capacity according to size and usable fraction

        :return: usable size in bytes
        """
        return self.size * self.usable

    @property
    def way(self) -> float:
        """
        Size of one way: sets x line size.

        :return: bytes per way, or 0 if the associativity is unknown
        """
        return self.size / self.assoc if self.assoc else 0.0


@dataclass
class ukernel:
    """
    microkernel model

    :param m_r: Blocksize m_r
    :param n_r: Blocksize n_r
    :param s: Element size in bytes (FP32=4, FP64=8, ...), for every operand
              that `sizes` does not name
    :param peak: FLOP/cycle in the k-loop
    :param epi_peak: FLOP/cycle in the epilogue
    :param epi_flops: Epilogue flops per C element:
                        - general: 3
                        - beta==1: 2
                        - beta==0: 1
                        - alpha==1, beta==0: 0
    :param c_read: Whether the epilogue loads C (False for beta==0)
    :param sizes: Element size per operand, overriding s -- e.g.
                  {'A': 1, 'B': 1, 'C': 4} for int8 inputs accumulating into
                  int32.  peak and epi_peak are then in that precision's
                  operations per cycle.
    """
    m_r: int
    n_r: int
    s: int = 8
    peak: float = 16.0
    epi_peak: float = 16.0

    epi_flops: int = 3
    c_read: bool = True
    sizes: dict[str, int] = field(default_factory=dict)

    def size(self, op : str) -> int:
        """
        :param op: 'A', 'B' or 'C'
        :return: element size of that operand in bytes
        """
        return self.sizes.get(op, self.s)

    def time(self, k_c : int) -> float:
        """
        Ideal (compute-bound) cycles for one micro-kernel call.

        :param k_c: Blocksize k_c
        :return: Cycles to execute the microkernel at 100% compute efficiency
        """
        return (2 * k_c * self.m_r * self.n_r / self.peak
                + self.epi_flops * self.m_r * self.n_r / self.epi_peak)


@dataclass
class hw_prefetcher:
    """
    Hardware prefetcher.

    A stream is covered only if it is long enough to be detected and there is
    a tracker free for it.  Covered fills still occupy a fill buffer -- the
    prefetcher does not raise the concurrency ceiling, it only starts the fill
    early so the demand load does not stall on it.

    Set `level` to make its coverage count: a prefetcher filling level t
    relieves every holder inside t of the latency beyond t.  Left empty, the
    stream table is reported as a diagnostic only.

    :param streams: Maximum number of streams the prefetcher can track
    :param min_run: Minimum number of lines until the hw prefetcher detects a stream
    :param level: name of the level it fills into, e.g. 'L2'; '' = diagnostic
                  only
    """
    streams: int = 0
    min_run: int = 4
    level: str = ''


@dataclass
class sw_prefetch:
    """
    One software prefetch, issued in a loop of the nest and fetching what the
    NEXT iteration of that loop will use.

    What it fetches is that operand's per-iteration instance.  By default it
    is issued at the start of an iteration, for the next one, so its lead is
    one iteration -- but `distance` can make the lead shorter, in k-steps:
    ir:C:L2:keep:16 is the C tile prefetched 16 k-steps before the epilogue
    reads it.  The k-loop inside the micro-kernel is the one non-rung case:
    there the prefetch runs `distance` k-steps ahead, 1 by default.

    A rung-loop prefetch is issued as a burst, so its lines go through the
    target's fill buffers in waves.  C is read a whole tile at a time by the
    epilogue; A and B are read progressively, so later lines are also needed
    later and a burst can keep pace.

    :param loop: loop it is issued in: 'k', 'ir', 'jr', 'ic', 'pc' or 'jc'
    :param operand: 'A', 'B' or 'C'; must change across `loop`
    :param target: name of the level it fills into, e.g. 'L1', 'L2'
    :param hint: insertion policy at the target: 'keep' (MRU, the normal
                 case) or 'strm' (near LRU, so the line evicts other
                 streamers rather than reused data).  'strm' changes capacity,
                 not latency.
    :param distance: lead in k-steps from issue to use; 0 means the default
                     -- 1 for loop 'k', one full iteration for the others
    """
    loop: str
    operand: str
    target: str
    hint: str = 'keep'
    distance: int = 0


@dataclass
class packing:
    """
    packing model

    :param rate_a: elements per cycle that can be packed for A when not memory
                   bound.  0 means packing is charged nothing at all.
    :param rate_b: elements per cycle that can be packed for B when not memory
                   bound
    :param amp_a: bytes fetched for A per byte used on the read side (see
                  module docstring)
    :param amp_b: bytes fetched for B per byte used on the read side (see
                  module docstring)
    :param rfo: whether a partial cacheline store causes a read-for-ownership
    :param outside: operands packed outside the modelled region, e.g. 'A' or
                    'AB' -- packed once and untimed, as when a benchmark
                    repeats only the rungs below the packing.  They cost no
                    packing time and no packing traffic.
    """
    rate_a: float = 4.0
    rate_b: float = 4.0
    amp_a: float = 1.0
    amp_b: float = 1.0
    rfo: bool = False
    outside: str = ''


    @staticmethod
    def preset(name : str, s : int, line=64) -> packing:
        """
        Named presets for packing models
          
          'none'   : packing is free/not counted
          'scalar' : 0.25 elements per cycle, amp_b = line/s, rfo=True
          default : rate=4.0, amp=1.0, no RFO

        :param name: name of the preset
        :param s: element size of B in bytes -- 'scalar' gathers B across
                  the long stride, so one element costs a line
        :param line: line size of the level the pack kernel reads through,
                     for the strided read amplification of 'scalar'
        :return: packing model corresponding to the preset
        """
        if name == 'none':
            return packing(0.0, 0.0, 1.0, 1.0, False)
        if name == 'scalar':
            # naive loops: ~1 element per 4 cycles; B is gathered across the
            # long stride so every element costs a whole line on the read side,
            # and the element-at-a-time stores cannot fill a line, so each one
            # also drags the line in first
            return packing(0.25, 0.25, 1.0, line / s, True)
        return packing(4.0, 4.0, 1.0, 1.0, False)   # 'vector'


@dataclass
class blocking:
    """
    Blocksizes that aren't microkernel sizes
    """
    k_c: int
    m_c: int
    n_c: int


@dataclass
class problem:
    """
    Matrix sizes, A=m x k; B=k x n; C=m x n
    """
    m: int
    n: int
    k: int


@dataclass
class goto_rung:
    """
    Rungs modeling the loops of a GOTO-style BLAS implementation

    :param loop: name of the loop
    :param operand: which block it operates on: 'A', 'B' or 'C'
    :param invariant: How many bytes are invariant
    :param streamers: bytes of one working instance of each operand that
                      changes across the loop, e.g. one A_r and one C_r for ir
    :param reuse: How many times the invariant is reused
    """
    loop: str
    operand: str
    invariant: float
    streamers: dict[str, float]
    reuse: float

    def need_at(self, level : cache_level, strm : dict[str, int]) -> float:
        """
        Footprint this rung needs at one level, given which operands are
        inserted there with a streaming hint.

        A streaming-inserted line goes in near the LRU end, so streamers evict
        each other rather than the invariant.  Each is then charged only the
        ways its prefetch window occupies -- normally one -- instead of its
        whole footprint.  That needs the associativity; without it this falls
        back to the plain-LRU footprint: the invariant plus one working
        instance of each streamer, which is also what an empty `strm` gives.

        :param level: the level being checked
        :param strm: ways each streaming-inserted operand occupies here
        :return: bytes that must fit
        """
        need = self.invariant
        for op, nbytes in self.streamers.items():
            if op in strm and level.assoc:
                need += min(nbytes, strm[op] * level.way)
            else:
                need += nbytes
        return need


@dataclass
class loop_nest:
    """
    The loops a blocking actually runs on one problem.

    A block can be no larger than its dimension, and a dimension that is not
    a multiple of its block leaves a partial last block, so each cache-block
    loop runs ceil(dim / block) times.  Counting dim / block instead is wrong
    at both ends: a block wider than the problem gives fewer than one pass
    over a matrix, and a block just short of it hides a whole extra pass.

    Footprints need the largest block, because that is what has to fit;
    per-iteration averages need the mean realised block, dim / trips.

    :param sizes: the problem
    :param k_c: largest k block, min(k_c, k)
    :param m_c: largest m block, min(m_c, m)
    :param n_c: largest n block, min(n_c, n)
    :param n_pc: pc trips, ceil(k / k_c)
    :param n_ic: ic trips, ceil(m / m_c)
    :param n_jc: jc trips, ceil(n / n_c)
    """
    sizes: problem
    k_c: int
    m_c: int
    n_c: int
    n_pc: int
    n_ic: int
    n_jc: int

    @property
    def k_mean(self) -> float:
        """
        :return: mean k block -- what one micro-kernel call does on average
        """
        return self.sizes.k / self.n_pc

    @property
    def m_mean(self) -> float:
        """
        :return: mean m block -- ir trips per ic block are m_mean / m_r
        """
        return self.sizes.m / self.n_ic

    @property
    def n_mean(self) -> float:
        """
        :return: mean n block -- jr trips per jc block are n_mean / n_r
        """
        return self.sizes.n / self.n_jc


# --------------------------------------------------------------- results ---

def _pick_phase(phase : str, ukr : float, pack : float) -> float:
    """
    Select the figure belonging to one phase.

    :param phase: 'ukr' or 'pack'
    :param ukr: value for the micro-kernel phase
    :param pack: value for the packing phase
    :return: the value for `phase`
    """
    if phase == 'ukr':
        return ukr
    if phase == 'pack':
        return pack
    raise ValueError(f"unknown phase {phase!r}; expected 'ukr' or 'pack'")


@dataclass
class stream:
    """
    One operand's access pattern within a micro-kernel call, as the hardware
    prefetcher sees it.

    :param operand: 'A', 'B' or 'C'
    :param count: independent streams per call
    :param lines: lines per stream, in the innermost level's line size
    """
    operand: str
    count: int
    lines: float


@dataclass
class traffic:
    """
    Bytes the load/store units move over the whole GEMM, before any reuse.

    Micro-kernel traffic is kept per operand so the goto ladder can divide each
    one by its own reuse.  Packing traffic is kept per operand too, because
    which of it crosses a boundary depends on which packed blocks are
    resident inside that boundary.

    :param rd: micro-kernel read bytes per operand
    :param wr: micro-kernel write bytes per operand
    :param pack_rd: packing reads of the source matrix per operand, read
                    amplification included
    :param pack_wr: packing writes into the packed buffer per operand
    :param rfo: whether each pack write first reads its line -- charged only
                where the packed buffer is not resident, since a resident
                one is written in place
    """
    rd: dict[str, float]
    wr: dict[str, float]
    pack_rd: dict[str, float]
    pack_wr: dict[str, float]
    rfo: bool


@dataclass
class boundary:
    """
    Traffic crossing one boundary: what `level` supplies inward, to the level
    inside it -- or to the core, for the innermost level.

    :param level: the level doing the supplying
    :param rd: micro-kernel read bytes per operand, each divided by the reuse
               of that operand's own rungs held inside this boundary
    :param wr: micro-kernel write bytes per operand, likewise
    :param pack_rd: packing read bytes crossing this boundary: the source,
                    plus the read-for-ownership of pack writes that miss
    :param pack_wr: packing write bytes crossing this boundary: write-backs
                    of packed buffers that are not resident inside it
    :param holds: rungs that become resident in this level.  They reduce the
                  traffic on the boundary outside it, not on this one.
    :param miss: innermost rung that does not fit this level, if any
    :param req_line: line size of the requester at this boundary -- the level
                     inside, whose fills are what cross it
    :param miss_out: fraction of this boundary's reads that also cross the
                     next boundary out
    :param waste: extra read bytes from prefetches that were evicted before
                  use, keyed by the index of the level they were issued into.
                  Real bandwidth, but carried only by that level's buffers and
                  those outside it.
    """
    level: cache_level
    rd: dict[str, float]
    wr: dict[str, float]
    pack_rd: float
    pack_wr: float
    holds: list[goto_rung]
    miss: goto_rung | None
    req_line: int
    miss_out: float = 0.0
    waste: dict[int, float] = field(default_factory=dict)

    @property
    def ukr_rd(self) -> float:
        """
        :return: micro-kernel read bytes, all operands together, including
                 any wasted prefetches
        """
        return sum(self.rd.values()) + sum(self.waste.values())

    @property
    def ukr_wr(self) -> float:
        """
        :return: micro-kernel write bytes, all operands together
        """
        return sum(self.wr.values())

    @property
    def tot_rd(self) -> float:
        """
        :return: read bytes over both phases
        """
        return self.ukr_rd + self.pack_rd

    @property
    def tot_wr(self) -> float:
        """
        :return: write bytes over both phases
        """
        return self.ukr_wr + self.pack_wr

    @property
    def fills(self) -> float:
        """
        :return: lines the requester pulls across this boundary, counted in
                 the requester's own line size
        """
        return self.tot_rd / self.req_line

    def phase_rd(self, phase : str) -> float:
        """
        :param phase: 'ukr' or 'pack'
        :return: read bytes crossing this boundary in that phase
        """
        return _pick_phase(phase, self.ukr_rd, self.pack_rd)

    def phase_wr(self, phase : str) -> float:
        """
        :param phase: 'ukr' or 'pack'
        :return: write bytes crossing this boundary in that phase
        """
        return _pick_phase(phase, self.ukr_wr, self.pack_wr)


@dataclass
class analysis:
    """
    Everything analyse() derives for one configuration.  It carries its own
    inputs, so the functions that consume it need nothing else.

    :param kernel: microkernel model
    :param blocks: blocksizes
    :param sizes: problem sizes
    :param pk: packing model used
    :param pf: hardware prefetcher model used
    :param write_allocate: whether a store that misses fetches the line first
    :param core: pseudo-level standing for the core's load/miss queue
    :param rungs: the goto ladder for this blocking
    :param boundaries: one per level, innermost first
    :param c_layout: layout of C, 'rm' or 'cm'
    :param streams: per-operand access patterns within a micro-kernel call
    :param cov: fraction of each operand's fills the prefetcher starts early
    :param pack_elems: elements packed over the whole GEMM, per operand
    :param flops: total FLOP, k-loop plus epilogue
    :param calls: micro-kernel invocations
    :param t_ukr: ideal (compute-bound) micro-kernel cycles
    :param t_pack: ideal packing cycles
    :param sw: software prefetches requested
    :param strm: ways each streaming-inserted operand occupies, per level
    :param evals: one evaluation per software prefetch
    :param pf_cover: combined prefetch coverage, by operand and then by the
                     name of the level filled into -- hardware and software
                     together
    """
    kernel: ukernel
    blocks: blocking
    sizes: problem
    pk: packing
    pf: hw_prefetcher
    write_allocate: bool
    core: cache_level
    rungs: list[goto_rung]
    boundaries: list[boundary]
    c_layout: str
    streams: list[stream]
    cov: dict[str, float]
    pack_elems: dict[str, float]
    flops: float
    calls: float
    t_ukr: float
    t_pack: float
    sw: list[sw_prefetch] = field(default_factory=list)
    strm: dict[str, dict[str, int]] = field(default_factory=dict)
    evals: list[prefetch_eval] = field(default_factory=list)
    pf_cover: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def nest(self) -> loop_nest:
        """
        :return: the loops this configuration's blocking runs on its problem
        """
        return make_loop_nest(self.blocks, self.sizes)

    @property
    def t_ideal(self) -> float:
        """
        :return: ideal cycles for the whole run, both phases in series
        """
        return self.t_ukr + self.t_pack

    @property
    def core_rd(self) -> float:
        """
        :return: bytes the core reads from the innermost level, both phases
        """
        return self.boundaries[0].tot_rd

    @property
    def core_wr(self) -> float:
        """
        :return: bytes the core writes to the innermost level, both phases
        """
        return self.boundaries[0].tot_wr


@dataclass
class prefetch_eval:
    """
    What one software prefetch does to this configuration, and why.

    :param pf: the prefetch evaluated
    :param target_index: index of the level it fills, or None if invalid
    :param instance: bytes of one prefetched instance -- what the next
                     iteration of the loop needs of the operand
    :param lead: cycles from issue to use: one iteration of the loop, at the
                 ideal rate.  The real loop is slower, so this is a lower
                 bound and the timeliness estimate is conservative.
    :param exposed: bytes of the operand arriving from beyond the target,
                    i.e. what there is to hide
    :param latency: expected latency of one of those fills
    :param timely: fraction of that latency the lead time covers
    :param free: capacity left at the target after the resident set
    :param coverage: fraction of the exposed fills that arrive in time
    :param resident: whether the instance is already resident at the target
    :param evicted: whether the instance cannot survive until it is used
    :param wasted: read bytes fetched for nothing, because of eviction
    :param hw_cover: fraction the hardware prefetcher already covers here
    :param relieves: holders whose buffers it relieves -- those inside the
                     target, and only of the latency beyond it
    :param newly_resident: rungs that fit only because of its 'strm' hint
    :param spills: whether a 'strm' window exceeds the one way it is given
    :param instr_per_kstep: prefetch instructions it adds per k-step; not
                            charged in the prediction
    :param notes: why it does or does not help
    """
    pf: sw_prefetch
    target_index: int | None = None
    instance: float = 0.0
    lead: float = 0.0
    exposed: float = 0.0
    latency: float = 0.0
    timely: float = 0.0
    free: float = 0.0
    coverage: float = 0.0
    resident: bool = False
    evicted: bool = False
    wasted: float = 0.0
    hw_cover: float = 0.0
    relieves: list[str] = field(default_factory=list)
    newly_resident: list[goto_rung] = field(default_factory=list)
    spills: bool = False
    instr_per_kstep: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class buffer_load:
    """
    Cycles one holder's outstanding-request structures need to cover a phase.
    Each figure is a lower bound on that phase's duration; dividing it by the
    actual phase time gives the fraction of the phase the structure is busy.

    :param holder: 'core' or a level name
    :param rd_cycles: cycles the read buffers need -- or, for a shared pool,
                      the reads and writes together.  None if not modelled.
    :param wr_cycles: cycles the write buffers need.  None for a shared pool,
                      whose writes are already in rd_cycles, or if not
                      modelled.
    :param shared: whether reads and writes share one pool
    """
    holder: str
    rd_cycles: float | None
    wr_cycles: float | None
    shared: bool


@dataclass
class phase_result:
    """
    Roofline outcome for one phase.

    :param time: cycles the phase takes: the largest of its roofline terms
    :param limiter: which term that was
    :param tied: other terms within TIE of it -- relieving the limiter alone
                 then buys at most that much
    """
    time: float
    limiter: str
    tied: list[str] = field(default_factory=list)


@dataclass
class prediction:
    """
    Predicted run time: the two phases in series, since packing and the
    micro-kernel do not overlap in single-threaded BLIS.

    :param ukr: micro-kernel phase
    :param pack: packing phase
    :param flops: total FLOP, for the throughput figure
    """
    ukr: phase_result
    pack: phase_result
    flops: float

    @property
    def time(self) -> float:
        """
        :return: predicted cycles for the whole run
        """
        return self.ukr.time + self.pack.time

    @property
    def flop_per_cycle(self) -> float:
        """
        :return: predicted throughput
        """
        return self.flops / self.time

    @property
    def limiter(self) -> str:
        """
        :return: the limiter of whichever phase dominates
        """
        if self.ukr.time >= self.pack.time:
            return self.ukr.limiter
        return (f'{self.pack.limiter} '
                f'(pack {100 * self.pack.time / self.time:.0f}% of run)')

    @property
    def tied(self) -> list[str]:
        """
        :return: terms tied with the dominant phase's limiter
        """
        return (self.ukr.tied if self.ukr.time >= self.pack.time
                else self.pack.tied)

    @property
    def bound(self) -> str:
        """
        :return: the limiter of whichever phase dominates, without the share
                 annotation `limiter` adds -- so two predictions compare
        """
        return (self.ukr.limiter if self.ukr.time >= self.pack.time
                else self.pack.limiter)


@dataclass
class pmu_prediction:
    """
    Predicted PMU event counts, for checking the model against `perf stat`.

    :param events: predicted count per ARM PMU event name
    :param ksteps: k-steps over the whole GEMM, for per-k-step figures
    :param calls: micro-kernel calls, for per-call figures
    :param pred: the timing prediction CPU_CYCLES comes from
    """
    events: dict[str, float]
    ksteps: float
    calls: float
    pred: prediction


def stream_shape(u : ukernel, b : blocking, p : problem, line : int,
                 c_layout : str ='rm') -> list[stream]:
    """
    Per-operand access pattern within one micro-kernel call.

    A_r and B_r are single contiguous packed panels: one long stream each.
    C is m_r short runs at stride rs_c (or n_r at cs_c), restarting at an
    unrelated address on every call -- which is why no prefetcher follows it.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes, which bound a call's k block
    :param line: cache line size
    :param c_layout: - rm: C is row-major
                     - cm: C is column-major
    :return: one stream per operand
    """
    k_c = make_loop_nest(b, p).k_c
    out = [stream('A', 1, k_c * u.m_r * u.size('A') / line),
           stream('B', 1, k_c * u.n_r * u.size('B') / line)]
    if c_layout == 'rm':
        out.append(stream('C', u.m_r, max(1.0, u.n_r * u.size('C') / line)))
    else:
        out.append(stream('C', u.n_r, max(1.0, u.m_r * u.size('C') / line)))
    return out


def coverage(shape : list[stream], pf : hw_prefetcher) -> dict[str,float]:
    """
    Fraction of each operand's fills the prefetcher gets to ahead of time.
    Trackers go to the longest runs first.

    :param shape: list of per-operand stream shapes
    :param pf: hardware prefetcher model
    :return: prefetch coverage per operand
    """
    cov, budget = {}, pf.streams
    for st in sorted(shape, key=lambda st: -st.lines):
        if st.lines < pf.min_run or budget <= 0:
            cov[st.operand] = 0.0
        else:
            take = min(st.count, budget)
            cov[st.operand] = take / st.count
            budget -= take
    return cov


# ----------------------------------------------------------------- model ---

def make_loop_nest(b : blocking, p : problem) -> loop_nest:
    """
    Clamp each block to its dimension and count whole trips.

    :param b: blocksizes as requested
    :param p: problem sizes
    :return: the loops that blocking runs on that problem
    """
    def trips(dim, blk):
        return -(-dim // blk)
    return loop_nest(p, min(b.k_c, p.k), min(b.m_c, p.m), min(b.n_c, p.n),
                     trips(p.k, b.k_c), trips(p.m, b.m_c), trips(p.n, b.n_c))


def make_goto_ladder(u : ukernel, b : blocking, p : problem) -> list[goto_rung]:
    """
    Creates a ladder modeling the 5 outer loops of a GOTO-style GEMM implementation

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :return: list of goto_rung objects for each loop
    """
    t = make_loop_nest(b, p)
    sa, sb, sc = u.size('A'), u.size('B'), u.size('C')
    A_r = t.k_c * u.m_r * sa
    B_r = t.k_c * u.n_r * sb
    C_r = u.m_r * u.n_r * sc
    A_c = t.m_c * t.k_c * sa
    B_c = t.k_c * t.n_c * sb
    C_p = p.m * t.n_c * sc
    A_f = p.m * p.k * sa
    return [
        goto_rung('ir', 'B', B_r, {'A': A_r, 'C': C_r}, t.m_mean / u.m_r),
        goto_rung('jr', 'A', A_c, {'B': B_r, 'C': C_r}, t.n_mean / u.n_r),
        goto_rung('ic', 'B', B_c, {'A': A_c, 'C': C_r}, t.n_ic),
        goto_rung('pc', 'C', C_p, {'A': A_c, 'B': B_c}, t.n_pc),
        goto_rung('jc', 'A', A_f, {'B': B_c, 'C': C_p}, t.n_jc),
    ]


def pack_work(u : ukernel, b : blocking, p : problem) -> dict[str,float]:
    """
    Elements packed over the whole GEMM, per operand.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :return: number of elements packed
    """
    return {
        'A': float(make_loop_nest(b, p).n_jc) * p.m * p.k,  # per (jc,pc,ic)
        'B': p.k * p.n,                      # once per (jc,pc)
    }


def packed_elems(u : ukernel, b : blocking, p : problem,
                 pk : packing) -> dict[str,float]:
    """
    Elements packed inside the modelled region: pack_work(), less the
    operands packed outside it.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :param pk: packing model
    :return: elements packed per operand
    """
    if set(pk.outside) - {'A', 'B'}:
        raise ValueError(f"packing outside {pk.outside!r}: only A and B are packed")
    return {op: 0.0 if op in pk.outside else v
            for op, v in pack_work(u, b, p).items()}


def core_traffic(u : ukernel, b : blocking, p : problem,
                 pk : packing) -> traffic:
    """
    Bytes the load/store units move over the whole GEMM.  ukr streams are
    keyed by operand so the ladder can divide them; the packing streams are
    one-pass and never amortise, so they are kept apart.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :param pk: packing model
    :return: bytes moved over the whole GEMM, before any reuse
    """
    t = make_loop_nest(b, p)
    calls = (p.m / u.m_r) * (p.n / u.n_r) * t.n_pc
    w = packed_elems(u, b, p, pk)
    rd = {
        'A': calls * t.k_mean * u.m_r * u.size('A'),
        'B': calls * t.k_mean * u.n_r * u.size('B'),
        'C': calls * u.m_r * u.n_r * u.size('C') if u.c_read else 0.0,
    }
    wr = {'C': calls * u.m_r * u.n_r * u.size('C')}
    pack_rd = {'A': w['A'] * u.size('A') * pk.amp_a,
               'B': w['B'] * u.size('B') * pk.amp_b}
    pack_wr = {'A': w['A'] * u.size('A'), 'B': w['B'] * u.size('B')}
    return traffic(rd, wr, pack_rd, pack_wr, pk.rfo)


def _fmt_size(n : float) -> str:
    """
    :param n: a capacity in bytes, or inf
    :return: it in the largest binary unit that keeps it short: 32K, 768K, 4G
    """
    if n == float('inf'):
        return 'inf'
    for unit, scale in (('G', 2**30), ('M', 2**20), ('K', 2**10)):
        if n >= scale:
            return f"{n/scale:g}{unit}"
    return f"{n:g}B"


def _fmt_bytes(n : float) -> str:
    """
    :param n: a size in bytes
    :return: it in B, K or M, whichever reads best
    """
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024 ** 2:
        return f"{n/1024:.0f}K" if n >= 10 * 1024 else f"{n/1024:.1f}K"
    return f"{n/1024**2:.1f}M"


# two roofline terms within this fraction of each other are reported as tied
TIE = 0.005

# The loop whose invariant is each packed block.  When that rung is held
# inside a boundary, the packed block is resident there.
PACKED = {'A': 'jr', 'B': 'ic'}

# loops of the nest, innermost first; 'k' is the k-loop inside the ukr
LOOPS = ('k', 'ir', 'jr', 'ic', 'pc', 'jc')

# operands that change across each loop -- the only ones there is anything
# new to prefetch for.  The rest is the loop's invariant (C lives in
# registers for the whole k-loop).
VARYING = {'k': {'A', 'B'}, 'ir': {'A', 'C'}, 'jr': {'B', 'C'},
           'ic': {'A', 'C'}, 'pc': {'A', 'B'}, 'jc': {'B', 'C'}}


def loop_period(an : analysis, loop : str) -> float:
    """
    Cycles one iteration of a loop takes, at the ideal (compute-bound) rate.
    The real loop is slower, so leads built from this are lower bounds.

    :param an: analysis of the configuration
    :param loop: 'k', 'ir', 'jr', 'ic', 'pc' or 'jc'
    :return: cycles per iteration
    """
    u, t = an.kernel, an.nest
    t_call = u.time(t.k_mean)
    t_jr = (t.m_mean / u.m_r) * t_call
    t_ic = (t.n_mean / u.n_r) * t_jr
    t_pc = t.n_ic * t_ic
    return {'k': 2 * u.m_r * u.n_r / u.peak, 'ir': t_call, 'jr': t_jr,
            'ic': t_ic, 'pc': t_pc, 'jc': t.n_pc * t_pc}[loop]


def loop_lead(an : analysis, sp : sw_prefetch) -> float:
    """
    Cycles from issuing a prefetch to the data being used: `distance` k-steps
    if given, otherwise one iteration of its loop -- and never more than one
    iteration, since it fetches only the next one.

    :param an: analysis of the configuration
    :param sp: the prefetch
    :return: lead time in cycles
    """
    t_k = loop_period(an, 'k')
    if sp.loop == 'k':
        return max(1, sp.distance) * t_k
    period = loop_period(an, sp.loop)
    return min(sp.distance * t_k, period) if sp.distance else period


def _burst_timeliness(an : analysis, ev : prefetch_eval,
                      level : cache_level) -> float:
    """
    Fraction of the latency a burst of prefetches hides, line by line.

    The burst's lines wait for the target's fill buffers: line i is issued
    floor(i / M) waves of one latency late.  It is needed at the lead, plus
    however much later the micro-kernel reads it: a C tile is read whole by
    the epilogue, one tile per call, while A and B are read progressively at
    the k-loop's rate.

    :param an: analysis of the configuration
    :param ev: the prefetch's evaluation, with lead, latency and instance
    :param level: the level it fills; without buffers modelled, no queueing
    :return: mean fraction hidden over the burst's lines
    """
    sp, u, lat = ev.pf, an.kernel, ev.latency
    n = max(1, int(ev.instance / level.line))
    if sp.operand == 'C':
        tile = max(1, int(u.m_r * u.n_r * u.size('C') / level.line))
        def offset(i):
            return (i // tile) * loop_period(an, 'ir')
    else:
        per_k = (u.m_r if sp.operand == 'A' else u.n_r) * u.size(sp.operand)
        step = loop_period(an, 'k') * level.line / per_k
        def offset(i):
            return i * step
    idx = range(n) if n <= 4096 else [round(j * (n - 1) / 4095)
                                      for j in range(4096)]
    def hidden(i):
        wait = (i // level.mshr) * lat if level.mshr else 0.0
        return min(1.0, max(0.0, ev.lead + offset(i) - wait) / lat)
    return sum(hidden(i) for i in idx) / len(idx)


def prefetch_instance(u : ukernel, b : blocking, p : problem,
                      sp : sw_prefetch) -> float:
    """
    Bytes one prefetch fetches: what the next iteration of its loop needs of
    its operand.  For loop 'k' this is also the window of lines it keeps in
    flight ahead of use.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :param sp: the prefetch
    :return: bytes in one prefetched instance
    """
    t = make_loop_nest(b, p)
    d = max(1, sp.distance)
    elems = {('k', 'A'): d * u.m_r,
             ('k', 'B'): d * u.n_r,
             ('ir', 'A'): t.k_c * u.m_r, ('ir', 'C'): u.m_r * u.n_r,
             ('jr', 'B'): t.k_c * u.n_r, ('jr', 'C'): t.m_c * u.n_r,
             ('ic', 'A'): t.m_c * t.k_c, ('ic', 'C'): t.m_c * t.n_c,
             ('pc', 'A'): p.m * t.k_c,   ('pc', 'B'): t.k_c * t.n_c,
             ('jc', 'B'): p.k * t.n_c,   ('jc', 'C'): p.m * t.n_c,
             }[(sp.loop, sp.operand)]
    return elems * u.size(sp.operand)


def evaluate_prefetch(an : analysis, sp : sw_prefetch) -> prefetch_eval:
    """
    Decide what one software prefetch does, and record why.

    In order: the target must be a real cache, and the operand must change
    across the loop.  Then it does nothing if the instance is already
    resident at the target, or if no traffic of that operand comes from
    beyond it.  Otherwise it hides the beyond-target latency for the holders
    inside the target -- in proportion to how much of that latency its lead
    time covers, and not at all if the instance cannot survive until use.
    A 'strm' hint is assessed separately, for its effect on capacity.

    :param an: analysis of the configuration, before any prefetch is applied
    :param sp: the prefetch
    :return: its evaluation
    """
    bds = an.boundaries
    idx = {bd.level.name: i for i, bd in enumerate(bds)}
    ev = prefetch_eval(sp)
    if sp.loop not in LOOPS:
        ev.notes.append(f"unknown loop {sp.loop!r}; expected one of "
                        f"{', '.join(LOOPS)}")
        return ev
    if sp.target not in idx or idx[sp.target] == len(bds) - 1:
        ev.notes.append(f"{sp.target!r} is not a cache level that can be "
                        f"prefetched into")
        return ev
    if sp.operand not in VARYING[sp.loop]:
        ev.notes.append(f"{sp.operand} is invariant across {sp.loop}: there "
                        f"is nothing new to fetch")
        return ev

    t = idx[sp.target]
    L = bds[t].level
    strm = an.strm.get(L.name, {})
    ev.target_index = t
    ev.instance = prefetch_instance(an.kernel, an.blocks, an.sizes, sp)
    ev.lead = loop_lead(an, sp)
    ev.relieves = ['core'] + [bds[i].level.name for i in range(t)]

    # capacity effect of a streaming hint -- independent of latency
    if sp.hint == 'strm':
        if not L.assoc:
            ev.notes.append(f"'strm' has no capacity effect: associativity "
                            f"of {L.name} is unknown")
        else:
            without = {op: w for op, w in strm.items() if op != sp.operand}
            ev.newly_resident = [r for r in an.rungs
                                 if r.need_at(L, strm) <= L.cap
                                 < r.need_at(L, without)]
            ev.spills = ev.instance > L.way
            if ev.newly_resident:
                got = ', '.join(f"{r.loop}:{r.operand}"
                                for r in ev.newly_resident)
                ev.notes.append(f"capacity: streaming {sp.operand} makes "
                                f"{got} resident in {L.name}")
            else:
                cand = [r for r in an.rungs if sp.operand in r.streamers]
                fit = [r for r in cand if r.need_at(L, without) <= L.cap]
                if fit:
                    got = ', '.join(f"{r.loop}:{r.operand}" for r in fit)
                    ev.notes.append(f"capacity: {got} already fits {L.name} "
                                    f"without the hint, so it adds nothing")
                elif cand:
                    r = min(cand, key=lambda r: r.need_at(L, strm))
                    ev.notes.append(
                        f"capacity: nothing newly fits {L.name} -- closest is "
                        f"{r.loop}:{r.operand}, needing "
                        f"{_fmt_bytes(r.need_at(L, strm))} with the hint "
                        f"({_fmt_bytes(r.need_at(L, without))} without) of "
                        f"{_fmt_bytes(L.cap)}")
            if ev.spills:
                ev.notes.append(f"the {_fmt_bytes(ev.instance)} window needs "
                                f"{strm.get(sp.operand, 1)} ways of "
                                f"{_fmt_bytes(L.way)}, not one -- shorten the "
                                f"distance, or it eats the capacity the hint "
                                f"was meant to free")

    # instruction cost, reported but not charged
    ks = an.sizes.m * an.sizes.n * an.sizes.k / (an.kernel.m_r * an.kernel.n_r)
    n_iter = (ks / max(1, sp.distance) if sp.loop == 'k'
              else an.t_ukr / loop_period(an, sp.loop))
    ev.instr_per_kstep = ev.instance / L.line * n_iter / ks

    # already resident: part of an invariant held at the target by an
    # enclosing loop, so the lines are there before the prefetch is issued
    for r in an.rungs:
        if (r.operand == sp.operand
                and LOOPS.index(r.loop) > LOOPS.index(sp.loop)
                and r.need_at(L, strm) <= L.cap):
            ev.resident = True
            ev.notes.append(f"already resident in {L.name}, as part of the "
                            f"{r.loop} loop's invariant "
                            f"({_fmt_bytes(r.invariant)}): nothing to hide")
            return ev

    ev.exposed = bds[t + 1].rd[sp.operand]
    if ev.exposed <= 0:
        packed = (sp.operand in PACKED
                  and any(r.loop == PACKED[sp.operand]
                          and r.need_at(L, strm) <= L.cap for r in an.rungs))
        why = (f"the packed {sp.operand}_c is resident in {L.name}, so the "
               f"micro-kernel reads no {sp.operand} from beyond it: it "
               f"arrives through packing" if packed else
               f"no {sp.operand} traffic comes from beyond {L.name}")
        ev.notes.append(f"{why} -- nothing to hide")
        return ev
    ev.latency = sum(bds[j].rd[sp.operand] / ev.exposed * bds[j].level.lat
                     for j in range(t + 1, len(bds)))
    ev.timely = min(1.0, ev.lead / ev.latency) if ev.latency else 1.0
    if sp.loop != 'k' and ev.latency:
        plain = ev.timely
        ev.timely = _burst_timeliness(an, ev, L)
        if ev.timely < plain - 0.005:
            n = int(ev.instance / L.line)
            how = ("the epilogue reads a tile all at once" if sp.operand == 'C'
                   else "they are read faster than the buffers deliver them")
            ev.notes.append(
                f"burst-limited: its {n} lines go through {L.mshr} {L.name} "
                f"fill buffers in {-(-n // L.mshr)} waves of "
                f"{ev.latency:.0f} cy, and {how} -- {ev.lead:.0f} cy of lead "
                f"hides {100*ev.timely:.0f}% of the latency")
    if sp.distance and sp.loop != 'k' and \
            sp.distance * loop_period(an, 'k') > loop_period(an, sp.loop):
        ev.notes.append(f"lead capped at one {sp.loop} iteration "
                        f"({loop_period(an, sp.loop):.0f} cy): it fetches "
                        f"only the next one")

    held = [r.need_at(L, strm) for r in an.rungs
            if r.need_at(L, strm) <= L.cap]
    ev.free = L.cap - (max(held) if held else 0.0)
    # a 'strm' line lives inside the way reserved for it, which is its whole
    # capacity check (see `spills`); only a normally inserted line has to fit
    # in what the resident set leaves over
    strm_ok = sp.hint == 'strm' and L.assoc
    if not strm_ok and ev.instance > ev.free:
        ev.evicted = True
        ev.notes.append(f"evicted before use: the next instance "
                        f"({_fmt_bytes(ev.instance)}) does not fit in what "
                        f"{L.name} has free ({_fmt_bytes(ev.free)}), so it is "
                        f"fetched twice")
    elif ev.lead < ev.latency:
        if any(n.startswith('burst') for n in ev.notes):
            ev.notes.append(f"late as well: {ev.lead:.0f} cycles of lead "
                            f"against {ev.latency:.0f} of latency, so even the "
                            f"first wave arrives late")
        else:
            ev.notes.append(f"late: {ev.lead:.0f} cycles of lead against "
                            f"{ev.latency:.0f} of latency hides only "
                            f"{100*ev.timely:.0f}% of it")
    ev.coverage = 0.0 if ev.evicted else ev.timely

    if t == 0:
        tail = ("" if an.core.mshr else
                ", which is not modelled here (--core-mshr)")
        ev.notes.insert(0, f"{L.name} is the innermost level: its own fill "
                           f"buffers still carry the fetch, so its latency "
                           f"benefit goes only to the core's load "
                           f"queue{tail}")
    # a hardware prefetcher filling the same level relieves exactly the same
    # holders, so what this one adds is its coverage of what is left over
    if an.pf.level == L.name and an.cov.get(sp.operand, 0.0) > 0:
        ev.hw_cover = an.cov[sp.operand]
        adds = ev.coverage * (1 - ev.hw_cover)
        if ev.coverage and adds < 0.005:
            ev.notes.insert(0, f"redundant: the hardware prefetcher already "
                               f"covers {100*ev.hw_cover:.0f}% of "
                               f"{sp.operand} into {L.name}, so this adds no "
                               f"latency coverage")
        else:
            ev.notes.append(f"the hardware prefetcher covers "
                            f"{100*ev.hw_cover:.0f}% of {sp.operand} into "
                            f"{L.name}; this adds {100*adds:.0f}% on top")
    return ev


def _apply_prefetch(an : analysis):
    """
    Record wasted traffic from evicted prefetches on the boundaries it
    crosses, then combine every prefetch's coverage -- hardware and software
    -- by operand and target level.

    Wasted traffic is snapshotted before any of it is applied, so that two
    evicted prefetches of the same operand do not compound.

    :param an: analysis whose evaluations are already filled in
    """
    bds = an.boundaries
    base = [dict(bd.rd) for bd in bds]
    for ev in an.evals:
        if ev.evicted:
            t, op = ev.target_index, ev.pf.operand
            for j in range(t + 1, len(bds)):
                bds[j].waste[t] = bds[j].waste.get(t, 0.0) + base[j][op]
                ev.wasted += base[j][op]

    cover = {}
    if an.pf.level:
        for op, c in an.cov.items():
            if c > 0:
                cover.setdefault(op, {})[an.pf.level] = c
    for ev in an.evals:
        if ev.coverage > 0:
            per = cover.setdefault(ev.pf.operand, {})
            prev = per.get(ev.pf.target, 0.0)
            per[ev.pf.target] = 1 - (1 - prev) * (1 - ev.coverage)
    an.pf_cover = cover


def _kept(an : analysis, op : str, holder : int, j : int,
          idx : dict[str, int]) -> float:
    """
    Fraction of an operand's latency term that a holder still pays at one
    boundary, after prefetching.

    A prefetch into level t relieves holder `holder` of boundary j's latency
    only if the holder is inside the target and the boundary is beyond it:
    holder < t < j.

    :param an: analysis of the configuration
    :param op: operand
    :param holder: holder index; -1 for the core, otherwise its level index
    :param j: boundary index
    :param idx: level index by name
    :return: remaining fraction, 1.0 if nothing applies
    """
    kept = 1.0
    for target, c in an.pf_cover.get(op, {}).items():
        if holder < idx[target] < j:
            kept *= 1 - c
    return kept


def _pack_crossing(tr : traffic, resident : set[str],
                   a_reuse : float) -> tuple[float, float]:
    """
    Packing traffic crossing one boundary.  Private, and unpacked on the spot.

    :param tr: traffic before any reuse
    :param resident: operands whose packed buffer is resident inside the
                     boundary -- their writes never cross it
    :param a_reuse: packs of A that share one read of its source here: n_jc
                    if the whole of A is resident inside, else 1
    :return: (read bytes, write bytes) crossing the boundary
    """
    wr = sum(v for op, v in tr.pack_wr.items() if op not in resident)
    src = {op: v / (a_reuse if op == 'A' else 1)
           for op, v in tr.pack_rd.items()}
    return sum(src.values()) + (wr if tr.rfo else 0.0), wr


def analyse(u : ukernel, b : blocking, p : problem, levels : list[cache_level],
            pk : packing=None, write_allocate : bool=True, core_mshr : int=0,
            pf : hw_prefetcher=None, c_layout : str='rm',
            sw : list[sw_prefetch]=None) -> analysis:
    """
    Traffic crossing every boundary, bottom up.  Boundary i sits between
    level i and level i-1 (the core for i=0).  What crosses it is worked out
    per operand: that operand's core-side traffic, divided by the reuse of
    its own rungs held strictly inside -- a rung's reuse applies to its
    invariant operand only, never to the operands streaming past it.  Beyond
    a resident packed block, the micro-kernel's pass over that operand does
    not cross at all; packing traffic is handled apart, as the module
    docstring describes under PACKING.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :param levels: cache/memory levels
    :param pk: packing model
    :param write_allocate: whether writes cause RFO
    :param core_mshr: mshr for reading from L1
    :param pf: hw prefetcher model
    :param c_layout: - rm: C is row-major
                     - cm: C is column-major
    :param sw: software prefetches; their 'strm' hints change residency here,
               and their latency effect is applied once the traffic is known
    :return: the analysis of this configuration
    """
    pk = pk or packing()
    pf = pf or hw_prefetcher()
    sw = sw or []
    levels = sorted(levels, key=lambda L: L.size)
    caches = [L.name for L in levels if L.size != float('inf')]
    if pf.level and pf.level not in caches:
        raise ValueError(f"the hardware prefetcher cannot fill "
                         f"{pf.level!r}: the cache levels are "
                         f"{', '.join(caches)}")
    shape = stream_shape(u, b, p, levels[0].line, c_layout)
    # the core's load queue tracks loads only; stores go through the store
    # queue, so it is a read-only holder
    core = cache_level('core', 0.0, mshr=core_mshr, shared_buf=False)
    rungs = make_goto_ladder(u, b, p)
    tr = core_traffic(u, b, p, pk)
    # operands a 'strm' hint inserts near LRU, per level, with the ways each
    # one's window occupies -- this changes what fits, so it has to be known
    # before residency is decided
    strm = {L.name: {} for L in levels}
    for sp in sw:
        L = next((L for L in levels if L.name == sp.target), None)
        if (sp.hint == 'strm' and L is not None and L.assoc
                and sp.loop in LOOPS and sp.operand in VARYING[sp.loop]):
            ways = max(1, -(-prefetch_instance(u, b, p, sp) // L.way))
            strm[L.name][sp.operand] = max(strm[L.name].get(sp.operand, 0),
                                           int(ways))

    t = make_loop_nest(b, p)
    flops = 2 * p.m * p.n * p.k + u.epi_flops * p.m * p.n * t.n_pc
    calls = (p.m / u.m_r) * (p.n / u.n_r) * t.n_pc
    w = packed_elems(u, b, p, pk)
    t_ukr = calls * u.time(t.k_mean)
    t_pack = ((w['A'] / pk.rate_a if pk.rate_a else 0.0)
              + (w['B'] / pk.rate_b if pk.rate_b else 0.0))

    bounds, inner, fits = [], [], []
    for li, L in enumerate(levels):
        # the requester at this boundary -- its line size sets fill granularity
        req = levels[li - 1] if li else L
        reuse = {'A': 1.0, 'B': 1.0, 'C': 1.0}
        for r in inner:
            reuse[r.operand] *= r.reuse
        rd = {op: v / reuse[op] for op, v in tr.rd.items()}
        wr = {op: v / reuse[op] for op, v in tr.wr.items()}
        # A packed block resident inside this boundary is read from there by
        # the micro-kernel, and the next block is packed over it in place: so
        # neither the micro-kernel's pass over that operand nor the buffer's
        # write-back crosses this boundary.  The pack's read of the source
        # still does -- once per jc trip, or once in all if the whole of A is
        # resident inside too.
        held = {r.loop for r in inner}
        resident = {op for op, loop in PACKED.items() if loop in held}
        for op in resident:
            rd[op] = 0.0
        pk_rd, pk_wr = _pack_crossing(tr, resident,
                                      t.n_jc if 'jc' in held else 1)
        # A store that misses pulls the line in first, unless C is read anyway.
        # That fetch is issued by the cache, not the core, so it never crosses
        # the innermost boundary.
        if write_allocate and not u.c_read and li:
            rd['C'] = wr['C']

        if L.size == float('inf'):
            holds, miss = [r for r in rungs if r not in inner], None
        else:
            fits = [r for r in rungs if r.need_at(L, strm[L.name]) <= L.cap]
            holds = [r for r in fits if r not in inner]
            nofit = [r for r in rungs if r.need_at(L, strm[L.name]) > L.cap]
            miss = nofit[0] if nofit else None

        bounds.append(boundary(L, rd, wr, pk_rd, pk_wr, holds, miss,
                               req.line))
        if L.size != float('inf'):
            inner = fits

    # what fraction of the fills a level supplies have to continue outward
    for bd, nxt in zip(bounds, bounds[1:]):
        bd.miss_out = nxt.tot_rd / bd.tot_rd if bd.tot_rd else 0.0

    an = analysis(kernel=u, blocks=b, sizes=p, pk=pk, pf=pf,
                  write_allocate=write_allocate, core=core, rungs=rungs,
                  boundaries=bounds, c_layout=c_layout, streams=shape,
                  cov=coverage(shape, pf), pack_elems=w, flops=flops,
                  calls=calls, t_ukr=t_ukr, t_pack=t_pack, sw=sw, strm=strm)
    # the latency effect needs the traffic, so it comes last
    an.evals = [evaluate_prefetch(an, sp) for sp in sw]
    _apply_prefetch(an)
    return an


def occupancy(an : analysis, phase : str='ukr') -> list[buffer_load]:
    """
    Cycles each holder's outstanding-request structures need to cover one
    phase, innermost holder first.

    READS chain.  A fill holds its buffer for the whole round trip, so a line
    that misses two levels pays both latencies:

        t >= ( N_rd(i)*L(i) + N_rd(i+1)*L(i+1) + ... ) / M_rd(i-1)

    WRITES do not.  A dirty eviction frees the inner buffer as soon as the
    next level accepts it, so each level's write buffer only sees its own
    eviction stream:

        t >= N_wr(i)*L(i) / M_wr(i-1)

    Evaluated per phase, because pack traffic is not spread over the whole
    run -- it all happens inside t_pack.  A buffer holds one of its holder's
    own lines, so counts use the holder's line size.

    Prefetching scales the read terms: a prefetch into level t relieves a
    holder inside t of every boundary beyond t, for the operand it covers.
    Prefetches evicted before use add their wasted fetches back, but only to
    the target's buffers and those outside it -- an inner holder never saw
    that request.

    :param an: analysis of the configuration
    :param phase: 'ukr' or 'pack'
    :return: one buffer_load per holder -- the core, then every cache level
    """
    out = []
    for name, L, rnum, wnum in _buffer_cycles(an, phase):
        if L.shared_buf and L.mshr:
            # one pool: reads and writes compete for the same slots
            out.append(buffer_load(name, (rnum + wnum) / L.mshr, None, True))
        else:
            out.append(buffer_load(name,
                                   rnum / L.mshr if L.mshr else None,
                                   wnum / L.wbuf if L.wbuf else None, False))
    return out


def _buffer_cycles(an : analysis, phase : str='ukr'):
    """
    The numerators of occupancy(): buffer-cycles each holder's read and write
    buffers are occupied for over one phase, i.e. requests in flight summed
    over the phase's cycles -- what an occupancy counter integrates.
    :return: (holder name, holder, read buffer-cycles, write buffer-cycles)
             per holder, the core first, then every cache level
    """
    bds = an.boundaries
    idx = {bd.level.name: i for i, bd in enumerate(bds)}
    # the core is served from the innermost level out, in that level's lines;
    # holder index -1 is the core, otherwise the level whose fills it holds
    holders = [('core', an.core, -1, bds[0].level.line)]
    holders += [(bd.level.name, bd.level, i, bd.level.line)
                for i, bd in enumerate(bds[:-1])]
    out = []
    for name, L, hi, line in holders:
        first = hi + 1
        rnum = 0.0
        for j in range(first, len(bds)):
            if phase == 'ukr':
                nb = sum(v * _kept(an, op, hi, j, idx)
                         for op, v in bds[j].rd.items())
                nb += sum(w for t, w in bds[j].waste.items() if t <= hi)
            else:
                nb = bds[j].pack_rd
            rnum += nb / line * bds[j].level.lat
        wnum = (bds[first].phase_wr(phase) / line * bds[first].level.lat
                if first < len(bds) else 0.0)
        out.append((name, L, rnum, wnum))
    return out


def phase_time(an : analysis, phase : str) -> phase_result:
    """
    Roofline for one phase: its own compute time, its own traffic against
    each bus, and its own buffer occupancy.

    :param an: analysis of the configuration
    :param phase: 'ukr' or 'pack'
    :return: the phase's duration, and the term that set it
    """
    terms = [(_pick_phase(phase, an.t_ukr, an.t_pack), f'{phase} compute')]
    for bd in an.boundaries:
        L = bd.level
        rd, wr = bd.phase_rd(phase), bd.phase_wr(phase)
        if L.shared_bw:
            if L.rbw:
                terms.append(((rd + wr) / L.rbw, f"{phase}: {L.name} bus"))
        else:
            if L.rbw:
                terms.append((rd / L.rbw, f"{phase}: {L.name} rd"))
            if L.wbw:
                terms.append((wr / L.wbw, f"{phase}: {L.name} wr"))
    for bl in occupancy(an, phase):
        kind = ('load queue' if bl.holder == 'core' else
                'buffers' if bl.shared else 'fill buffers')
        if bl.rd_cycles:
            terms.append((bl.rd_cycles, f'{phase}: {bl.holder} {kind}'))
        if bl.wr_cycles:
            terms.append((bl.wr_cycles, f'{phase}: {bl.holder} write buffers'))
    # the first maximal term binds, as before; the ones level with it tie
    t, why = max(terms, key=lambda tm: tm[0])
    tied = [tag for tm, tag in terms if tag != why and tm >= t * (1 - TIE)]
    return phase_result(t, why, tied)


def predict(an : analysis) -> prediction:
    """
    Predicted run time.  The two phases run in series -- packing and the
    micro-kernel do not overlap in single-threaded BLIS.

    :param an: analysis of the configuration
    :return: per-phase results and their total
    """
    return prediction(phase_time(an, 'ukr'), phase_time(an, 'pack'), an.flops)


# ------------------------------------------------------------- hierarchy ---

_SIZE_SUFFIX = {'': 1, 'B': 1, 'K': 1024, 'KB': 1024, 'KIB': 1024,
                'M': 1024**2, 'MB': 1024**2, 'MIB': 1024**2,
                'G': 1024**3, 'GB': 1024**3, 'GIB': 1024**3,
                'T': 1024**4, 'TB': 1024**4, 'TIB': 1024**4}


def parse_size(tok : str) -> float:
    """
    Parse a size such as 32K, 768KiB, 1M or INF.  Suffixes are binary.

    :param tok: a number with an optional suffix, or one of INF, INFINITY,
                MEM, DRAM for main memory
    :return: size in bytes, or inf for main memory
    """
    t = tok.strip().upper()
    if t in ('INF', 'INFINITY', 'MEM', 'DRAM'):
        return float('inf')
    i = 0
    while i < len(t) and (t[i].isdigit() or t[i] == '.'):
        i += 1
    num, suf = t[:i], t[i:]
    if not num or suf not in _SIZE_SUFFIX:
        raise ValueError(f"bad size {tok!r}")
    return float(num) * _SIZE_SUFFIX[suf]


def _pair(tok : str, cast : type=float, unknown_neg : bool=True) -> tuple:
    """
    Parse one of the `a/b` fields of a cache spec.

    :param tok: '', 'a' or 'a/b'
    :param cast: type to convert each value to
    :param unknown_neg: whether a negative value means unknown (None)
    :return: (a, b, shared): shared is True for '' and 'a', which describe one
             resource serving both directions, and False for 'a/b'
    """
    tok = tok.strip()
    if not tok:
        return None, None, True
    if '/' in tok:
        a, b = tok.split('/', 1)
        va, vb = cast(a), cast(b)
        if unknown_neg:
            va = None if va < 0 else va
            vb = None if vb < 0 else vb
        return va, vb, False
    v = cast(tok)
    if unknown_neg and v < 0:
        v = None
    return v, v, True


def parse_hierarchy(spec : str | list[str]) -> list[cache_level]:
    """
    Parse a hierarchy, innermost level first.

    Accepts a list of entry strings (one per level, the --cache nargs form) or
    a single string with ';' between entries.  A bare preset name also works.

    Each entry is positional, ':'-separated, and any field may be left empty
    to take its default:

        size : line : assoc : rd[/wr] : usable : mshr[/wbuf] : latency

        32K:64:8:128/64:1.0:16/8:5    fully specified
        768K:::64/32                  only size and bandwidth given
        INF:::8/4:::90                only bandwidth and latency given

    A single number in the rd[/wr] or mshr[/wbuf] field means one shared
    resource -- one bus carrying reads and writes, or one buffer pool serving
    both -- and the two are then checked together rather than separately.
    -1 means unknown: the level is reported but never blamed.  INF marks main
    memory: its capacity is never consulted for residency, it only terminates
    the chain and supplies a bandwidth and a latency.

    :param spec: a list of entries, one ';'-separated string, or a preset name
    :return: the levels, in the order given
    """
    if isinstance(spec, str):
        spec = PRESETS.get(spec.strip().lower(), spec)
        entries = [e for e in spec.replace('{', '').replace('}', '')
                   .replace(',', ';').split(';') if e.strip()]
    else:
        if len(spec) == 1 and spec[0].strip().lower() in PRESETS:
            return parse_hierarchy(spec[0])
        entries = [e for e in spec if e.strip()]

    levels, nfin = [], 0
    for ent in entries:
        f = [x.strip() for x in ent.split(':')]
        f += [''] * (7 - len(f))
        if not f[0]:
            raise ValueError(f"entry {ent!r} needs a size")
        size = parse_size(f[0])
        line = int(f[1]) if f[1] else 64
        assoc = int(f[2]) if f[2] else 0
        rbw, wbw, shared_bw = _pair(f[3])
        usable = float(f[4]) if f[4] else 1.0
        mshr, wbuf, shared_buf = _pair(f[5], cast=int, unknown_neg=False)
        mshr, wbuf = int(mshr or 0), int(wbuf or 0)
        lat = float(f[6]) if f[6] else 0.0
        if size == float('inf'):
            name = 'MEM'
        else:
            nfin += 1
            name = f'L{nfin}'
        levels.append(cache_level(name, size, line, assoc, rbw, wbw,
                                  shared_bw, usable, mshr, wbuf, shared_buf,
                                  lat))
    if not levels:
        raise ValueError('empty hierarchy')
    return levels


def parse_sizes(tokens : list[str]) -> dict[str, int]:
    """
    Parse --s: one element size for every operand, per-operand sizes, or a
    default plus overrides.

        8                 every operand 8 bytes
        A:1 B:1 C:4       int8 inputs accumulating into int32
        2 C:4             A and B 2 bytes, C 4

    :param tokens: the --s arguments
    :return: element size in bytes for each of A, B and C
    """
    default, sizes = None, {}
    for tok in tokens:
        if ':' in tok:
            op, val = (x.strip() for x in tok.split(':', 1))
            op = op.upper()
            if op not in ('A', 'B', 'C'):
                raise ValueError(f"{tok!r}: the operand must be A, B or C")
            if op in sizes:
                raise ValueError(f"{op} is given twice")
            sizes[op] = int(val)
        else:
            if default is not None:
                raise ValueError("more than one default size")
            default = int(tok)
    missing = [op for op in 'ABC' if op not in sizes]
    if missing and default is None:
        raise ValueError(f"no size for {', '.join(missing)}: give each "
                         f"operand one, or a default as in '8 C:4'")
    sizes.update({op: default for op in missing})
    if min(sizes.values()) <= 0:
        raise ValueError("element sizes must be positive")
    return {op: sizes[op] for op in 'ABC'}


def parse_prefetch(tok : str) -> sw_prefetch:
    """
    Parse one --sw-prefetch entry:

        loop : operand : target [: hint [: distance]]

        ir:C:L2              next C tile into L2, normal insertion
        jr:B:L2:keep         next B_r into L2
        k:A:L1:strm:4        A into L1, 4 k-steps ahead, streaming insertion

    :param tok: the entry
    :return: the prefetch it describes
    """
    f = [x.strip() for x in tok.split(':')]
    if len(f) < 3:
        raise ValueError(f"prefetch {tok!r}: need loop:operand:target")
    loop, op, target = f[0].lower(), f[1].upper(), f[2].upper()
    hint = f[3].lower() if len(f) > 3 and f[3] else 'keep'
    dist = int(f[4]) if len(f) > 4 and f[4] else 0
    if dist < 0:
        raise ValueError(f"prefetch {tok!r}: the distance cannot be negative")
    if loop not in LOOPS:
        raise ValueError(f"prefetch {tok!r}: loop must be one of "
                         f"{', '.join(LOOPS)}")
    if op not in ('A', 'B', 'C'):
        raise ValueError(f"prefetch {tok!r}: operand must be A, B or C")
    if hint not in ('keep', 'strm'):
        raise ValueError(f"prefetch {tok!r}: hint must be keep or strm")
    return sw_prefetch(loop, op, target, hint, dist)


def name_levels(levels : list[cache_level],
                names : list[str]) -> list[cache_level]:
    """
    Rename levels -- to call a memory-side cache HBM rather than L3, say.
    Names are what --pf-level and --sw-prefetch targets refer to.

    :param levels: the levels, in the order they were given
    :param names: one name per level; an empty one keeps the default
    :return: the renamed levels
    """
    if len(names) != len(levels):
        raise ValueError(f"{len(names)} names for {len(levels)} levels; leave "
                         f"a name empty to keep its default, e.g. ',,HBM,DDR'")
    out = [replace(lvl, name=nm.strip().upper()) if nm.strip() else lvl
           for lvl, nm in zip(levels, names)]
    given = [lvl.name for lvl in out]
    for name in given:
        if not (name[0].isalpha() and name.replace('_', '').isalnum()):
            raise ValueError(f"{name!r}: a name is a letter followed by "
                             f"letters, digits or _")
    dup = sorted({name for name in given if given.count(name) > 1})
    if dup:
        raise ValueError(f"names must be unique; {', '.join(dup)} repeats")
    return out


PRESETS = {
    # Illustrative only -- substitute measured numbers for a real target.
    #        size : line : assoc : rd/wr : usable : mshr/wbuf : latency
    'generic': '32K:64:8:128/64:1.0:16/8:4;'
               '1M:64:8:48/24:1.0:16/8:20;'
               'INF:64::8/4:::150',
    'l3':      '64K:64:8:128/64:1.0:20/10:4;'
               '1M:64:8:64/32:1.0:16/8:18;'
               '32M:64:16:24/12:1.0:16/8:60;'
               'INF:64::12/6:::180',
    'l1only':  '32K:64:8:64/32:1.0:16/8:4;INF:64::8/4:::150',
    'bw-only': '32K:64:8:64/32;1M:64:8:48/24;INF:64::8/4',
    # 256 B lines at every level -- changes fill counts, not bytes
    'wide':    '64K:256:4:128/64:1.0:16/8:6;'
               '8M:256:16:64/32:1.0:16/8:40;'
               'INF:256::32/16:::200',
    # uniform 64 B inside, a wider line only at the outermost cache
    'mixed':   '64K:64:8:64/32:1.0:16/8:4;'
               '512K:64:8:48/24:1.0:16/8:20;'
               '32M:128:16:24/12:1.0:16/8:60;'
               'INF:128::12/6:::180',
}


# ---------------------------------------------------------------- report ---

def _row_prefix(bd : boundary) -> str:
    """
    Leading columns shared by the READS and WRITES tables.

    :param bd: boundary to describe
    :return: name, size and line size of its level, and the rungs it holds
    """
    L = bd.level
    hl = ', '.join(f"{r.loop}:{r.operand}" for r in bd.holds) or '-'
    sz = _fmt_size(L.size)
    return f"{L.name:>6}{sz:>8}{L.line:>5}B{hl:>22}"


def _fmt_sizes(u : ukernel) -> str:
    """
    :param u: microkernel model
    :return: its element size, or one per operand when they differ
    """
    per = [u.size(op) for op in 'ABC']
    if len(set(per)) == 1:
        return str(per[0])
    return ' '.join(f"{op}:{sz}" for op, sz in zip('ABC', per))


def _loops_line(an : analysis) -> str:
    """
    The cache-block loops a configuration actually runs: trips, block, and
    whether the block was clamped to the problem or leaves a partial last one.

    :param an: analysis of the configuration
    :return: e.g. "jc 1 x 2000 (clamped from 4000)   pc 13 x 160, last 80"
    """
    b, p, t = an.blocks, an.sizes, an.nest
    out = []
    for name, dim, want, got, trips in (('jc', p.n, b.n_c, t.n_c, t.n_jc),
                                        ('pc', p.k, b.k_c, t.k_c, t.n_pc),
                                        ('ic', p.m, b.m_c, t.m_c, t.n_ic)):
        last = dim - (trips - 1) * got
        note = (f" (clamped from {want})" if want > dim else
                f", last {last}" if last != got else "")
        out.append(f"{name} {trips} x {got}{note}")
    return '   '.join(out)


def _packed_notes(an : analysis) -> list[str]:
    """
    Where each packed block lives, which decides what of its traffic is real.

    :param an: analysis of the configuration
    :return: one line per packed operand
    """
    out = []
    for op, loop in PACKED.items():
        home = next((bd.level.name for bd in an.boundaries[:-1]
                     if any(r.loop == loop for r in bd.holds)), None)
        out.append(f"packed {op}_c is resident in {home}: beyond it the "
                   f"micro-kernel reads no {op}, and the buffer is repacked in "
                   f"place, never written back" if home else
                   f"packed {op}_c fits no cache: every pass over it, and its "
                   f"write-back, goes to {an.boundaries[-1].level.name}")
    return out


def _pct(v : float | None, t : float) -> str:
    """
    :param v: cycles a structure needs, or None if not modelled
    :param t: phase time
    :return: v as a percentage of t, or a dash
    """
    return f"{100*v/t:8.1f}%" if v else "        -"


def _effect(before : prediction, after : prediction) -> str:
    """
    Say what a change did, using nothing but the predictions with and without
    it -- so the description cannot disagree with the number printed beside
    it.  Why it did so is left to the notes, which state mechanism only.

    :param before: prediction without the change
    :param after: prediction with it
    :return: which limiter bound the run before, and which after
    """
    gain = after.flop_per_cycle - before.flop_per_cycle
    tie = f" (tied with {', '.join(after.tied)})" if after.tied else ""
    if abs(gain) < 0.05:                 # below what is printed
        return f"no change: still bound by {before.bound}{tie}"
    if after.bound == before.bound:
        verb = 'eases' if gain > 0 else 'worsens'
        return f"{verb} {before.bound}; still bound by it{tie}"
    if gain > 0:
        return f"was bound by {before.bound}, now by {after.bound}{tie}"
    return (f"costs time: was bound by {before.bound}, now by "
            f"{after.bound}{tie}")


def _rerun(an : analysis, sw : list[sw_prefetch]=None,
           blocks : blocking=None) -> analysis:
    """
    Re-analyse the same configuration with the software prefetches, the
    blocking, or both replaced.  Every other input comes from `an` itself, so
    a comparison or a sweep built from it is like for like by construction.

    :param an: the configuration to repeat
    :param sw: software prefetches to use instead; None keeps an's
    :param blocks: blocksizes to use instead; None keeps an's
    :return: the new analysis
    """
    return analyse(an.kernel, blocks or an.blocks, an.sizes,
                   [bd.level for bd in an.boundaries], an.pk,
                   an.write_allocate, an.core.mshr, an.pf, an.c_layout,
                   an.sw if sw is None else sw)


def report_prefetch(an : analysis, pred : prediction):
    """
    Print what each software prefetch does -- alone, against the same
    configuration without any, and within the set, against the set without
    it.  The two differ whenever prefetches interact: one can be what moves
    the limiter onto the resource another relieves.

    :param an: analysis including the software prefetches
    :param pred: its prediction
    """
    base = predict(_rerun(an, []))
    several = len(an.sw) > 1
    print(f"\nSOFTWARE PREFETCH   (each fetches what the NEXT iteration of "
          f"its loop needs)")
    for i, ev in enumerate(an.evals):
        sp = ev.pf
        dist = (f" d={max(1, sp.distance)}" if sp.loop == 'k' else
                f" lead={sp.distance}k" if sp.distance else "")
        tag = f"{sp.loop}:{sp.operand} -> {sp.target} {sp.hint}{dist}"
        alone = predict(_rerun(an, [sp]))
        d_alone = alone.flop_per_cycle - base.flop_per_cycle
        gain = f"{d_alone:+6.1f} FLOP/cy alone"
        effects = [('', _effect(base, alone))]
        if several:
            rest = predict(_rerun(an, an.sw[:i] + an.sw[i + 1:]))
            gain += (f", {pred.flop_per_cycle - rest.flop_per_cycle:+.1f} "
                     f"with the others")
            in_set = _effect(rest, pred)
            effects = ([('alone and with the others: ', in_set)]
                       if in_set == effects[0][1] else
                       [('alone: ', effects[0][1]),
                        ('with the others: ', in_set)])
        print(f"  {tag:<22} {gain}")
        if ev.target_index is not None and not ev.resident and ev.exposed > 0:
            print(f"      instance {_fmt_bytes(ev.instance)}, lead "
                  f"{ev.lead:.0f} cy vs {ev.latency:.0f} cy latency, covers "
                  f"{100*ev.coverage:.0f}%")
        for label, text in effects:
            print(f"      = {label}{text}")
        if ev.coverage > 0 and ev.target_index not in (None, 0):
            print(f"      - relieves {', '.join(ev.relieves)} of the latency "
                  f"beyond {sp.target}")
        for n in ev.notes:
            print(f"      - {n}")
        if ev.wasted:
            print(f"      - wastes {ev.wasted/2**30:.2f} GB, on {sp.target}'s "
                  f"buffers and the bandwidth beyond it; that costs time only "
                  f"if one of those binds")
        if ev.instr_per_kstep:
            print(f"      - adds {ev.instr_per_kstep:.2f} prefetch "
                  f"instructions per k-step (not charged)")
    print(f"  {'all together':<22} "
          f"{pred.flop_per_cycle - base.flop_per_cycle:+6.1f} FLOP/cy   "
          f"({base.flop_per_cycle:.1f} -> {pred.flop_per_cycle:.1f})")
    print(f"      = {_effect(base, pred)}")


def report(u : ukernel, b : blocking, p : problem, levels : list[cache_level],
           pk : packing=None, wa : bool=True, core_mshr : int=0,
           pf : hw_prefetcher=None, c_layout : str='rm',
           sw : list[sw_prefetch]=None) -> analysis:
    """
    Analyse one configuration and print the full breakdown: traffic per
    boundary, stream shapes, the two phases, buffer occupancy, software
    prefetching if any, and the prediction.

    :param u: microkernel model
    :param b: blocksizes
    :param p: problem sizes
    :param levels: cache/memory levels
    :param pk: packing model
    :param wa: whether a store that misses fetches the line first
    :param core_mshr: core load/miss queue entries
    :param pf: hw prefetcher model
    :param c_layout: - rm: C is row-major
                     - cm: C is column-major
    :param sw: software prefetches
    :return: the analysis, so that --counters can reuse it
    """
    an = analyse(u, b, p, levels, pk, wa, core_mshr, pf, c_layout, sw)
    pred = predict(an)
    pk, G = an.pk, 2 ** 30
    print(f"micro-kernel {u.m_r}x{u.n_r}  s={_fmt_sizes(u)}  peak={u.peak} FLOP/cy  "
          f"epi={u.epi_peak} FLOP/cy x {u.epi_flops} FLOP/elem  "
          f"C {'read+written' if u.c_read else 'write-only'}"
          f"{'' if wa else ', no write-allocate'}")
    print(f"blocking     k_c={b.k_c} m_c={b.m_c} n_c={b.n_c}   "
          f"problem {p.m}x{p.n}x{p.k}")
    t = an.nest
    print(f"loops        {_loops_line(an)}")
    rung = {r.loop: r for r in an.rungs}
    ir = rung['ir']
    print(f"A_c={rung['jr'].invariant/1024:.0f}K  "
          f"B_c={rung['ic'].invariant/1024:.0f}K  "
          f"B_r={ir.invariant/1024:.0f}K  A_r={ir.streamers['A']/1024:.0f}K  "
          f"C_r={ir.streamers['C']/1024:.1f}K")
    w = an.pack_elems
    print(f"packing      A {w['A']/1e6:.1f}M elem @ {pk.rate_a} elem/cy amp "
          f"{pk.amp_a}, B {w['B']/1e6:.1f}M elem @ {pk.rate_b} elem/cy amp "
          f"{pk.amp_b}")
    print(f"\nideal time {an.t_ideal/1e6:.1f} Mcycles  "
          f"= ukr {an.t_ukr/1e6:.1f} + pack {an.t_pack/1e6:.1f} "
          f"({100*an.t_pack/an.t_ideal:.0f}% packing)  for "
          f"{an.flops/1e9:.1f} GFLOP")
    print(f"core <- L1 : {an.core_rd/an.t_ideal:5.1f} B/cy read, "
          f"{an.core_wr/an.t_ideal:5.1f} B/cy write\n")

    print("READS")
    print(f"{'level':>6}{'size':>8}{'line':>6}{'holds':>22} | {'A':>8}{'B':>8}"
          f"{'C':>8}{'pack':>8}{'total':>9}{'B/cy':>8}{'have':>8}   supplies")
    print('-' * 119)
    prev = 'core'
    for bd in an.boundaries:
        L, t = bd.level, bd.rd
        have = f"{L.rbw:.1f}" if L.rbw else '?'
        print(f"{_row_prefix(bd)} | {t['A']/G:>7.2f}G{t['B']/G:>7.2f}G"
              f"{t['C']/G:>7.2f}G{bd.pack_rd/G:>7.2f}G{bd.tot_rd/G:>8.2f}G"
              f"{bd.tot_rd/an.t_ideal:>8.1f}{have:>8}   -> {prev}")
        prev = L.name

    print("\nWRITES")
    print(f"{'level':>6}{'size':>8}{'line':>6}{'holds':>22} | {'C':>8}"
          f"{'pack':>8}{'total':>9}{'B/cy':>8}{'have':>8}   supplies")
    print('-' * 103)
    prev = 'core'
    for bd in an.boundaries:
        L, t = bd.level, bd.wr
        have = 'shared' if L.shared_bw else (f"{L.wbw:.1f}" if L.wbw else '?')
        print(f"{_row_prefix(bd)} | {t['C']/G:>7.2f}G{bd.pack_wr/G:>7.2f}G"
              f"{bd.tot_wr/G:>8.2f}G{bd.tot_wr/an.t_ideal:>8.1f}{have:>8}"
              f"   -> {prev}")
        prev = L.name

    pf = an.pf
    l0 = an.boundaries[0].level.line
    print(f"\nSTREAMS per ukr call (innermost line {l0}B, prefetcher tracks "
          f"{pf.streams} stream(s) of >= {pf.min_run} lines).")
    if pf.level:
        print(f"  The prefetcher fills {pf.level}, so this coverage enters")
        print(f"  the prediction: it relieves every holder inside "
              f"{pf.level} of")
        print(f"  the latency beyond it, for the operands it covers.")
    else:
        print("  Diagnostic only: with no --pf-level the prefetcher fills no")
        print("  modelled level, so this does not enter the prediction.  It")
        print("  shows which operand the hardware starts early, i.e. where")
        print("  software prefetch is redundant and where it is the only "
              "option.")
    for st in an.streams:
        print(f"  {st.operand}  {st.count:3d} stream(s) x {st.lines:6.0f} lines"
              f"   -> {100*an.cov[st.operand]:3.0f}% prefetched")

    print(f"\nPHASES (serial: packing and the micro-kernel do not overlap)")
    print(f"{'phase':>7}{'compute':>10}{'actual':>10}{'share':>8}   "
          f"{'rd GB':>8}{'wr GB':>8}{'rd B/cy':>9}{'wr B/cy':>9}   limiter")
    mem = an.boundaries[-1]
    for nm, tc, ph in (('ukr', an.t_ukr, pred.ukr),
                       ('pack', an.t_pack, pred.pack)):
        rd, wr = mem.phase_rd(nm), mem.phase_wr(nm)
        print(f"{nm:>7}{tc/1e6:>9.1f}M{ph.time/1e6:>9.1f}M"
              f"{100*ph.time/pred.time:>7.1f}%   {rd/G:>8.2f}{wr/G:>8.2f}"
              f"{rd/ph.time:>9.1f}{wr/ph.time:>9.1f}   {ph.limiter}")

    print(f"\nBUFFER OCCUPANCY   (% of that phase's time the structure is busy)")
    print(f"{'holder':>7}{'ukr rd':>9}{'ukr wr':>9}{'pack rd':>10}{'pack wr':>10}")
    tu, tp = pred.ukr.time, pred.pack.time
    occ_pack = {bl.holder: bl for bl in occupancy(an, 'pack')}
    for bl in occupancy(an, 'ukr'):
        bp = occ_pack[bl.holder]
        print(f"{bl.holder:>7}{_pct(bl.rd_cycles, tu):>9}"
              f"{_pct(bl.wr_cycles, tu):>9}{_pct(bp.rd_cycles, tp):>10}"
              f"{_pct(bp.wr_cycles, tp):>10}")
    print("  a read holds a buffer at every level it passes through, until its "
          "data is back")
    print("  (a memory line holds L2's for MEM latency, L1's for L2+MEM, the "
          "core's for all);")
    print("  writes do not (the inner buffer frees as soon as the next level "
          "takes it)")
    receiver = 'core'
    for bd in an.boundaries:
        print(f"  {bd.level.name:>5} -> {receiver:<5}: {bd.fills/1e6:8.1f}M fills "
              f"of {bd.req_line}B, {100*bd.miss_out:4.1f}% continue outward")
        receiver = bd.level.name
    print("  'holds' is what fits in that level; it reduces the row BELOW it.")
    for note in _packed_notes(an):
        print(f"  {note}")
    print("  Nearest misses:")
    for bd in an.boundaries:
        m = bd.miss
        if m and bd.level.size != float('inf'):
            need = m.need_at(bd.level, an.strm.get(bd.level.name, {}))
            print(f"    {bd.level.name}: {m.loop}:{m.operand} needs "
                  f"{_fmt_bytes(need)} resident > "
                  f"{_fmt_size(bd.level.cap)} -> its {m.reuse:.0f}x reuse "
                  f"is lost")

    if an.sw:
        report_prefetch(an, pred)

    print(f"\npredicted: {pred.flop_per_cycle:6.1f} FLOP/cycle "
          f"({100*pred.flop_per_cycle/u.peak:.0f}% of peak), "
          f"limited by {pred.limiter}"
          + (f", tied with {', '.join(pred.tied)}" if pred.tied else ""))
    return an


def _core_cache(name : str) -> bool:
    """
    :param name: a level name
    :return: whether it names one of the core's own caches, L1, L2, ...
    """
    return name[0] == 'L' and name[1:].isdigit()


def counters(an : analysis, vl : int=64) -> pmu_prediction:
    """
    Predicted ARM PMU event counts, for checking the model against
    `perf stat`.

    Boundary convention: boundary i supplies level i-1, so what boundary i+1
    supplies is exactly level i's refill stream.  Refills and writebacks are
    line-granular and follow straight from the traffic; L1D_CACHE counts
    memory uops, so it needs the vector length (one SVE ld1d of SVL bits that
    does not straddle a line is one access).

    :param an: analysis of the configuration
    :param vl: vector length in bytes, for the L1D_CACHE uop count
    :return: predicted event counts, with the normalisers to print them
    """
    u, p = an.kernel, an.sizes
    bds = an.boundaries
    pred = predict(an)

    sa, sb, sc = u.size('A'), u.size('B'), u.size('C')
    uops_ld = an.calls * (an.nest.k_mean * (u.m_r * sa + u.n_r * sb) / vl
                          + (u.m_r * u.n_r * sc / vl if u.c_read else 0))
    uops_st = an.calls * u.m_r * u.n_r * sc / vl
    w = an.pack_elems
    uops_pack = (w['A'] * sa + w['B'] * sb) * 2 / vl    # read once, written once

    ev = {'CPU_CYCLES': pred.time,
          'L1D_CACHE': uops_ld + uops_st + uops_pack,
          'L1D_CACHE_RD': uops_ld + uops_pack / 2,
          'L1D_CACHE_WR': uops_st + uops_pack / 2}
    for i, bd in enumerate(bds[:-1]):            # every real cache
        L, nxt = bd.level, bds[i + 1]
        # the core's PMU counts its own caches, L1D, L2D, ...; a level named
        # otherwise -- a memory-side HBM cache, say -- goes under its name,
        # for an uncore or memory-controller counter to check
        pre = f'{L.name}D_CACHE' if _core_cache(L.name) else L.name
        if i:                                    # L2 accesses = L1 refill+wb
            up = bds[i - 1].level.line
            if _core_cache(L.name):
                ev[pre] = (bd.tot_rd + bd.tot_wr) / up
                # split as Arm's L2D_CACHE_RD/_WR: the refills of the level
                # above, and its write-backs -- vendors whose access counters
                # see only the reads compare against _RD
                ev[f'{pre}_RD'] = bd.tot_rd / up
                ev[f'{pre}_WR'] = bd.tot_wr / up
            else:
                ev[f'{pre}_ACCESS'] = (bd.tot_rd + bd.tot_wr) / up
        # a refill brings in one of THIS level's lines
        ev[f'{pre}_REFILL'] = nxt.tot_rd / L.line
        ev[f'{pre}_WB'] = nxt.tot_wr / L.line
    # requests in flight in each core cache's miss buffers, summed over the
    # run's cycles: what an occupancy counter integrates (AMD's
    # ls_alloc_mab_count, Intel's l1d_pend_miss.pending for L1).  Built from
    # the unloaded latencies, so a measured excess is queueing; divided by
    # the level's refills it is the mean miss latency
    for ph in ('ukr', 'pack'):
        for name, _, rnum, _ in _buffer_cycles(an, ph):
            if name != 'core' and _core_cache(name):
                key = f'{name}D_MISS_OCCUPANCY'
                ev[key] = ev.get(key, 0.0) + rnum
    ksteps = p.m * p.n * p.k / (u.m_r * u.n_r)
    return pmu_prediction(ev, ksteps, an.calls, pred)


def report_counters(an : analysis, vl : int=64):
    """
    Print the predicted PMU counts, the derived ratios that discriminate the
    residency and latency hypotheses, and a ready-made perf command line.

    :param an: analysis of the configuration
    :param vl: vector length in bytes, for the L1D_CACHE uop count
    """
    pm = counters(an, vl)
    lines = ', '.join(f"{bd.level.name} {bd.level.line}B"
                      for bd in an.boundaries
                      if bd.level.size != float('inf'))
    print(f"\nPREDICTED PMU COUNTS   (vector length {vl} B; lines: {lines})")
    print(f"{'event':>24}{'total':>12}{'per k-step':>13}{'per ukr call':>14}")
    print('-' * 63)
    for k, v in pm.events.items():
        print(f"{k:>24}{v/1e9:>10.3f} G{v/pm.ksteps:>13.2f}"
              f"{v/pm.calls:>14.1f}")
    other = [bd.level.name for bd in an.boundaries[:-1]
             if not _core_cache(bd.level.name)]
    if other:
        print(f"  ({', '.join(other)}: not a core cache, so no core PMU event "
              f"-- compare with an uncore or memory-controller counter)")
    rf1 = pm.events.get('L1D_CACHE_REFILL', 0)
    rf2 = pm.events.get('L2D_CACHE_REFILL', 0)
    t = pm.pred.time
    print(f"\nDERIVED  (these are what discriminate the hypotheses)")
    print(f"  L1D_CACHE_REFILL / CPU_CYCLES        {rf1/t:8.3f} fills/cy"
          f"   = {rf1/t*an.boundaries[0].level.line:5.1f} B/cy")
    if rf2:
        print(f"  L2D_CACHE_REFILL / L1D_CACHE_REFILL  {100*rf2/rf1:8.1f} %"
              f"       <- residency vs latency test")
    occ1 = pm.events.get('L1D_MISS_OCCUPANCY', 0)
    if occ1 and rf1:
        print(f"  L1D_MISS_OCCUPANCY / L1D_CACHE_REFILL {occ1/rf1:7.1f} cy"
              f"      <- mean L1 miss latency, unloaded")
        print(f"  L1D_MISS_OCCUPANCY / CPU_CYCLES      {occ1/t:8.2f}"
              f"          <- misses in flight on average")
    print(f"  predicted limiter                    {pm.pred.limiter}")
    print(f"\n  perf stat -e cycles,l1d_cache,l1d_cache_refill,l1d_cache_wb,"
          f"l2d_cache,l2d_cache_refill,l2d_cache_wb \\\n"
          f"            -e stall_backend,l1d_tlb_refill,l2d_tlb_refill <cmd>")


def sweep(an : analysis, var : str='kc', ac_budget : float=None):
    """
    Sweep k_c over a configuration and print a row per step.  m_c is
    re-derived at every step so that the A_c footprint stays constant,
    because that is the constraint that actually binds.  Everything else --
    hierarchy, packing, core queue, prefetching, layout -- is taken from `an`.

    :param an: the configuration to sweep; its n_c is kept
    :param var: parameter to sweep; only 'kc' is supported
    :param ac_budget: A_c footprint to hold, in bytes; defaults to an's
    """
    u, b = an.kernel, an.blocks
    ac = ac_budget or b.m_c * b.k_c * u.size('A')
    print(f"sweeping {var}, holding A_c ~ {ac/1024:.0f} KiB   (rd / wr B/cy)\n")
    hdr = f"{'k_c':>6}{'m_c':>6}{'B_r':>7}{'pack%':>7}"
    for bd in an.boundaries:
        hdr += f"{bd.level.name:>14}"
    print(hdr + f"{'FLOP/cy':>10}  limiter")
    print('-' * (len(hdr) + 22))
    for kc in (32, 48, 64, 96, 128, 144, 192, 256, 384, 512, 768, 1024):
        mc = max(u.m_r, int(ac / (kc * u.size('A')) / u.m_r) * u.m_r)
        sa = _rerun(an, blocks=blocking(kc, mc, b.n_c))
        pred = predict(sa)
        b_r = next(r for r in sa.rungs if r.loop == 'ir').invariant
        row = (f"{kc:>6}{mc:>6}{b_r/1024:>6.0f}K"
               f"{100*sa.t_pack/sa.t_ideal:>6.0f}%")
        for bd in sa.boundaries:
            row += (f"{bd.tot_rd/sa.t_ideal:>8.1f}/"
                    f"{bd.tot_wr/sa.t_ideal:<5.1f}")
        print(row + f"{pred.flop_per_cycle:>10.1f}  {pred.limiter}")


def selftest() -> int:
    """
    Internal consistency checks.  No measured numbers from any particular
    machine -- these verify the model against its own closed forms.

    :return: 0 if every check passed, 1 otherwise, for use as an exit status
    """
    u = ukernel(8, 8, 8, 16.0, 16.0, 3, True)
    p = problem(2048, 2048, 2048)
    b = blocking(256, 256, 1024)
    free = packing(0, 0, 1, 1, False)
    ok = True

    def chk(name : str, got : float, want : float, tol : float=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol * max(1.0, abs(want))
        ok = ok and good
        print(f"  [{'ok ' if good else 'FAIL'}] {name:<52}"
              f"{got:14.4g} vs {want:14.4g}")

    print("reuse factors must multiply to the total reuse of each operand")
    prod = {'A': 1.0, 'B': 1.0, 'C': 1.0}
    for r in make_goto_ladder(u, b, p):
        prod[r.operand] *= r.reuse
    chk("A: prod(rungs) == n/n_r", prod['A'], p.n / u.n_r)
    chk("B: prod(rungs) == m/m_r", prod['B'], p.m / u.m_r)
    chk("C: prod(rungs) == ceil(k/k_c)", prod['C'],
        make_loop_nest(b, p).n_pc)

    print("\ncore-side traffic per k-step is fixed by the micro-kernel shape")
    lv = parse_hierarchy('32K:::-1;1M:::-1;INF:::-1')
    an = analyse(u, b, p, lv, free)
    ks = p.m * p.n * p.k / (u.m_r * u.n_r)
    b0 = an.boundaries[0]
    chk("read  == (m_r+n_r)*s + m_r*n_r*s/k_c",
        b0.ukr_rd / ks, (u.m_r * u.size('A') + u.n_r * u.size('B'))
        + u.m_r * u.n_r * u.size('C') / b.k_c)
    chk("write == m_r*n_r*s/k_c", b0.ukr_wr / ks,
        u.m_r * u.n_r * u.size('C') / b.k_c)

    print("\nwith unbounded caches, memory sees only compulsory traffic")
    an = analyse(u, b, p, parse_hierarchy('1T:::-1;2T:::-1;INF:::-1'), free)
    mem = an.boundaries[-1]
    chk("MEM read  == (m*k + k*n + m*n)*s", mem.tot_rd,
        p.m*p.k*u.size('A') + p.k*p.n*u.size('B') + p.m*p.n*u.size('C'))
    chk("MEM write == m*n*s", mem.tot_wr, p.m*p.n*u.size('C'))
    chk("... with A and B arriving through packing, not the micro-kernel",
        mem.rd['A'] + mem.rd['B'], 0.0)

    print("\nwith no cache at all, memory sees the full core-side traffic")
    an = analyse(u, b, p, parse_hierarchy('1:::-1;2:::-1;INF:::-1'), free)
    chk("MEM read == core read", an.boundaries[-1].ukr_rd, b0.ukr_rd)

    print("\nwrite-allocate fills are issued by the cache, never by the core")
    uz = ukernel(8, 8, 8, 16.0, 16.0, 1, False)        # beta == 0
    az = analyse(uz, b, p, parse_hierarchy(PRESETS['generic']), free)
    chk("no C read crosses the innermost boundary",
        az.boundaries[0].rd['C'], 0.0)
    chk("the next boundary out carries the RFO",
        az.boundaries[1].rd['C'], az.boundaries[1].wr['C'])

    print("\nphase times sum to the prediction")
    an = analyse(u, b, p, parse_hierarchy(PRESETS['generic']),
                 packing.preset('vector', u.size('B')))
    pred = predict(an)
    chk("t_total == t_ukr_eff + t_pack_eff", pred.time,
        phase_time(an, 'ukr').time + phase_time(an, 'pack').time)
    chk("FLOP/cycle == flops / t", pred.flop_per_cycle, an.flops / pred.time)

    print("\nrefill counts use the line size of the level being filled")
    pm = counters(an)
    bds = an.boundaries
    chk("L1D_CACHE_REFILL == L2-supplied bytes / L1 line",
        pm.events['L1D_CACHE_REFILL'], bds[1].tot_rd / bds[0].level.line)
    chk("L2D_CACHE_REFILL == MEM-supplied bytes / L2 line",
        pm.events['L2D_CACHE_REFILL'], bds[2].tot_rd / bds[1].level.line)

    am = analyse(u, b, p, parse_hierarchy('64K:64:8:-1:1.0:16/8:4;'
                                          '8M:256:16:-1:1.0:16/8:40;'
                                          'INF:256::-1:::200'), free)
    pmm = counters(am)
    chk("wider L2 line -> fewer L2 refills for the same bytes",
        pmm.events['L2D_CACHE_REFILL'], am.boundaries[2].tot_rd / 256)
    chk("L1 refills still counted in 64 B lines",
        pmm.events['L1D_CACHE_REFILL'], am.boundaries[1].tot_rd / 64)

    print("\nread occupancy chains outward, write occupancy does not")
    occ = {bl.holder: bl for bl in occupancy(an, 'ukr')}
    l1 = bds[0].level                       # the holder's own line size
    chk("L1 read  occupancy sums L2 and MEM latencies", occ['L1'].rd_cycles,
        sum(bds[j].ukr_rd / l1.line * bds[j].level.lat for j in (1, 2))
        / l1.mshr)
    chk("L1 write occupancy uses L2 latency only", occ['L1'].wr_cycles,
        bds[1].ukr_wr / l1.line * bds[1].level.lat / l1.wbuf)

    print("\noccupancy counts the HOLDER's lines, not the supplier's")
    occm = {bl.holder: bl for bl in occupancy(am, 'ukr')}
    lm = am.boundaries
    chk("L1 (64B lines) holding fills served by a 256B L2",
        occm['L1'].rd_cycles,
        sum(lm[j].ukr_rd / 64 * lm[j].level.lat for j in (1, 2))
        / lm[0].level.mshr)

    print("\nprefetching relieves a holder only if holder < target < boundary")
    u2 = ukernel(16, 32, 8, 128.0, 32.0, 3, True)
    p2, b2 = problem(4000, 4000, 4000), blocking(192, 256, 800)
    h2 = parse_hierarchy(["32K:64:8:128/64:1.0:16/8:4",
                          "768K:64:12:48/16:1.0:16/8:21", "INF:64::8/8:::150"])
    pk2 = packing(8, 8, 1, 1, False)
    a0 = analyse(u2, b2, p2, h2, pk2)
    o0 = {bl.holder: bl for bl in occupancy(a0)}
    ac = analyse(u2, b2, p2, h2, pk2, sw=[sw_prefetch('ir', 'C', 'L2')])
    oc = {bl.holder: bl for bl in occupancy(ac)}
    mem = a0.boundaries[2]
    chk("L2 prefetch of C removes exactly C's MEM term from L1",
        oc['L1'].rd_cycles, o0['L1'].rd_cycles
        - mem.rd['C'] / 64 * mem.level.lat / 16)
    chk("... and leaves the L2 buffers, which still fetch it, alone",
        oc['L2'].rd_cycles, o0['L2'].rd_cycles)
    al = analyse(u2, b2, p2, h2, pk2, sw=[sw_prefetch('k', 'A', 'L1',
                                                      'keep', 64)])
    ol = {bl.holder: bl for bl in occupancy(al)}
    chk("L1 prefetch never relieves the L1 fill buffers",
        ol['L1'].rd_cycles, o0['L1'].rd_cycles)

    print("\nprefetches with nothing to do are recognised")
    ev = analyse(u2, b2, p2, h2, pk2,
                 sw=[sw_prefetch('ir', 'A', 'L2')]).evals[0]
    chk("next A_r is already resident in L2 (part of A_c)",
        float(ev.resident), 1.0)
    ev = analyse(u2, b2, p2, h2, pk2,
                 sw=[sw_prefetch('ir', 'B', 'L2')]).evals[0]
    chk("B is ir's invariant: no coverage", ev.coverage, 0.0)
    ae = analyse(u2, b2, p2, h2, pk2, sw=[sw_prefetch('ic', 'A', 'L2')])
    chk("next A_c: nothing to hide, it reaches L2 through packing",
        float(ae.evals[0].exposed == 0 and ae.evals[0].coverage == 0), 1.0)
    ae = analyse(u2, b2, p2, h2, pk2, sw=[sw_prefetch('ic', 'C', 'L2')])
    chk("next C block cannot survive in L2: evicted, no coverage",
        ae.evals[0].coverage + (0.0 if ae.evals[0].evicted else 1.0), 0.0)
    chk("its waste is exactly C's traffic from beyond L2",
        ae.evals[0].wasted, a0.boundaries[2].rd['C'])
    oe = {bl.holder: bl for bl in occupancy(ae)}
    chk("... which the L1 buffers never see", oe['L1'].rd_cycles,
        o0['L1'].rd_cycles)

    print("\na 'strm' hint charges a streamer the ways its window occupies")
    b3 = blocking(96, 512, 800)
    held = lambda a: [f"{r.loop}:{r.operand}" for r in a.boundaries[0].holds]
    chk("k_c=96: B_r does not fit L1 under plain LRU",
        float('ir:B' in held(analyse(u2, b3, p2, h2, pk2))), 0.0)
    chk("... but does with A streamed (B_r + 2 ways = 32K)",
        float('ir:B' in held(analyse(u2, b3, p2, h2, pk2, sw=[
            sw_prefetch('k', 'A', 'L1', 'strm', 32)]))), 1.0)
    chk("... and not once the window outgrows one way (d=33)",
        float('ir:B' in held(analyse(u2, b3, p2, h2, pk2, sw=[
            sw_prefetch('k', 'A', 'L1', 'strm', 33)]))), 0.0)

    print("\nverdicts are derived from the predictions, never asserted")
    h4 = parse_hierarchy(["32K:64:8:128/64:1.0:16/8:5",
                          "768K:64:12:64/32:1.0:32/16:18", "INF:::5/5:::90"])
    u4 = ukernel(16, 32, 8, 128.0, 32.0, 3, True)
    p4 = problem(2000, 2000, 2000)
    b4, pk4 = blocking(160, 512, 800), packing.preset('vector', 8)
    kb = sw_prefetch('k', 'B', 'L1', 'keep', 8)
    q0 = predict(analyse(u4, b4, p4, h4, pk4, core_mshr=16))
    q1 = predict(analyse(u4, b4, p4, h4, pk4, core_mshr=16, sw=[kb]))
    chk("L1 prefetch helps when the core queue is what binds",
        float(q1.flop_per_cycle > q0.flop_per_cycle + 1), 1.0)
    chk("... and the verdict names the queue it relieved",
        float(_effect(q0, q1) == "was bound by ukr: core load queue, now by "
              "ukr: L1 fill buffers"), 1.0)
    mk = lambda f, lim: prediction(phase_result(1024 / f, lim),
                                   phase_result(0.0, ''), 1024.0)
    chk("a gain is never described as no change",
        float(_effect(mk(80, 'x'), mk(81, 'x')).startswith('eases')), 1.0)
    chk("a loss is never described as easing",
        float(_effect(mk(80, 'x'), mk(70, 'y')).startswith('costs time')), 1.0)
    chk("an unchanged run is described as unchanged",
        float(_effect(mk(80, 'x'), mk(80, 'x')).startswith('no change')), 1.0)
    a5 = analyse(u2, b3, p2, h2, pk2, sw=[sw_prefetch('k', 'A', 'L1',
                                                      'strm', 4)])
    chk("a 'strm' hint into L1 can help with no core queue (capacity)",
        float(predict(a5).flop_per_cycle
              > predict(_rerun(a5, [])).flop_per_cycle + 1), 1.0)
    # C is 16 streams of 4 lines row-major, 32 of 2 column-major: with a
    # 3-line minimum run the prefetcher covers C in one layout only, so the
    # layout changes the answer -- otherwise this test could not fail
    hw = hw_prefetcher(20, 3, 'L2')
    a_rm, a_cm = (analyse(u4, b4, p4, h4, pk4, pf=hw, c_layout=cl, sw=[kb])
                  for cl in ('rm', 'cm'))
    chk("the C layout changes this prediction (so the next check can fail)",
        float(abs(predict(a_rm).flop_per_cycle
                  - predict(a_cm).flop_per_cycle) > 1), 1.0)
    for a6 in (a_rm, a_cm):
        chk(f"a re-run reproduces the run it repeats (C {a6.c_layout})",
            predict(_rerun(a6, a6.sw)).flop_per_cycle,
            predict(a6).flop_per_cycle)

    try:
        analyse(u4, b4, p4, parse_hierarchy('l1only'), pk4,
                pf=hw_prefetcher(2, 4, 'L2'))
        raised = 0.0
    except ValueError:
        raised = 1.0
    chk("a prefetcher level that does not exist is a clear error", raised, 1.0)

    print("\none buffer count means one pool for reads and writes together")
    sh = parse_hierarchy(["32K:64:8:128/64:1.0:16:5",
                          "768K:64:12:64/32:1.0:32/16:18", "INF:::5/5:::90"])
    sp_ = parse_hierarchy(["32K:64:8:128/64:1.0:16/16:5",
                           "768K:64:12:64/32:1.0:32/16:18", "INF:::5/5:::90"])
    o_sh = {bl.holder: bl for bl in occupancy(analyse(u4, b4, p4, sh, pk4))}
    o_sp = {bl.holder: bl for bl in occupancy(analyse(u4, b4, p4, sp_, pk4))}
    chk("shared L1 pool: reads and writes against the same 16 slots",
        o_sh['L1'].rd_cycles, o_sp['L1'].rd_cycles + o_sp['L1'].wr_cycles)
    chk("... reported as one figure, with no separate write term",
        float(o_sh['L1'].wr_cycles is None and o_sh['L1'].shared), 1.0)

    print("\na sweep varies the blocking and keeps every other input")
    # One case per input _rerun must carry.  Each pair differs in that input
    # alone, and is first shown to give different answers at the new
    # blocking -- so dropping any one input fails its own check.
    # chosen so that every input below moves the answer by > 5 FLOP/cy
    b6, hw2 = blocking(100, 250, 500), hw_prefetcher(20, 3, 'L2')
    uz = ukernel(16, 32, 8, 128.0, 32.0, 1, False)       # beta == 0
    cases = [('core queue',     u4, {'core_mshr': 16},     {}),
             ('C layout',       u4, {'pf': hw2, 'c_layout': 'cm'},
                                    {'pf': hw2}),
             ('hw prefetcher',  u4, {'pf': hw2},           {}),
             ('sw prefetches',  u4, {'sw': [sw_prefetch('ir', 'C', 'L2')]},
                                    {}),
             ('packing',        u4, {'pk': packing.preset('scalar', 8)}, {}),
             ('write-allocate', uz, {'write_allocate': False}, {})]
    def fresh(uk, bl, kw):
        return predict(analyse(uk, bl, p4, h4, **kw))
    for name, uk, on, off in cases:
        kw = {'pk': pk4, **on}
        gap = abs(fresh(uk, b6, kw).flop_per_cycle
                  - fresh(uk, b6, {'pk': pk4, **off}).flop_per_cycle)
        chk(f"{name}: changes the answer at the new blocking", float(gap > 0.1),
            1.0)
        chk(f"{name}: kept by a re-run at the new blocking",
            predict(_rerun(analyse(uk, b4, p4, h4, **kw), blocks=b6))
            .flop_per_cycle, fresh(uk, b6, kw).flop_per_cycle)

    print("\nloops run whole trips over blocks no larger than the problem")
    def same_as_clamped(field_, big):
        wide = blocking(**{**vars(b4), field_: big})
        snug = blocking(**{**vars(b4), field_: 2000})
        return (predict(analyse(u4, wide, p4, h4, pk4)).flop_per_cycle,
                predict(analyse(u4, snug, p4, h4, pk4)).flop_per_cycle)
    for fld, nm in (('n_c', 'n'), ('m_c', 'm'), ('k_c', 'k')):
        chk(f"{fld} = 2 {nm} behaves exactly like {fld} = {nm}",
            *same_as_clamped(fld, 4000))
    a7 = analyse(u4, blocking(160, 256, 4000), p4, h4, pk4)
    chk("A is packed at least once, even with n_c > n",
        float(a7.pack_elems['A'] >= p4.m * p4.k), 1.0)
    chk("n_c one short of n runs two jc trips: A is packed twice",
        pack_work(u4, blocking(160, 256, 1999), p4)['A'], 2.0 * p4.m * p4.k)

    print("\neach operand's element size moves that operand's quantities only")
    base = {'A': 1, 'B': 2, 'C': 4}
    b8, h8 = blocking(160, 256, 800), parse_hierarchy('generic')
    pk8 = packing.preset('scalar', 2)
    loops = [(lp, op) for lp in LOOPS for op in sorted(VARYING[lp])]
    def facts(sz):
        uk = ukernel(16, 32, 8, 128.0, 32.0, 3, True, sizes=sz)
        an = analyse(uk, b8, p4, h8, pk8)
        tr = core_traffic(uk, b8, p4, pk8)
        foot = {}
        for r in make_goto_ladder(uk, b8, p4):
            foot[(r.loop, r.operand)] = r.invariant
            for o, v in r.streamers.items():
                foot[(r.loop, o)] = v
        return {'rd': tr.rd, 'wr': tr.wr, 'foot': foot,
                'stream': {st.operand: st.lines
                           for st in stream_shape(uk, b8, p4, 64)},
                'inst': {lo: prefetch_instance(uk, b8, p4, sw_prefetch(*lo,
                                                                       'L2'))
                         for lo in loops},
                'pack_wr': sum(tr.pack_wr.values()),
                'uops': counters(an).events['L1D_CACHE'],
                'calls': an.calls, 'k': an.nest.k_mean, 'w': an.pack_elems}
    f0 = facts(base)
    for op in 'ABC':
        f1 = facts({**base, op: 2 * base[op]})
        good = all(f1[t][x] == f0[t][x] * (2 if x == op else 1)
                   for t in ('rd', 'wr') for x in f0[t])
        good &= all(f1['foot'][key] == f0['foot'][key] *
                    (2 if key[1] == op else 1) for key in f0['foot'])
        good &= all(f1['stream'][x] == f0['stream'][x] * (2 if x == op else 1)
                    for x in 'ABC')
        good &= all(f1['inst'][lo] == f0['inst'][lo] * (2 if lo[1] == op else 1)
                    for lo in loops)
        own = f0['w'].get(op, 0.0) * base[op]
        good &= abs(f1['pack_wr'] - f0['pack_wr'] - own) < 1e-6
        mr_nr = {'A': 16 * f0['k'], 'B': 32 * f0['k'], 'C': 2 * 16 * 32}[op]
        d_uops = (f0['calls'] * mr_nr * base[op] + 2 * own) / 64
        good &= abs((f1['uops'] - f0['uops']) - d_uops) <= 1e-6 * d_uops
        chk(f"doubling {op}'s size doubles {op}'s traffic, footprints, "
            f"streams, prefetches, packing and uops -- nothing else",
            float(good), 1.0)
    u9 = ukernel(16, 32, 8, 128.0, 32.0, 3, True, sizes={'A': 1, 'B': 2,
                                                       'C': 4})
    u0 = ukernel(16, 32, 999, 128.0, 32.0, 3, True, sizes={'A': 1, 'B': 2,
                                                         'C': 4})
    chk("the default size is unused once every operand has its own",
        predict(analyse(u0, b8, p4, h8, pk8)).flop_per_cycle,
        predict(analyse(u9, b8, p4, h8, pk8)).flop_per_cycle)

    print("\na limit level with the binding one is reported as tied")
    # all loads / B loads = (16 + 32 + 512/160) / 32 = 1.6 = lead / L1 latency
    # = 8 / 5, so a distance-1 prefetch of B hides exactly the excess of the
    # core queue over the L1 fill buffers: the two tie
    u10, b10 = ukernel(16, 32, 8, 128.0, 32.0, 3, True), blocking(160, 256, 8000)
    p10 = problem(8000, 8000, 8000)
    q = predict(analyse(u10, b10, p10, h4, pk4, core_mshr=16,
                        sw=[sw_prefetch('k', 'B', 'L1', 'keep', 1)]))
    chk("d=1 B prefetch: core queue binds, tied with the L1 fill buffers",
        float(q.bound == 'ukr: core load queue'
              and q.tied == ['ukr: L1 fill buffers']), 1.0)

    print("\na packed block resident in a cache is not re-read or written "
          "back beyond it")
    # A_c (320K) fits the 768K L2; B_c (k_c*n_c*w) does not until n_c is small
    p11 = problem(8000, 8000, 8000)
    def at_mem(n_c):
        a = analyse(u4, blocking(160, 256, n_c), p11, h4, pk4)
        return a, a.boundaries[-1]
    a11, m11 = at_mem(8000)
    b_pack = p11.k * p11.n * u4.size('B')
    chk("A_c resident in L2: the micro-kernel reads no A from memory",
        m11.rd['A'], 0.0)
    chk("... and its buffer is never written back: writes are C's and B_c's",
        m11.tot_wr, m11.wr['C'] + b_pack)
    ir11 = next(r for r in a11.rungs if r.loop == 'ir')
    chk("B_c fits no cache: every pass over it still comes from memory",
        m11.rd['B'], a11.boundaries[1].rd['B'] / ir11.reuse)
    chk("write-backs do not depend on n_c while A_c is resident",
        float(len({round(at_mem(nc)[1].tot_wr) for nc in (8000, 1024, 512)})),
        1.0)
    a12, m12 = at_mem(256)
    chk("n_c = 256 makes B_c resident too: no B from memory either",
        float(any(r.loop == 'ic' for r in a12.boundaries[1].holds)
              and m12.rd['B'] == 0.0 and m12.tot_wr == m12.wr['C']), 1.0)

    print("\na prefetch the hardware already makes is reported as redundant")
    hw_l1 = hw_prefetcher(1, 4, 'L1')        # one tracker: the longest, B
    p13 = problem(8000, 8000, 8000)
    def pred13(**kw):
        return analyse(u4, blocking(160, 256, 8000), p13, h4, pk4,
                       core_mshr=16, **kw)
    a13 = pred13(pf=hw_l1, sw=[sw_prefetch('k', 'B', 'L1', 'keep', 4)])
    chk("B into L1 on top of a hardware stream into L1: no gain",
        predict(a13).flop_per_cycle, predict(pred13(pf=hw_l1)).flop_per_cycle)
    chk("... and the verdict says why, first",
        float(a13.evals[0].notes[0].startswith('redundant')), 1.0)
    # 10 trackers: B and A take one each, C gets 8 of its 16 streams
    a14 = pred13(pf=hw_prefetcher(10, 4, 'L2'),
                 sw=[sw_prefetch('ir', 'C', 'L2')])
    chk("C into L2 with the hardware covering half of it: adds the other half",
        float(any('adds 50% on top' in n for n in a14.evals[0].notes)), 1.0)

    print("\nlevels can be renamed, and are targeted by name")
    h15 = name_levels(parse_hierarchy(["32K:64:8:128/64:1.0:16/8:5",
                                       "768K:64:12:64/32:1.0:32/16:18",
                                       "4G:::5/5:::90", "INF:::2/2:::160"]),
                      ['', '', 'hbm', 'DDR'])
    chk("names: defaults kept where empty, others upper-cased",
        float([L.name for L in h15] == ['L1', 'L2', 'HBM', 'DDR']), 1.0)
    a15 = analyse(u4, b4, p4, h15, pk4, sw=[sw_prefetch('ir', 'C', 'HBM')])
    chk("a prefetch can target a renamed level",
        float(a15.evals[0].target_index == 2), 1.0)
    ev15 = counters(a15).events
    chk("a level not named L<n> gets no core PMU event name",
        float('HBM_REFILL' in ev15 and 'HBMD_CACHE_REFILL' not in ev15
              and 'L2D_CACHE_REFILL' in ev15), 1.0)
    for bad, why in ((['L1', 'L2'], 'count'), (['A', 'A', 'B', 'C'], 'dup'),
                     (['L1', 'L2', 'H:M', 'DDR'], 'char')):
        try:
            name_levels(h15, bad)
            raised = 0.0
        except ValueError:
            raised = 1.0
        chk(f"bad names are rejected ({why})", raised, 1.0)
    chk("capacities print short",
        float([_fmt_size(x) for x in (32 * 2**10, 768 * 2**10, 4 * 2**30,
                                      float('inf'))]
              == ['32K', '768K', '4G', 'inf']), 1.0)

    print("\na prefetch's lead can be set, and a burst arrives in waves")
    h16 = name_levels(parse_hierarchy(["32K:64:8:128/64:1.0:16/8:5",
                                       "768K:64:12:64/32:1.0:32/16:18",
                                       "4G:::5/5:::90", "INF:::2/2:::160"]),
                      ['', '', 'HBM', 'DDR'])
    b16 = blocking(160, 256, 2000)
    def ev16(*spec):
        return analyse(u4, b16, problem(4000, 4000, 4000), h16, pk4,
                       sw=[sw_prefetch(*spec)]).evals[0]
    a16 = analyse(u4, b16, problem(4000, 4000, 4000), h16, pk4)
    t_k, t_call = loop_period(a16, 'k'), loop_period(a16, 'ir')
    chk("ir:C with no distance leads by one call",
        ev16('ir', 'C', 'L2').lead, t_call)
    chk("ir:C:...:16 leads by 16 k-steps", ev16('ir', 'C', 'L2', 'keep', 16).lead,
        16 * t_k)
    e99 = ev16('ir', 'C', 'L2', 'keep', 999)
    chk("a lead longer than the loop is capped at one iteration",
        float(e99.lead == t_call and any('capped' in n for n in e99.notes)),
        1.0)
    chk("k-loop prefetches still default to 1 k-step ahead",
        ev16('k', 'B', 'L1').lead, t_k)
    # 64 C lines through 32 L2 buffers: two waves of ~96 cycles
    # (the latency is 96.4 cy -- a little goes on to DDR -- so two waves need
    # 192.8 cy: 25 k-steps, or 28 at an unroll of 4)
    e16, e28 = ev16('ir', 'C', 'L2', 'keep', 16), ev16('ir', 'C', 'L2', 'keep', 28)
    chk("C tile 16 k-steps ahead: the second wave is late",
        float(0.5 < e16.coverage < 0.8), 1.0)
    chk("... 28 k-steps ahead: both waves in time", e28.coverage, 1.0)
    chk("B_r read as it goes: a burst one call ahead keeps pace",
        ev16('jr', 'B', 'L2', 'keep', 160).coverage, 1.0)
    chk("... but not one issued at the next call's start",
        float(ev16('jr', 'B', 'L2', 'keep', 1).coverage < 0.1), 1.0)
    chk("a burst with more lead than latency is not called late",
        float(not any(n.startswith('late') for n in e16.notes)
              and any(n.startswith('burst') for n in e16.notes)), 1.0)
    e8 = ev16('ir', 'C', 'L2', 'keep', 8)
    chk("... but one with less is, as well as burst-limited",
        float(any(n.startswith('late') for n in e8.notes)
              and any(n.startswith('burst') for n in e8.notes)), 1.0)
    chk("instructions count iterations, not lead",
        ev16('ir', 'C', 'L2', 'keep', 16).instr_per_kstep,
        ev16('ir', 'C', 'L2').instr_per_kstep)

    print("\npacking outside the region costs nothing inside it")
    p17 = problem(2000, 2000, 2000)
    a_in = analyse(u4, b4, p17, h4, pk4)
    a_out = analyse(u4, b4, p17, h4, packing(pk4.rate_a, pk4.rate_b,
                                             outside='AB'))
    a_a = analyse(u4, b4, p17, h4, packing(pk4.rate_a, pk4.rate_b,
                                           outside='A'))
    chk("packed outside: no packing time", a_out.t_pack, 0.0)
    chk("... and no packing traffic at any boundary",
        sum(bd.pack_rd + bd.pack_wr for bd in a_out.boundaries), 0.0)
    chk("... while the micro-kernel's traffic is unchanged",
        a_out.boundaries[1].ukr_rd, a_in.boundaries[1].ukr_rd)
    chk("only A outside: B's packing remains",
        a_a.pack_elems['B'], a_in.pack_elems['B'])

    print("\nmiss occupancy chains the latencies: a mean L1 miss pays the L2")
    print("latency, plus the memory latency for the share that misses L2 too")
    lv = parse_hierarchy('32K:64::-1::16:5;1M:64::-1::32:18;INF:::-1:::150')
    ev = counters(analyse(u, b, p, lv, free)).events
    r1, r2 = ev['L1D_CACHE_REFILL'], ev['L2D_CACHE_REFILL']
    chk("L1 occupancy / L1 refills == 18 + (R2/R1)*150",
        ev['L1D_MISS_OCCUPANCY'] / r1, 18 + r2 / r1 * 150)
    chk("L2 occupancy / L2 refills == 150", ev['L2D_MISS_OCCUPANCY'] / r2, 150)
    chk("L2D_CACHE == L2D_CACHE_RD + L2D_CACHE_WR", ev['L2D_CACHE'],
        ev['L2D_CACHE_RD'] + ev['L2D_CACHE_WR'])
    chk("L2D_CACHE_RD == L1D_CACHE_REFILL", ev['L2D_CACHE_RD'], r1)
    chk("L1D_CACHE == L1D_CACHE_RD + L1D_CACHE_WR", ev['L1D_CACHE'],
        ev['L1D_CACHE_RD'] + ev['L1D_CACHE_WR'])

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ main ---

def build_parser() -> argparse.ArgumentParser:
    """
    The command-line interface.

    :return: a parser for every option the model takes
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', nargs='+', default=['generic'],
                    metavar='LEVEL',
                    help='one argument per level, innermost first: '
                         'size:line:assoc:rd[/wr]:usable:mshr[/wbuf]:latency'
                         ' . Any field may be empty to take its default, e.g.'
                         ' --cache "32K:64:8:128/64:1.0:16/8:5" "768K:::64/32"'
                         ' "INF:::8/4:::90" . A single number in the '
                         'bandwidth '
                         'or buffer field means one shared resource. '
                         '-1 = unknown, INF = memory. Presets: '
                         + ', '.join(sorted(PRESETS)))
    ap.add_argument('--level-names', default='', metavar='NAMES',
                    help='names for the --cache levels, comma-separated and '
                         'innermost first, e.g. L1,L2,HBM,DDR; an empty name '
                         'keeps its default (L1, L2, ... and MEM), e.g. '
                         ',,HBM,DDR.  Prefetch targets use these names')
    ap.add_argument('--mr', type=int, default=8)
    ap.add_argument('--nr', type=int, default=8)
    ap.add_argument('--s', nargs='+', default=['8'], metavar='SIZE',
                    help='element size in bytes: one for every operand, '
                         'or per operand, e.g. --s A:1 B:1 C:4 for int8 '
                         'inputs into an int32 accumulator, or a default '
                         'plus overrides, e.g. --s 2 C:4.  --peak and '
                         '--epi-peak are then in that precision\'s '
                         'operations per cycle')
    ap.add_argument('--peak', type=float, default=16.0)
    ap.add_argument('--epi-peak', type=float, default=16.0)
    ap.add_argument('--beta', choices=['general', 'one', 'zero'], default='general',
                    help='general: fmul+fma, C read; one: fma only, C read; '
                         'zero: fmul only, C not read')
    ap.add_argument('--no-write-allocate', action='store_true')
    ap.add_argument('--pack', choices=['none', 'vector', 'scalar'],
                    default='vector', help='packing kernel quality preset')
    ap.add_argument('--pack-rate', type=float, default=None,
                    help='override elements/cycle for both pack kernels')
    ap.add_argument('--pack-amp-b', type=float, default=None,
                    help='override B read amplification (bytes fetched per '
                         'byte used); line/s for a kernel walking the long '
                         'stride, 1 for one that transposes in registers')
    ap.add_argument('--kc', type=int, default=256)
    ap.add_argument('--mc', type=int, default=256)
    ap.add_argument('--nc', type=int, default=1024)
    ap.add_argument('--size', type=int, nargs=3, default=[2048, 2048, 2048],
                    metavar=('M', 'N', 'K'))
    ap.add_argument('--mem-bw', type=float, default=None,
                    help='override the outermost level\'s read bandwidth, '
                         'in bytes/cycle; its write bandwidth too if shared')
    ap.add_argument('--core-mshr', type=int, default=0,
                    help="core load queue entries.  Same role as a level's "
                         "mshr field but one step in: it holds every load, "
                         "L1 hits included, so it is normally much larger "
                         "(64-128) than an L1 fill-buffer count.  Stores go "
                         "through the store queue and are not counted.  "
                         "0 = not modelled.")
    ap.add_argument('--pf-streams', type=int, default=0,
                    help='streams the hardware prefetcher can track; 0 = none')
    ap.add_argument('--pf-min-run', type=int, default=4,
                    help='lines a stream needs before the prefetcher locks on')
    ap.add_argument('--pf-level', default='',
                    help='level the hardware prefetcher fills, e.g. L2.  Set, '
                         'its coverage relieves the holders inside that level '
                         'of the latency beyond it; empty, the stream table '
                         'is a diagnostic only')
    ap.add_argument('--sw-prefetch', nargs='+', default=[], metavar='PF',
                    help='software prefetches, loop:operand:target[:hint'
                         '[:distance]] -- issued in that loop, fetching what '
                         'its next iteration needs.  loop is k, ir, jr, ic, '
                         'pc or jc; hint keep (default) or strm; distance is '
                         'the lead in k-steps: for loop k how far ahead (1 by '
                         'default), for the others how long before use (one '
                         'iteration by default).  e.g. ir:C:L2 k:A:L1:strm:4 '
                         'ir:C:L2:keep:24')
    ap.add_argument('--c-layout', choices=['rm', 'cm'], default='rm',
                    help='how the ukr sees C: rm = m_r row runs, cm = n_r col runs')
    ap.add_argument('--vl', type=int, default=64,
                    help='vector length in bytes, for the L1D_CACHE uop count')
    ap.add_argument('--counters', action='store_true',
                    help='print predicted PMU counts for perf verification')
    ap.add_argument('--sweep', choices=['kc'])
    ap.add_argument('--selftest', action='store_true',
                    help='internal consistency checks, no machine data')
    return ap


def _make_levels(ap : argparse.ArgumentParser,
                 a : argparse.Namespace) -> list[cache_level]:
    """
    Build the hierarchy from the command line: --cache, then --level-names
    and --mem-bw.

    :param ap: the parser, for reporting a bad argument
    :param a: parsed arguments
    :return: the levels, innermost first
    """
    try:
        lv = parse_hierarchy(a.cache)
    except ValueError as e:
        ap.error(f"--cache: {e}")
    if a.level_names:
        try:
            lv = name_levels(lv, a.level_names.split(','))
        except ValueError as e:
            ap.error(f"--level-names: {e}")
    if a.mem_bw is not None:
        lv[-1] = replace(lv[-1], rbw=a.mem_bw,
                         wbw=a.mem_bw if lv[-1].shared_bw else lv[-1].wbw)
    return lv


def _make_kernel(ap : argparse.ArgumentParser,
                 a : argparse.Namespace) -> ukernel:
    """
    Build the micro-kernel model from the command line.

    :param ap: the parser, for reporting a bad --s
    :param a: parsed arguments
    :return: the micro-kernel model
    """
    try:
        sizes = parse_sizes(a.s)
    except ValueError as e:
        ap.error(f"--s: {e}")
    epi = {'general': 3, 'one': 2, 'zero': 1}[a.beta]
    return ukernel(a.mr, a.nr, sizes['A'], a.peak, a.epi_peak, epi,
                   c_read=a.beta != 'zero', sizes=sizes)


def main(argv : list[str]=None) -> int:
    """
    Run the model from the command line.

    Everything here is local, so no function can come to depend on a name
    that only exists when the file is run as a script.

    :param argv: arguments, without the program name; None reads sys.argv
    :return: exit status
    """
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    u = _make_kernel(ap, a)
    b = blocking(a.kc, a.mc, a.nc)
    p = problem(*a.size)
    lv = _make_levels(ap, a)
    pk = packing.preset(a.pack, u.size('B'), lv[0].line)
    if a.pack_rate is not None:
        pk.rate_a = pk.rate_b = a.pack_rate
    if a.pack_amp_b is not None:
        pk.amp_b = a.pack_amp_b
    wa = not a.no_write_allocate

    pf = hw_prefetcher(a.pf_streams, a.pf_min_run, a.pf_level.upper())
    caches = [L.name for L in lv if L.size != float('inf')]
    if pf.level and pf.level not in caches:
        ap.error(f"--pf-level {a.pf_level}: the cache levels are "
                 f"{', '.join(caches)}")
    try:
        sw = [parse_prefetch(t) for t in a.sw_prefetch]
    except ValueError as e:
        ap.error(f"--sw-prefetch: {e}")

    if a.sweep:
        if a.counters:
            ap.error("--counters describes one configuration; it cannot be "
                     "combined with --sweep")
        sweep(analyse(u, b, p, lv, pk, wa, a.core_mshr, pf, a.c_layout, sw),
              a.sweep)
    else:
        an = report(u, b, p, lv, pk, wa, a.core_mshr, pf, a.c_layout, sw)
        if a.counters:
            report_counters(an, a.vl)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
