GOTO-nest harness
=================

`goto_bench` drives BLIS's own micro-kernel and pack kernel through a GOTO loop
nest that can be cut at any rung and rearranged, so each rung of `model.py`'s
ladder can be measured on its own, bottom up. Its flags follow `model.py`
wherever the concept is the same, so any run can be replayed through the model.

```
jc: for n   step n_c          [pack B, if placed here]
  pc: for k   step k_c        [pack B_c / A, if here]
    ic: for m   step m_c      [pack A_c, if here]
      jr: for n_c step n_r
        ir: for m_c step m_r
          micro-kernel, k_c steps
```

Build: `cmake --build .` builds it next to `bench` (target `goto_bench`).
Single-threaded; pin it with `--cpu N` or `taskset`.

Examples
--------

```
# full GEMM, BLIS's packing placement
goto_bench --size 4000 4000 4000 --kc 160 --mc 256 --nc 2000

# one macro-kernel, packed blocks prepared outside the timed region
goto_bench --size 4000 4000 4000 --kc 160 --mc 256 --nc 2000 --top jr

# the micro-kernel alone (sweep --kc for peak and per-call overhead)
goto_bench --size 16 32 128 --kc 128 --mc 16 --nc 32 --top ukr

# pack rate of one A_c / B_c block
goto_bench --size 256 2000 160 --kc 160 --mc 256 --nc 2000 --top packA

# next C tile into L2, and next_b (B_r of the next jr, one call ahead)
goto_bench ... --prefetch ir:C:L2 jr:B:L2:keep:1

# blocks as multiples of the micro-tile; C two calls ahead
goto_bench --size 4000 4000 4000 --kc 160 --mc 20mr --nc 200nr --prefetch ir:C:L2:2kc

# counters, as found by probe_events.py
goto_bench ... --events-file events.txt --group 4

# correctness: against a plain loop (small sizes), or the library's own GEMM
goto_bench --size 203 147 131 --kc 37 --mc 44 --nc 60 --pack A:pc,B:jc --verify
goto_bench --size 4000 4000 4000 --kc 160 --mc 256 --nc 2000 --verify-lib
```

Options
-------

* `--top ukr|ir|jr|ic|pc|jc` -- outermost rung executed. Rungs above it run one
  iteration; packing placed above it happens once, outside the timed region.
  `packA` / `packB` time one block's packing alone.
* `--pack A:R,B:R` -- where packing happens: A at `ic` (BLIS), `pc`, `jc` or
  `pre`; B at `pc` (BLIS), `jc` or `pre`. `pre` packs once, untimed.
* `--prefetch loop:operand:target[:hint[:lead]]` -- same syntax as `model.py`.
  Issued between micro-kernel calls, for what the *next* iteration of `loop`
  uses in execution order (crossing loop boundaries, so `ir:C` reaches the
  next column's first tile). If the operand is packed above that loop, the
  packed block is prefetched; otherwise the source its next pack will read.
  `lead` is in micro-kernel calls before the end of the loop's iteration,
  `N` or `Nkc` (N calls of k_c k-steps), and may stand in the hint's place:
  `ir:C:L2:2kc` is `ir:C:L2:keep:2`; default: issued at its start, i.e. one
  iteration ahead. C is prefetched for writing. The k-loop is inside the
  kernel and cannot be prefetched from here.
* `--kc`, `--mc`, `--nc` take `N` or a multiple of the micro-tile: `4mr`,
  `200nr`, `mr` -- resolved once the kernel is known; `validate.py run` takes
  several of each.
* `--c-layout rm|cm`, `--beta general|one|zero|X`, `--alpha X`. A and B are
  column-major.
* `--events E,E,...` / `--events-file F`, `--group N`, `--cycles-event E` --
  counters are measured in groups of N, each its own pass, each led by the
  cycle counter: one group with more events than the PMU has counters would
  never be scheduled. Each event is first opened on its own beside the cycle
  counter; one the kernel refuses is skipped with a warning naming it, so it
  costs neither the rest of its group nor the cycles. A run takes
  ceil(events / N) passes (one, the cycle counter alone, without events), each
  `reps` measurements of `inner` region runs; each event reports its own
  pass's minimum. A group
  the PMU cannot hold at once is reported as multiplexed. A group counts from
  when it is opened; each measurement reads it before and after the timed
  body, two read() system calls. What they cost the counts is measured around
  an empty body, by an instance of the group opened and closed beforehand (two
  at once may not fit: RISC-V has one cycle counter), and subtracted from
  every count (`# counter overhead` in the header): tens of cycles where the
  counters leave out kernel time, microseconds where they cannot -- the fixed
  cycle and instruction counters of RISC-V cores without Smcntrpmf count the
  kernel's part of each read, which inflated short regions' cycles by 3-9% on
  the SpacemiT K1 (a 1.6 GHz clock read as 1.65-1.75 GHz, while seconds-long
  runs read 1.60). There `u_mode_cycle` counts user-mode cycles only, which
  `probe_events.py` picks for `CPU_CYCLES` and `validate.py` passes on as
  `--cycles-event`: nothing of the kernel's to subtract, at the price of a
  programmable counter in every group.
* `--reps N` / `--min-time S`, `--warmup N`, `--inner N` -- `inner` repeats
  the region inside each timed measurement (default: enough for ~200 us), so
  small regions are not timer-bound. Every reported value is per region run.
* `--verify`, `--verify-lib` -- run the region once and compare it with a
  plain triple loop, or with the linked library's `cblas_dgemm` -- its
  `bli_dgemm` if it was built without CBLAS, which is BLIS's default
  (`configure --enable-cblas`). Both may be given. The library reference is
  fast enough for full-size problems, and with the library's own default
  blocking the two results agree bit for bit.
* `--cpu N`, `--hugepages`, `--line BYTES`, `--label TEXT`.

Output is `# key: value` metadata, then a CSV header and one row.

What each ISA can prefetch
--------------------------

The header prints what each requested prefetch became on this build.

* AArch64: `PRFM {PLD,PST}{L1,L2,L3}{KEEP,STRM}` -- every combination.
* x86-64: `prefetcht0/t1/t2` choose L1/L2/L3 -- on Zen 5, t1 fills every
  level but L1 and t2 every level but L1 and L2; `strm` is `prefetchnta`.
  Write intent is `prefetchw`, which fills L1 only, so a write prefetch aimed
  at L2 or L3 (`ir:C:L2`) becomes `prefetcht1`/`t2`: the level is kept, the
  hint dropped.
* RISC-V: Zicbop `prefetch.r/.w`, with a Zihintntl hint in front for the
  level: `ntl.p1` for L2 (skips the innermost private level), `ntl.pall` for
  L3 (skips every private level, so it lands in the innermost shared one --
  L2 where L2 is shared, as on K1). No strm/keep distinction: `strm` is
  ignored. Both are base-ISA HINT encodings, emitted raw: no `-march` flag
  needed, and a core without the extensions executes them as no-ops.

Every prefetch is inline `asm volatile`. GCC's tree DCE removed
`__builtin_prefetch` from this code once it was inlined -- silently, which
its documentation allows -- so the builtin is only a last resort for ISAs
without an asm path, and the header says so.

Counters: probe_events.py
-------------------------

```
probe_events.py                 # map model.py's events onto this machine's
probe_events.py --write events  # events.txt for goto_bench, events.json map
probe_events.py --list          # every cache/memory/latency event found
probe_events.py --include 'l2_a[rw]_channel|lsu|vlsu' --write events
                                # ... plus every event matching, measured only
probe_events.py --perf-json saved.json   # perf list -j saved elsewhere
```

`prefetch_test` answers whether a software prefetch reaches L1 on a core, and
how far ahead it must go, without BLIS or counters:

```
build/prefetch_test --cpu 5 --ghz 1.6 16 256 4096 [--huge]
```

It walks a buffer's 64-byte lines in a random order, which no hardware
prefetcher follows, each line holding the index of the next: a pointer chase,
so every miss costs its full latency. `chase` is that latency for the level
the buffer lives in. The same walk with a prefetch D lines ahead -- `pf.L1`
(RISC-V `prefetch.r`, x86 `prefetcht0`, AArch64 `prfm pldl1keep`), `pf.L2`
(`ntl.p1` + `prefetch.r`, `prefetcht1`, `pldl2keep`), or `load`, a plain load
whose value is discarded (`ld x0`) -- falls to the L1 row's cost once D steps
cover the latency. `pf.L1` no better than `chase` at any D means the core
ignores the prefetch (Zicbop allows that); `load` better than `pf.L1` means it
drops prefetches but not loads. An out-of-order core runs ahead and hides D; on
an in-order one the D where a column flattens is the distance needed. `--huge`
asks for transparent huge pages, keeping TLB misses out of the larger sizes.

`--copy` packs instead, the way BLIS packs A: 8-row panels of a column-major
matrix, one 64-byte line per column, four columns per iteration, loaded then
stored -- with vector loads (`vle64` at LMUL 2, VLEN >= 256) and with scalar
`ld`/`sd`, each with `prefetch.r` D columns ahead (D = 0: none). The chase has
one miss in flight and scalar loads; packing has four and vector loads, so a
prefetch that helps the chase but not `--copy` points at one of those.

perf's own errors can land in the middle of its JSON (`Error: failed to open
tracing events directory` before the closing bracket); the probe then parses
entry by entry. perf's tables call the core PMU `default_core` whatever sysfs
calls it (`cpu` on RISC-V and x86, `armv8_pmuv3_N` on Arm); the probe maps
it. `--include REGEX` writes every encodable event whose name matches, under
its own name: `validate.py` reports them as measured, not modelled -- the way
to explore a core whose events the model has no slot for, such as the
SpacemiT X60's L2 AR/AW channel stalls, LSU and vector-LSU stalls.

Reads `/sys/bus/event_source/devices/*` and, if installed, `perf list -j`
(perf 6.x; an older perf is reported as such), and computes encodings as perf
does. Every pick is then opened with `perf_event_open` -- directly, no perf
needed -- and a candidate the kernel refuses is passed over, with the reason
printed. On Arm, sysfs lists only what the PMU's PMCEID registers advertise,
which never covers the recommended implementation-defined range 0x40-0xbf
(`L1D_CACHE_RD/WR` 0x40/0x41, `L2D_CACHE_RD/WR` 0x50/0x51, ...): those are
tried by number on the core PMU, and kept only if they count something in a
short memory-touching run. If this process cannot open even the cycle counter
(`perf_event_paranoid`, no PMU), nothing is checked and the picks say
unverified; `--no-verify` skips the checks. Each event has candidates in order of
preference, checked against perf list on Kunpeng 920, Zen 5 and SpacemiT K1;
a candidate that only approximates the model's meaning is printed with a note
(`[demand misses only]`). Kernel generic events are taken only where perf
lists them as supported, and generic last-level-cache events go to whichever
level is last here (from sysfs: L2 on K1, L3 on Zen 5).

Besides the model's events (L1D/L2D accesses, refills and write-backs, the
`_RD`/`_WR` splits, and `L1D_MISS_OCCUPANCY`) it maps a few the model does not
predict but that explain a run: `STALL_BACKEND`, `STALL_BACKEND_MEM`,
`LOAD_QUEUE_STALL`, and -- to attribute refills -- L1 fills by demand,
hardware prefetch and software prefetch, the L1 and L2 prefetchers' requests
served by DRAM, and software prefetches dispatched or finding their line
already in flight.

Where events are missing it says why: on AMD, an unloaded amd-uncore module;
on RISC-V, a core whose mvendorid-marchid-mimpid perf's mapfile lacks, so only
generic events have names.

Encodings come out as `PMU:<type>:<config>`, which needs the patched
`performance_counters` backend. Entries that are not one concrete event are
skipped and counted: the term syntax recent perf lists for each PMU
(`ibs_op/ldlat=0..0xfff,.../`), `?` parameters, and values too wide for their
field. Modifiers (`/u`) and perf's own terms (`period=`) are dropped. Needs
Python 3.8 or later.

### Miss occupancy

`L1D_MISS_OCCUPANCY` is the number of L1D misses in flight, summed over cycles
(AMD Zen 4/5 `ls_alloc_mab_count`, Intel `l1d_pend_miss.pending`). The model
predicts it from the same chained latencies it uses for its fill-buffer
limits, so with `L1D_CACHE_REFILL` it gives, by Little's law, the mean L1 miss
latency and the misses in flight on average -- measured and predicted. The
model uses unloaded latencies: a measured excess is queueing. `validate.py`
prints both whenever the two events are mapped.

Validating the model: validate.py
---------------------------------

```
validate.py detect-cache                              # sizes/lines/ways from sysfs
validate.py calibrate --bench build/goto_bench [--ghz G] [--c-layout cm] -o calib.json
validate.py run --bench build/goto_bench --calib calib.json \
    --cache <model.py spec for this machine> \
    --size 2000 4000 --kc 128 256 --mc 96 --nc 4096 --top jc \
    [--prefetch ir:C:L2] [--events-map events.json] [--per-kstep] -o results.csv
```

`calibrate` fits the micro-kernel at k small enough to stay in L1 as
t(k) = C + b*k, giving `--peak` (2*mr*nr/b) and the per-call cost C, which the
model charges as its epilogue -- leaving out k < 8, where the per-call work
overlaps the short k-loop differently, and any point more than 2% off the
line while three remain (a fitted peak above the hardware's is the sign of
one). `--peak` fixes the slope when the peak is known, so that only C is
fitted. It then times A_c and B_c packs three times each: with source and
packed copy in L1, in L2, and with the source in memory (4x the last level's
size, at most 64 MB), all sized from sysfs (`--l1`, `--l2` override). The L1
rate is the pack kernel's own cost and becomes `--pack-rate`: the model adds
where the source lives through its memory terms. The L2 and memory rates show
what each level costs on top. An out-of-order core keeps them close to the L1
rate; an in-order core stalls on every miss not prefetched, so its rates fall
to one line per latency -- on the SpacemiT X60, 0.21 elements/cycle with A in
L2 and 0.033 with B from memory, against ~1 in L1. It also times one `ir`
loop in L1 at two k.

`--c-layout` matters most here: a kernel given the C layout it does not prefer
may move C element by element or through a temporary tile. goto_bench prints
the kernel's preference; calibrate with the layout you run (`run` warns when
they differ).

`run` takes several problems: `--size 4000 512x14x96` is a 4000 cube and one
512x14x96 problem -- three bare numbers are three cubes, and the run lists the
problems it took. A run is flagged when C's leading dimension makes a tile's
columns (rows, for row-major C) share L1 sets beyond the associativity: with
ldc = 512 doubles, the 14 columns of an 8x14 tile fall in two set groups of a
32 KB 4-way L1, seven lines per set, and evict each other on every call.

`run --only EVENT...` measures a subset of `--events-map`: long runs pay one
pass per group, so a full map of 35 events at `--group 4` runs each
configuration nine times over.

`run --per-kstep` splits cycles and every counter, per call, into a fixed part
and a part per k-step, fitted per problem and region over runs that differ
only in the k each call runs, min(k_c, k) (Theil-Sen:
the median of pairwise slopes, so one run in another regime does not tilt the
line, and runs more than 10% off it are listed). The per-k-step part is the
k-loop's; the fixed part is the C update, the call, a prefetch hook. On the
X60's 8x14 dgemm kernel with row-major C:

```
                                 per k-step   per call
  cycles                              36.99       2092   = 6.06 FLOP/cycle in the k-loop
  INST_RETIRED                        51.05       1905
  L1D_CACHE_RD                        18.05        486
  L1D_CACHE_WR                         0.03        284
  LSU_LOAD_WAW_STALL                  12.50        740
  VIDU_VEC1_STRUCT_HAZARD_STALL        1.83        499
```

-- a third of the k-loop lost to load write-after-write stalls (a load to a
register an earlier write still holds: too few scalar registers rotated for
B), and ~2000 cycles per call around it. With
`--events-map`, the intercepts of loads and stores per call against k, less
the C tile's, become the default `--call-uops`, and the slope is printed as
loads per k-step, with the `--bcast` it matches. With `--prefetch`, the same
runs with goto_bench's prefetch hook give its cycles, loads and stores per
call; `run` with the same `--prefetch` charges them. The hook's cycles overlap
the epilogue's wait for C (`model.py --call-extra`): a call costs the larger.

`--cache` latencies are increments along the chain -- what each level adds to
a miss from the level above -- not load-to-use totals: with totals of 7, 14
and 46 cycles for L1, L2 and L3 hits, the spec says 7, 7 and 32.

C moves in lines, not bytes. Each row of a tile (column, for column-major C)
touches every line it overlaps: an 8x6 dgemm tile's 48-byte rows touch 1.5
64-byte lines on average, twice their bytes, while an 8x24 tile's 192-byte
rows are three whole lines. That matters for FMA kernels whose vectors are
narrower than a line -- NEON, 128-bit RVV. Neighbouring tiles share the line
between them, so the extra traffic crosses a boundary only where the C strip
does not stay inside it until the next jr iteration, i.e. where the jr rung
is not held. The model assumes C's rows start line-aligned. A software
prefetch into an outer level also counts as an access there of its own:
`ir:C:L2` adds its lines to `L2D_CACHE_RD`.

`--cpu N` pins goto_bench, for `calibrate` and `run` alike. On a desktop keep
the CPU's SMT sibling idle; on a server avoid the CPU that takes most
interrupts, often 0.

A trailing `:v` on a level marks it as one whose evictions all go to the next
level, clean lines too -- the next is exclusive of it, or a victim cache. Its
write-backs are then every line it was filled with, not just the dirty ones:
`64K:64:4:32/16:1.0:10:7:v`. TSV110's L1 and L2 behave so (their write-back
counters follow their fills, though A and B are never written), as does Zen's
L2, whose L3 is a victim cache. A `v` on the last cache adds nothing: memory
holds every line already. Without a cycle counter, cycles come from ns * `--ghz`, and
peak is then only relative to that clock.

`run` measures every combination of the sweep values and replays each through
the model with the same blocking, layout, beta, packing and prefetches.
Packing placed above `--top` is packed outside the region, so the model gets
`packing.outside`; a region below `jc` repeats on the same data, so the
smallest cache that holds it is treated as memory (`--no-steady` to turn that
off). It compares cycles, and FLOP/cycle on the same 2*m*n*k work, plus the
counters mapped by `--events-map`: the model's events with their ratio, the
others as measured only. `--vl` sets the vector length the model's
`L1D_CACHE` uop counts assume (64 B for SME or AVX-512, 32 for AVX2 or 256-bit
RVV, 16 for NEON), and `--bcast` the operands the k-loop loads element by
element: AOCL's 8x24 dgemm kernel broadcasts the 8 values of A and loads B as
three vectors, 11 loads per k-step where vectors throughout would be 4.
`--call-uops LD/ST` adds the memory operations each call makes outside the
k-loop and the C tile -- the kernel reading its arguments from memory, saved
registers, the harness's bookkeeping; about 90/47 for AOCL's kernel in
goto_bench on Zen 5, read off as the intercept of a k_c sweep. Prefetch
instructions, the kernel's and the harness's, count as reads.

The model sees only the prefetches goto_bench issues, unless told more:
`--kernel-prefetch` takes prefetches inside the micro-kernel, in model.py's
syntax with the lead in k-steps, and `--pf-streams`, `--pf-min-run`,
`--pf-level` describe the hardware prefetcher as in model.py. A kernel may
prefetch only above some k_c; end the spec in `@KMIN` for that. A lead of `k`
(or `kc`; `2kc` is two calls' worth, `0.5kc` half of one) means the whole
k-loop: a prefetch issued at the call's start, as BLIS's
armv8a 8x6 dgemm kernel does for its C rows (`ir:C:L1:keep:k`). AOCL's 8x24
dgemm kernel prefetches its C tile only once k/4 exceeds its 24-iteration
tail, i.e. for k_c >= 128, then 100-128 k-steps ahead: `ir:C:L1:keep:100@128`.
The SpacemiT X60's 2vx14 dgemm kernel (k unrolled by 2) issues `prefetch.w`
for its 14 C columns when 7 iterations remain, i.e. 14 k-steps ahead, from
k >= 18: `ir:C:L1:keep:14@18`. It loads B as scalars for `vfmacc.vf`, one load
per element, which the model counts as a broadcast (`--bcast B`); its 256-bit
`vle64` count two L1 accesses each, so its loads are counted at `--vl 16`.
Below that it prefetches nothing, so its C fetch is exposed every call.

The model charges that exposed fetch in series with the k-loop: the C tile's
lines from beyond L1, in waves through the tightest buffer pool on their way,
times the latency left after any prefetch, less `--lookahead` (how far ahead
of the epilogue the core issues the C loads; default 0). Later waves wait in
the buffers inside that pool -- in L1's MABs, if L2's buffers are the limit --
and the predicted `L1D_MISS_OCCUPANCY` includes that wait. With a memory-bound
C, this term usually binds, and its two unknowns separate on the counters: the
mean L1 miss latency (`L1D_MISS_OCCUPANCY / L1D_CACHE_REFILL`) gives the
latency, and the time per call then gives the waves, i.e. the buffer count.

Whether a software prefetch into an outer level also holds the inner levels'
fill buffers until it lands is implementation-dependent. The model assumes it
does not; `--sw-pf-holds-inner` says it does. It changes occupancy, not the
epilogue's wait. On Zen 5 with `ir:C:L2`, L1 miss occupancy matches only with
it set -- but not because `PREFETCHT1` holds L1 buffers: it fills nothing in
L1 (`L1D_SW_PREFETCH_REFILL` is 0), and most find their line already in
flight. The L1 hardware prefetcher fetches C into L1 -- its stride
prefetcher follows C's rows from call to call -- and so C's whole trip is in
L1's buffers anyway, which the switch happens to charge.

Levels outside the core's clock domain, main memory above all, are fixed in
time, so their cycles move with the clock. Give their latency in ns and their
bandwidth in GB/s (`INF:::60G/30G:::55ns`): `run` converts them at each run's
own clock -- cycles over time, shown in the GHz column -- and `model.py` at
`--ghz`. The calibration runs in L1, in core cycles, so the clock does not
move it; a sibling hyperthread busy on the same core does.

The residency test charges each streamer what one iteration of the loop
touches: for `jr` the whole C strip (m_c x n_r), for `ic` the whole C block
(m_c x n_c), because under LRU every line of it competes with the invariant
until the invariant is used again. A_c and B_c that fit alone can be evicted
by C streaming past them.

A block that is not a multiple of the register block wastes compute: the
micro-kernel always computes a whole m_r x n_r tile, so with m_c = 4 and
m_r = 8 every call is half empty, and the model charges it as a whole call.
For experiments that should stress memory rather than edge handling, keep
m_c a multiple of m_r -- m_c = m_r is the least reuse of B without waste.

Pick configurations by what they can test: a sweep the model predicts to be
compute-bound everywhere checks the peak and per-call cost, not the memory
parameters. Small k_c with C streaming from memory is where the miss buffers
and latencies bind; run the model first and look at its `model bound`. Runs with a packing placement other than
BLIS's are flagged: the model's in-place repacking rule assumes it.

The bandwidths, latencies and buffer counts in `--cache` are not in sysfs and
come from microbenchmarks or vendor data; `detect-cache` fills in the rest.

Verification
------------

Build with `-DCMAKE_CXX_FLAGS=-DGOTOBENCH_CHECK_PREFETCH` to abort if any
prefetched line falls outside the buffer its spec targets -- a wrong prefetch
address never faults, so it is otherwise invisible. `--verify` then also
prints how many lines were prefetched.
