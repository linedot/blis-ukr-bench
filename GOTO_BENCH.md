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
  `lead` is in micro-kernel calls before the end of the loop's iteration;
  default: issued at its start, i.e. one iteration ahead. C is prefetched for
  writing. The k-loop is inside the kernel and cannot be prefetched from here.
* `--c-layout rm|cm`, `--beta general|one|zero|X`, `--alpha X`. A and B are
  column-major.
* `--events E,E,...` / `--events-file F`, `--group N`, `--cycles-event E` --
  counters are measured in groups of N, each its own pass, each led by the
  cycle counter: one group with more events than the PMU has counters would
  never be scheduled.
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
* x86-64: `prefetcht0/t1/t2` choose L1/L2/L3; `strm` is `prefetchnta` and
  write intent is `prefetchw`, both without level control.
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
```

Reads `/sys/bus/event_source/devices/*` and, if installed, `perf list -j`, and
computes encodings as perf does. Each event has candidates in order of
preference, checked against perf list on Kunpeng 920, Zen 5 and SpacemiT K1;
a candidate that only approximates the model's meaning is printed with a note
(`[demand misses only]`). Kernel generic events are taken only where perf
lists them as supported, and generic last-level-cache events go to whichever
level is last here (from sysfs: L2 on K1, L3 on Zen 5).

Besides the model's events (L1D/L2D accesses, refills and write-backs, the
`_RD`/`_WR` splits, and `L1D_MISS_OCCUPANCY`) it maps a few the model does not
predict but that explain a run: `STALL_BACKEND`, `STALL_BACKEND_MEM`,
`LOAD_QUEUE_STALL`, `L1D_SW_PREFETCH_REFILL`.

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
validate.py calibrate --bench build/goto_bench [--ghz G] -o calib.json
validate.py run --bench build/goto_bench --calib calib.json \
    --cache <model.py spec for this machine> \
    --size 2000 4000 --kc 128 256 --mc 96 --nc 4096 --top jc \
    [--prefetch ir:C:L2] [--events-map events.json] -o results.csv
```

`calibrate` fits the micro-kernel at k small enough to stay in L1 as
t(k) = C + b*k, giving `--peak` (2*mr*nr/b) and the per-call cost C, which the
model charges as its epilogue; then times one A_c and one B_c pack in cache for
`--pack-rate`. Without a cycle counter, cycles come from ns * `--ghz`, and
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

The model sees only the prefetches goto_bench issues, unless told more:
`--kernel-prefetch` takes prefetches inside the micro-kernel, in model.py's
syntax with the lead in k-steps, and `--pf-streams`, `--pf-min-run`,
`--pf-level` describe the hardware prefetcher as in model.py. A kernel may
prefetch only above some k_c; end the spec in `@KMIN` for that. AOCL's 8x24
dgemm kernel prefetches its C tile only once k/4 exceeds its 24-iteration
tail, i.e. for k_c >= 128, then 100-128 k-steps ahead: `ir:C:L1:keep:100@128`.
Below that it prefetches nothing, so its C fetch is exposed every call.

The model charges that exposed fetch in series with the k-loop: the C tile's
lines from beyond L1, in waves through the tightest buffer pool on their way,
times the latency left after any prefetch, less `--lookahead` (how far ahead
of the epilogue the core issues the C loads; default 0). With a memory-bound
C, this term usually binds, and its two unknowns separate on the counters: the
mean L1 miss latency (`L1D_MISS_OCCUPANCY / L1D_CACHE_REFILL`) gives the
latency, and the time per call then gives the waves, i.e. the buffer count.

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
