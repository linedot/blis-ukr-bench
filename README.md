What is this?
=============

BLIS microkernel benchmarker. Instead of benchmarking the whole GEMM, it runs the microkernel for multiple iterations in memory regions of increasing size. This allows:

- Testing the level of microarchitectural optimization in the microkernel decoupled from the complex loop/packing structure of a full GEMM/BLAS3 implementation.
- Testing how well the different cache/memory levels can keep the compute on the processor fed, given the max. compute utilization of the microkernel

How to build:
-------------

Example 1 (System blis, which happens to be AOCL-BLIS):

```
git clone --recurse-submodules https://github.com/linedot/blis-ukr-bench.git
cd blis-ukr-bench
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PERF_COUNTERS=ON -DPC_BACKEND=perf ..
cmake --build .
```

Example 2 (Custom BLIS directory):

```
git clone --recurse-submodules https://github.com/linedot/blis-ukr-bench.git
cd blis-ukr-bench
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPC_BACKEND=perf -DUSE_PERF_COUNTERS=ON -DBLIS_ROOT=/NVMEData/tmp/blis-gen-avx512-8x3v ..
cmake --build .
```

How to use:
-----------

```
usage: ./bench <alignment> <start_offset> <offset_step> <byte_factor> <measurements> <max_k_power> <add_k>
```


1 core, up to 2^14 bytes, align to 128:

```
ᐅ OMP_NUM_THREADS=1 OMP_PROC_BIND=true OMP_PLACES=cores ./bench 128 0 64 16000000 1000 14 0
# ukr implementation: optimzd
# ukr size:            8x24
# threads:             1
Size[Byte],[GFLOP/s],FLOPS,cycles/iter,l1d_read_miss/iter
2048,149.33333333333334,1344,48,0
2560,162.46153846153845,2112,72,0
3584,173.71428571428572,3648,120,0
5632,176.8421052631579,6720,216,0
9728,176.21917808219177,12864,408,0
17920,175.8881118881119,25152,796,0
34304,178.23655913978496,49728,1560,0
67072,175.94306049822063,98880,3107,1032
132608,176.52999104744853,197184,6170,2093
263680,176.98516853932585,393792,12316,4125
525824,177.29398513178643,787008,24608,8238
1050112,174.61325047164576,1573440,49476,16432
2098688,167.78498293515358,3146304,102194,32866
```

16 cores, up to 2^22 bytes, align to 4096, offset each data block by 64 bytes

```
ᐅ OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=true ./bench 4096 0 64 160000000 80 22 0
# ukr implementation: optimzd
# ukr size:            8x24
# threads:             16
Size[Byte],[GFLOP/s],FLOPS,cycles/iter,l1d_read_miss/iter
2048,1654.1538461538462,21504,48,0
2560,1877.3333333333333,33792,72,0
3584,2084.5714285714284,58368,120,0
5632,2150.4,107520,216,0
9728,2213.1612903225805,205824,408,0
17920,2211.164835164835,402432,792,0
34304,2247.593220338983,795648,1560,0
67072,2087.1767810026386,1582080,3096,1032
132608,2118.834116856951,3154944,6169,2094
263680,2129.3247718823927,6300672,12314,4142
525824,2136.069211195929,12592128,24610,8243
1050112,1937.4357395721102,25175040,50018,16435
2098688,1702.6030371698178,50340864,103155,32873
4195840,1548.6887470194602,100672512,227641,65748
8390144,1341.4964253113278,201335808,583301,131220
16778752,143.90693601851274,402662400,9414662,262243
33555968,106.59224246387046,805315584,38465578,524430
67110400,103.56924833903065,1610621952,79495028,1048779
134219264,102.78872810497944,3221234688,162856771,2097502
268436992,102.5249975950041,6442460160,319375707,4194924
536872448,102.70118005944595,12884911104,638290113,8389951
```

Analysis script
---------------

See [ANALYSIS.md](ANALYSIS.md)
