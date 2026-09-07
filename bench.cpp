#include "dynamic_aligned_allocator.hpp"

#include "performance_counters/performance_counters.hpp"

#include <omp.h> // Include this first
extern "C" {
    #include <blis.h>
}

#include <chrono>
#include <cstdint>
#include <limits>
#include <print>
#include <string>
#include <vector>


int main(int argc, char *argv[])
{

    if (argc != 8)
    {
        std::print("usage: {} <alignment> <start_offset> <offset_step> <byte_factor> <measurements> <max_k_power> <add_k>\n", argv[0]);
        return -1;
    }
    using hrc=std::chrono::high_resolution_clock;


    std::vector<std::string> counters = {{"CYCLES"},{"L1D_READ_MISS"}};
    performance_counters pc(counters);

    bli_init();


    auto c = bli_info_get_gemm_ukr_impl_string(BLIS_NAT, BLIS_DOUBLE);

    std::print("# ukr implementation: {}\n",c);


    #if defined(AOCL_BLIS)
    auto* cntx = bli_gks_query_nat_cntx();
    #else
    auto* cntx = bli_gks_query_cntx();
    #endif

    #if defined(AOCL_BLIS)
    auto ukr = reinterpret_cast<gemm_ukr_vft>(bli_cntx_get_l3_nat_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx));
    #else
    auto ukr = reinterpret_cast<gemm_ukr_ft>(bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx));
    #endif

    auto mr = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    auto nr = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);


    // Need to somehow infer this. Not sure how
    #if defined(ROWSTRIDE)
    inc_t rs_c = nr;
    inc_t cs_c = 1;
    #elif defined(COLSTRIDE)
    inc_t rs_c = 1;
    inc_t cs_c = mr;
    #else
    inc_t rs_c = 1;
    inc_t cs_c = 1;
    #endif


    std::print("# ukr size:           {}x{}\n",mr,nr);

    double alpha = 2.1;
    double beta = 1.3;
    const std::uint64_t byte_factor = std::strtoull(argv[4], nullptr, 10);
    const std::uint64_t measurements = std::strtoull(argv[5], nullptr, 10);
    const std::uint64_t max_k_power = std::strtoull(argv[6], nullptr, 10);
    const std::uint64_t add_k = std::strtoull(argv[7], nullptr, 10);

    const std::uint64_t alignment = std::strtoull(argv[1], nullptr, 10);
    const std::uint64_t start_offset = std::strtoull(argv[2], nullptr, 10);
    const std::uint64_t offset_step = std::strtoull(argv[3], nullptr, 10);

    auto alloca = archcomp::dynamic_aligned_allocator<double>(alignment,start_offset);
    auto allocb = archcomp::dynamic_aligned_allocator<double>(alignment,start_offset+offset_step);
    auto allocc = archcomp::dynamic_aligned_allocator<double>(alignment,start_offset+2*offset_step);
    //auto alloca = archcomp::dynamic_aligned_allocator<double>(64);
    //auto allocb = archcomp::dynamic_aligned_allocator<double>(64);
    //auto allocc = archcomp::dynamic_aligned_allocator<double>(64);

        
    std::vector<double,archcomp::dynamic_aligned_allocator<double>> A(alloca);
    std::vector<double,archcomp::dynamic_aligned_allocator<double>> B(allocb);
    std::vector<double,archcomp::dynamic_aligned_allocator<double>> C(allocc);

    std::print("Size[Byte],[GFLOP/s],FLOPS,cycles,l1d_read_miss\n");
    for (std::uint64_t kpow = 1; kpow < max_k_power; kpow++)
    {
        std::uint64_t k = std::max(1UL,(1<< kpow) + add_k);
        double flops = 2.0*mr*nr*k + 3.0*mr*nr;
        A.resize(mr*k);
        B.resize(nr*k);
        C.resize(mr*nr);
        std::uint64_t size = A.size()*sizeof(double)+
                             B.size()*sizeof(double)+
                             C.size()*sizeof(double);

        std::uint64_t iterations = std::max(byte_factor/size,10UL);

        std::uint64_t max_ns = 0;
        std::uint64_t min_ns = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t avg_ns = 0;

        auxinfo_t aux;
        bli_auxinfo_set_next_a(A.data(), &aux);
        bli_auxinfo_set_next_b(B.data(), &aux);


        // Warmup
        for (std::uint64_t m = 0; m < iterations/4; m++)
        {
            #if defined(AOCL_BLIS)
            ukr( k,
            #else
            ukr( mr, nr, k,
            #endif
                &alpha, A.data(), B.data(),
                &beta, C.data(), rs_c, cs_c,
                &aux, cntx);
        }


        for (std::uint64_t i = 0; i < measurements; i++)
        {
            auto start = hrc::now();
            pc.tic();
            for (std::uint64_t m = 0; m < iterations; m++)
            {
                #if defined(AOCL_BLIS)
                ukr( k,
                #else
                ukr( mr, nr, k,
                #endif
                    &alpha, A.data(), B.data(),
                    &beta, C.data(), rs_c, cs_c,
                    &aux, cntx);
            }
            pc.toc_stat();
            auto stop = hrc::now();

            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            avg_ns += dur.count()/iterations;
            max_ns = std::max(max_ns, dur.count()/iterations);
            min_ns = std::min(min_ns, dur.count()/iterations);
        }
        std::uint64_t min_cycles = 0;
        std::uint64_t min_l1d_miss = 0;
        auto results = pc.get_counter_statistics();
        for (const auto& res : results)
        {
            auto [name,min,avg,max] = res;

            if("CYCLES" == name)
            {
                min_cycles = min;
            }
            if("L1D_READ_MISS" == name)
            {
                min_l1d_miss = min;
            }
        }
        std::print("{},{},{},{},{}\n",size,flops/min_ns,flops,min_cycles/iterations,min_l1d_miss/iterations);
        pc.reset_counter_storage();
    }


    
    return 0;
}
