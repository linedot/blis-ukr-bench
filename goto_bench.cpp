// GOTO-nest benchmark: BLIS's own micro-kernel and pack kernel, driven by a
// loop nest that can be cut at any rung and rearranged -- packing moved,
// prefetches inserted -- so that each rung of model.py's ladder can be
// measured on its own, bottom up.  Flags follow model.py wherever the
// concept is the same, so any run can be replayed through the model.
//
//   goto_bench --size 4000 4000 4000 --kc 160 --mc 256 --nc 2000
//              --top jc --pack A:ic,B:pc --prefetch ir:C:L2 jr:B:L2:keep:1
//
// One configuration per run.  Output is '# key: value' metadata, then a CSV
// header and one row -- what validate.py reads.

#include "harness/goto_nest.hpp"
#if defined(GOTOBENCH_HAVE_COUNTERS)
#include "performance_counters/performance_counters.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

using namespace gotobench;

namespace {

struct options
{
    nest_config cfg;
    std::string top = "jc";
    std::string beta_mode = "general";
    std::vector<std::string> events;
    std::string cycles_event = "CYCLES";
    std::size_t group = 4;
    long reps = 0;
    double min_time = 0.5;
    long warmup = 2;
    long inner = 0;
    bool verify = false;
    bool verify_lib = false;
    int cpu = -1;
    std::string label;
    std::string kc_text, mc_text, nc_text;   // resolved once the kernel's mr, nr are known
};

[[noreturn]] void usage(const char* prog, const std::string& why = "")
{
    if (!why.empty()) std::fprintf(stderr, "error: %s\n\n", why.c_str());
    std::fprintf(stderr, R"(usage: %s --size M N K --kc KC --mc MC --nc NC [options]

  KC, MC, NC          N, or a multiple of the micro-tile: 4mr, 200nr, mr

region
  --top RUNG          outermost rung executed: ukr ir jr ic pc jc (default jc),
                      or packA / packB to time one block's packing alone
  --pack A:R,B:R      where packing happens: A at ic pc jc pre, B at pc jc pre
                      (default A:ic,B:pc, as BLIS; 'pre' packs once, untimed)
  --c-layout rm|cm    C row- or column-major (default rm); A and B column-major
  --beta MODE|X       general (alpha 2.1, beta 1.3), one, zero, or a number
  --alpha X           (default 2.1)
  --prefetch PF...    loop:operand:target[:hint[:lead]], as in model.py, with
                        loop    ir jr ic pc jc
                        target  L1 L2 L3
                        hint    keep (default) or strm
                        lead    micro-kernel calls before the end of the loop's
                                iteration (default: issued at its start), as N
                                or Nkc; it may stand in the hint's place:
                                ir:C:L2:2kc is ir:C:L2:keep:2
                      C is prefetched for writing

counters
  --events E,E,...    e.g. CYCLES,L1D_READ_MISS,RAW:0x17
  --events-file F     one per line, as written by probe_events.py
  --group N           counters per group, each group its own pass (default 4)
  --cycles-event E    cycle counter added to every group (default CYCLES)

timing
  --warmup N          untimed region runs before measuring (default 2)
  --inner N           region runs per timed measurement (default: enough for
                      ~200 us, so small regions are not timer-bound); every
                      reported value is per region run
  --min-time S        seconds of timed runs per pass (default 0.5)
  --reps N            timed runs per pass (default: enough for --min-time)

checking
  --verify            check the region against a plain triple loop, then exit
  --verify-lib        check it against the linked library's cblas_dgemm, or
                      its bli_dgemm if it has no CBLAS layer, then exit; fast
                      enough for full-size problems.  Both may be given.

system
  --cpu N             pin to this CPU
  --hugepages         2 MiB-aligned buffers with madvise(MADV_HUGEPAGE)
  --line BYTES        cache line for prefetching (default: from the system)
  --label TEXT        carried into the output row
  -h, --help          this text
)",
        prog);
    std::exit(2);
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

// A block size: N, or a multiple of the micro-tile, Nmr / Nnr ("4mr", "200nr", "mr").
long long parse_block(const std::string& t, long long mr, long long nr)
{
    std::string s;
    for (char c : t) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    long long unit = 1;
    if (s.size() >= 2 && (s.compare(s.size() - 2, 2, "mr") == 0 || s.compare(s.size() - 2, 2, "nr") == 0))
    {
        unit = s[s.size() - 2] == 'm' ? mr : nr;
        s.resize(s.size() - 2);
        if (s.empty()) return unit;
    }
    std::size_t used = 0;
    long long n = 0;
    try { n = std::stoll(s, &used); } catch (const std::exception&) { used = 0; }
    if (used != s.size() || s.empty() || n <= 0)
        throw std::invalid_argument("block size " + t + ": give N, Nmr or Nnr");
    return n * unit;
}

// A prefetch lead in calls: N, or Nkc / Nk (N calls of k_c k-steps), "kc" for one.
long parse_lead(const std::string& t, const std::string& tok)
{
    if (t.empty()) return 0;
    std::string s;
    for (char c : t) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    bool suffix = false;
    for (const char* suf : {"kc", "k"})
    {
        const std::size_t n = std::char_traits<char>::length(suf);
        if (s.size() >= n && s.compare(s.size() - n, n, suf) == 0)
        {
            s.resize(s.size() - n);
            suffix = true;
            break;
        }
    }
    if (suffix && s.empty()) return 1;
    std::size_t used = 0;
    long n = -1;
    try { n = std::stol(s, &used); } catch (const std::exception&) { used = 0; }
    if (used != s.size() || s.empty() || n < 0)
        throw std::invalid_argument("prefetch " + tok + ": lead must be a whole number of "
                                    "calls, N or Nkc");
    return n;
}

prefetch_spec parse_prefetch(const std::string& tok)
{
    const auto f = split(tok, ':');
    if (f.size() < 3 || f.size() > 5)
        throw std::invalid_argument("prefetch " + tok + ": need loop:operand:target[:hint[:lead]]");
    prefetch_spec s;
    s.text = tok;
    if (f[0] == "k")
        throw std::invalid_argument("prefetch " + tok + ": the k-loop is inside the micro-kernel, "
                                    "which the harness cannot change; use a kernel variant");
    s.loop = parse_rung(f[0]);
    if (s.loop < R_IR || s.loop > R_JC)
        throw std::invalid_argument("prefetch " + tok + ": loop must be ir, jr, ic, pc or jc "
                                    "(the k-loop is inside the micro-kernel)");
    if (f[1].size() != 1 || std::string("ABC").find(f[1][0]) == std::string::npos)
        throw std::invalid_argument("prefetch " + tok + ": operand must be A, B or C");
    s.operand = f[1][0];
    if (f[2] == "L1") s.kind.level = pf_level::L1;
    else if (f[2] == "L2") s.kind.level = pf_level::L2;
    else if (f[2] == "L3") s.kind.level = pf_level::L3;
    else throw std::invalid_argument("prefetch " + tok + ": target must be L1, L2 or L3");
    std::string hint = "keep", lead;
    if (f.size() > 3 && !f[3].empty())
    {
        if (f[3] == "keep" || f[3] == "strm") hint = f[3];
        else if (f.size() == 4) lead = f[3];            // loop:operand:target:lead
        else throw std::invalid_argument("prefetch " + tok + ": hint must be keep or strm");
    }
    if (f.size() > 4) lead = f[4];
    s.kind.stream = hint == "strm";
    s.kind.write = s.operand == 'C';
    s.lead = parse_lead(lead, tok);
    return s;
}

std::size_t system_line()
{
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_LINESIZE)
    const long sc = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sc > 0) return static_cast<std::size_t>(sc);
#endif
    std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    std::size_t fs = 0;
    if (f >> fs && fs > 0) return fs;
    return 64;
}

options parse(int argc, char** argv)
{
    options o;
    o.cfg.line = 0;
    std::string pack = "A:ic,B:pc";
    auto need = [&](int& i, int n) {
        if (i + n >= argc) usage(argv[0], std::string(argv[i]) + " needs " + std::to_string(n) + " value(s)");
    };
    bool have_size = false, have_alpha = false;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "--size") { need(i, 3); o.cfg.m = std::stoll(argv[++i]); o.cfg.n = std::stoll(argv[++i]);
                             o.cfg.k = std::stoll(argv[++i]); have_size = true; }
        else if (a == "--kc") { need(i, 1); o.kc_text = argv[++i]; }
        else if (a == "--mc") { need(i, 1); o.mc_text = argv[++i]; }
        else if (a == "--nc") { need(i, 1); o.nc_text = argv[++i]; }
        else if (a == "--top") { need(i, 1); o.top = argv[++i]; }
        else if (a == "--pack") { need(i, 1); pack = argv[++i]; }
        else if (a == "--c-layout") { need(i, 1); const std::string v = argv[++i];
                                      if (v != "rm" && v != "cm") usage(argv[0], "--c-layout is rm or cm");
                                      o.cfg.c_row_major = v == "rm"; }
        else if (a == "--beta") { need(i, 1); o.beta_mode = argv[++i]; }
        else if (a == "--alpha") { need(i, 1); o.cfg.alpha = std::stod(argv[++i]); have_alpha = true; }
        else if (a == "--prefetch")
        {
            while (i + 1 < argc && argv[i + 1][0] != '-')
                o.cfg.prefetches.push_back(parse_prefetch(argv[++i]));
        }
        else if (a == "--events") { need(i, 1); for (auto& e : split(argv[++i], ',')) o.events.push_back(e); }
        else if (a == "--events-file")
        {
            need(i, 1);
            std::ifstream f(argv[++i]);
            if (!f) usage(argv[0], std::string("cannot read ") + argv[i]);
            std::string line;
            while (std::getline(f, line))
            {
                const auto h = line.find('#');
                if (h != std::string::npos) line.resize(h);
                std::stringstream ss(line);
                std::string e;
                if (ss >> e) o.events.push_back(e);
            }
        }
        else if (a == "--group") { need(i, 1); o.group = std::max(1, std::stoi(argv[++i])); }
        else if (a == "--cycles-event") { need(i, 1); o.cycles_event = argv[++i]; }
        else if (a == "--reps") { need(i, 1); o.reps = std::stol(argv[++i]); }
        else if (a == "--min-time") { need(i, 1); o.min_time = std::stod(argv[++i]); }
        else if (a == "--warmup") { need(i, 1); o.warmup = std::stol(argv[++i]); }
        else if (a == "--inner") { need(i, 1); o.inner = std::stol(argv[++i]); }
        else if (a == "--verify") o.verify = true;
        else if (a == "--verify-lib") o.verify_lib = true;
        else if (a == "--cpu") { need(i, 1); o.cpu = std::stoi(argv[++i]); }
        else if (a == "--hugepages") o.cfg.hugepages = true;
        else if (a == "--line") { need(i, 1); o.cfg.line = std::stoul(argv[++i]); }
        else if (a == "--label") { need(i, 1); o.label = argv[++i]; }
        else if (a == "-h" || a == "--help") usage(argv[0]);
        else usage(argv[0], "unknown option " + a);
    }
    if (!have_size || o.kc_text.empty() || o.mc_text.empty() || o.nc_text.empty())
        usage(argv[0], "--size, --kc, --mc and --nc are required");
    if (o.top != "packA" && o.top != "packB") o.cfg.top = parse_rung(o.top);
    if (o.cfg.top == R_PRE) usage(argv[0], "--top cannot be pre");
    for (const auto& item : split(pack, ','))
    {
        const auto kv = split(item, ':');
        if (kv.size() != 2 || (kv[0] != "A" && kv[0] != "B"))
            usage(argv[0], "--pack takes A:<rung>,B:<rung>");
        (kv[0] == "A" ? o.cfg.pack_a : o.cfg.pack_b) = parse_rung(kv[1]);
    }
    if (o.beta_mode == "general") { o.cfg.beta = 1.3; }
    else if (o.beta_mode == "one") { o.cfg.beta = 1.0; }
    else if (o.beta_mode == "zero") { o.cfg.beta = 0.0; }
    else { o.cfg.beta = std::stod(o.beta_mode); }
    if (!have_alpha) o.cfg.alpha = 2.1;
    if (!o.cfg.line) o.cfg.line = system_line();
    return o;
}

template <class F>
double time_ns(F&& f)
{
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

struct event_result { std::string name; double min = 0, avg = 0; };

} // namespace

int main(int argc, char** argv)
{
    options o;
    try { o = parse(argc, argv); }
    catch (const std::exception& e) { usage(argv[0], e.what()); }

#if defined(__linux__)
    if (o.cpu >= 0)
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(o.cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            std::fprintf(stderr, "# warning: could not pin to CPU %d\n", o.cpu);
    }
#endif

    blis_kernels ker;
    try
    {
        o.cfg.kc = parse_block(o.kc_text, ker.mr, ker.nr);
        o.cfg.mc = parse_block(o.mc_text, ker.mr, ker.nr);
        o.cfg.nc = parse_block(o.nc_text, ker.mr, ker.nr);
    }
    catch (const std::exception& e) { usage(argv[0], e.what()); }
    const bool pack_mode = o.top == "packA" || o.top == "packB";
    if (pack_mode) o.cfg.top = R_UKR;   // nothing of the nest runs; just pack

    goto_nest* nest = nullptr;
    try { nest = new goto_nest(ker, o.cfg); }
    catch (const std::exception& e) { usage(argv[0], e.what()); }

    if (o.verify || o.verify_lib)
    {
        if (pack_mode)
            usage(argv[0], "--verify and --verify-lib apply to the nest, not to --top packA/packB");
        bool ok = true;
        auto report = [&](const std::string& what, double err) {
            ok = ok && err < 1e-10;
            std::printf("# %s: m_exec=%lld n_exec=%lld k_exec=%lld max relative error %.3g -> %s\n",
                        what.c_str(), static_cast<long long>(nest->m_exec()),
                        static_cast<long long>(nest->n_exec()),
                        static_cast<long long>(nest->k_exec()), err, err < 1e-10 ? "OK" : "FAILED");
        };
        if (o.verify) report("verify", nest->verify(goto_nest::reference::loop));
        if (o.verify_lib)
            report(std::string("verify-lib (") + goto_nest::library_gemm() + ")",
                   nest->verify(goto_nest::reference::library));
#if defined(GOTOBENCH_CHECK_PREFETCH)
        std::printf("# prefetched lines: %ld\n", nest->prefetched);
#endif
        delete nest;
        return ok ? 0 : 1;
    }

    auto region = [&] {
        if (o.top == "packA") nest->run_pack_a();
        else if (o.top == "packB") nest->run_pack_b();
        else nest->run();
    };
    for (long w = 0; w < o.warmup; w++) region();
    long inner = o.inner;
    if (inner <= 0)
        inner = std::max(1L, static_cast<long>(200e3 / std::max(time_ns(region), 1.0)));
    auto body = [&] { for (long i = 0; i < inner; i++) region(); };
    long reps = o.reps;
    if (reps <= 0)
    {
        const double one = std::max(time_ns(body), 1.0);
        reps = std::clamp(static_cast<long>(o.min_time * 1e9 / one) + 1, 3L, 100000L);
    }

    // counter groups, each led by the cycle counter, each its own pass
    std::vector<std::vector<std::string>> groups;
    std::vector<std::string> rest;
    for (const auto& e : o.events)
        if (e != o.cycles_event) rest.push_back(e);
#if !defined(GOTOBENCH_HAVE_COUNTERS)
    if (!o.events.empty())
        std::fprintf(stderr, "# warning: built without counters (-DUSE_PERF_COUNTERS=ON); timing only\n");
    groups.push_back({});
#else
    // An event the kernel cannot open fails its whole group -- and every
    // group carries the cycle counter.  So open each first, the cycle counter
    // alone and every other event beside it: only the ones that fail are
    // lost, and by name.
    auto opens = [](const std::vector<std::string>& g, std::string& why) {
        try { performance_counters probe(g); return true; }
        catch (const std::exception& e) { why = e.what(); return false; }
    };
    std::string why, cyc_ev = o.cycles_event;
    if (!cyc_ev.empty() && !opens({cyc_ev}, why))
    {
        std::fprintf(stderr, "# warning: cycle counter %s unavailable (%s); cycles from timing only\n",
                     cyc_ev.c_str(), why.c_str());
        cyc_ev.clear();
    }
    std::vector<std::string> usable;
    for (const auto& e : rest)
    {
        std::vector<std::string> g;
        if (!cyc_ev.empty()) g.push_back(cyc_ev);
        g.push_back(e);
        if (opens(g, why)) usable.push_back(e);
        else std::fprintf(stderr, "# warning: counter %s unavailable (%s); skipped\n",
                          e.c_str(), why.c_str());
    }
    if (usable.empty())
        groups.push_back(cyc_ev.empty() ? std::vector<std::string>{} : std::vector<std::string>{cyc_ev});
    for (std::size_t i = 0; i < usable.size(); i += o.group)
    {
        std::vector<std::string> g;
        if (!cyc_ev.empty()) g.push_back(cyc_ev);
        for (std::size_t j = i; j < std::min(usable.size(), i + o.group); j++) g.push_back(usable[j]);
        groups.push_back(g);
    }
#endif

    std::vector<double> ns;
    std::vector<event_result> results;
    std::vector<double> cycles_min;
    for (const auto& g : groups)
    {
#if defined(GOTOBENCH_HAVE_COUNTERS)
        performance_counters* pc = nullptr;
        if (!g.empty())
        {
            try { pc = new performance_counters(g); }
            catch (const std::exception& e)
            {
                std::fprintf(stderr, "# warning: counters %s unavailable (%s); timing only\n",
                             g.size() > 1 ? "group" : g[0].c_str(), e.what());
            }
        }
        for (long r = 0; r < reps + (pc ? 1 : 0); r++)   // the backend drops a pass's first
        {
            if (pc) pc->tic();
            const double t = time_ns(body);
            if (pc) pc->toc_stat();
            if (!pc || r > 0) ns.push_back(t);
        }
        if (pc)
        {
            for (const auto& [name, mn, avg, mx] : pc->get_counter_statistics())
            {
                (void)mx;
                if (name == o.cycles_event) cycles_min.push_back(static_cast<double>(mn));
                else results.push_back({name, static_cast<double>(mn), static_cast<double>(avg)});
            }
            delete pc;
        }
#else
        (void)g;
        for (long r = 0; r < reps; r++) ns.push_back(time_ns(body));
#endif
    }
    std::sort(ns.begin(), ns.end());
    // everything below is per region run
    const double ns_min = ns.front() / inner, ns_med = ns[ns.size() / 2] / inner;
    const double cyc = cycles_min.empty() ? 0.0
                       : *std::min_element(cycles_min.begin(), cycles_min.end()) / inner;
    for (auto& r : results) { r.min /= inner; r.avg /= inner; }

    // -- report ------------------------------------------------------------
    std::printf("# ukr implementation: %s\n", ker.impl);
    std::printf("# ukr size:            %lldx%lld (broadcast %lldx%lld)\n",
                static_cast<long long>(ker.mr), static_cast<long long>(ker.nr),
                static_cast<long long>(ker.bbm), static_cast<long long>(ker.bbn));
    std::printf("# ukr prefers C: %s (BLIS would transpose the problem to match;"
                " --c-layout %s does the same here)\n",
                ker.prefers_rows ? "row-major" : "column-major", ker.prefers_rows ? "rm" : "cm");
    std::printf("# blis default blocking: kc=%lld mc=%lld nc=%lld\n",
                static_cast<long long>(ker.kc), static_cast<long long>(ker.mc),
                static_cast<long long>(ker.nc));
#if defined(__linux__)
    utsname u{};
    if (uname(&u) == 0) std::printf("# machine: %s %s\n", u.machine, u.release);
#endif
    std::printf("# compiler: %s\n", __VERSION__);
    std::printf("# cache line: %zu\n", o.cfg.line);
    std::printf("# footprint: %zu bytes\n", nest->footprint_bytes());
    for (const auto& s : o.cfg.prefetches)
        std::printf("# prefetch %s -> %s\n", s.text.c_str(), describe(s.kind).c_str());

    std::string pf;
    for (const auto& s : o.cfg.prefetches) pf += (pf.empty() ? "" : "+") + s.text;
    std::printf("label,top,m,n,k,mc,nc,kc,mr,nr,pack_a,pack_b,c_layout,alpha,beta,prefetch,"
                "m_exec,n_exec,k_exec,calls,work,reps,inner,ns_min,ns_med,cycles_min");
    for (const auto& r : results) std::printf(",%s_min,%s_avg", r.name.c_str(), r.name.c_str());
    std::printf("\n");
    const double work = o.top == "packA" ? static_cast<double>(nest->pack_a_elems())
                      : o.top == "packB" ? static_cast<double>(nest->pack_b_elems())
                      : nest->flops();
    std::printf("%s,%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%s,%s,%s,%g,%g,%s,"
                "%lld,%lld,%lld,%.0f,%.0f,%ld,%ld,%.1f,%.1f,%.1f",
                o.label.c_str(), o.top.c_str(), static_cast<long long>(o.cfg.m),
                static_cast<long long>(o.cfg.n), static_cast<long long>(o.cfg.k),
                static_cast<long long>(o.cfg.mc), static_cast<long long>(o.cfg.nc),
                static_cast<long long>(o.cfg.kc), static_cast<long long>(ker.mr),
                static_cast<long long>(ker.nr), rung_name(o.cfg.pack_a), rung_name(o.cfg.pack_b),
                o.cfg.c_row_major ? "rm" : "cm", o.cfg.alpha, o.cfg.beta, pf.c_str(),
                static_cast<long long>(nest->m_exec()), static_cast<long long>(nest->n_exec()),
                static_cast<long long>(nest->k_exec()), pack_mode ? 0.0 : nest->calls(), work,
                reps, inner, ns_min, ns_med, cyc);
    for (const auto& r : results) std::printf(",%.1f,%.1f", r.min, r.avg);
    std::printf("\n");
    delete nest;
    return 0;
}
