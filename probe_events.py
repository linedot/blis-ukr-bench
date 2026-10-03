#!/usr/bin/env python3
"""
Find the performance events this machine offers, and map the ones model.py
predicts onto them, in the syntax goto_bench takes.

Sources, in order of preference:

  sysfs      /sys/bus/event_source/devices/<pmu>/{type,events,format}: what
             the kernel knows the PMU implements.  Encodings are computed as
             perf does, from each event's terms and its PMU's bit layout.
  perf list  `perf list -j`, when perf is installed: adds the vendor events
             perf carries tables for (AMD, Intel, many Arm and RISC-V parts)
             that sysfs does not list.
  generic    the kernel's PERF_TYPE_HW_CACHE events (L1D_READ_MISS, ...),
             which every architecture maps somehow -- a fallback whose exact
             meaning is the kernel's choice for that PMU.

How trustworthy a mapping is depends on the architecture:

  Arm        the model's events are architected common events, with fixed
             numbers (L2D_CACHE_REFILL is 0x17 on every Armv8 core), and sysfs
             lists exactly those the CPU implements.  Reliable.
  x86        vendor-specific; the candidates below are perf's names for them,
             to be confirmed on the machine.
  RISC-V     whatever the vendor's SBI firmware and perf tables expose; the
             candidates are best guesses.  Check with --list.

Output: an events file for `goto_bench --events-file`, and a JSON map from the
model's event names to the chosen specs, for validate.py.

    probe_events.py                     # show the mapping
    probe_events.py --write events      # writes events.txt and events.json
    probe_events.py --list              # every cache/memory event found
"""

import argparse
import ctypes
import fcntl
import glob
import json
import os
import platform
import re
import signal
import shutil
import subprocess
import sys

# The model's events (model.py counters()), and a few it does not predict but
# that explain a run.  Candidates are tried in order and the first one this
# machine has wins: (event name as sysfs or perf list spells it, any case;
# a note when it only approximates the model's meaning).  'generic:' names a
# kernel generic event, taken only where perf lists it as supported.  Names
# checked against perf list on Kunpeng 920, AMD Zen 5 and SpacemiT K1 (X60).
CANONICAL = {
    'CPU_CYCLES': [('cpu_cycles', ''), ('cycles', ''), ('generic:CYCLES', '')],
    'INST_RETIRED': [('inst_retired', ''), ('instructions', ''),
                     ('generic:INSTRUCTIONS', '')],
    # memory uops: loads + stores, and each part
    'L1D_CACHE': [
        ('l1d_cache', ''),                                     # Arm 0x04
        ('l1d_access', ''),                                    # SpacemiT X60
        ('ls_dispatch.all', 'memory ops dispatched, speculative ones too')],
    'L1D_CACHE_RD': [
        ('l1d_cache_rd', ''),                                  # Arm 0x40
        ('l1d_load_access', ''),                               # X60
        ('ls_dispatch.ld_dispatch', 'load ops dispatched, speculative ones too'),
        ('mem_inst_retired.all_loads', ''),                    # Intel
        ('generic:L1D_READ_ACCESS', '')],
    'L1D_CACHE_WR': [
        ('l1d_cache_wr', ''),                                  # Arm 0x41
        ('l1d_store_access', ''),                              # X60
        ('ls_dispatch.store_dispatch', 'store ops dispatched, speculative ones too'),
        ('mem_inst_retired.all_stores', ''),                   # Intel
        ('generic:L1D_WRITE_ACCESS', '')],
    # lines brought into L1D, prefetched ones included
    'L1D_CACHE_REFILL': [
        ('l1d_cache_refill', 'whether refills by prefetch instructions count is '
                             'implementation-defined; TSV110 does not count them'),
        ('l1d.replacement', ''),                               # Intel
        ('ls_any_fills_from_sys.all', ''),                     # AMD Zen 4/5
        ('ls_refills_from_sys.all', ''),                       # AMD Zen 2/3
        ('l1d_miss', 'demand misses; prefetched lines are l1d_prefetch_refill'),
        ('generic:L1D_READ_MISS', 'load misses only')],
    'L1D_CACHE_WB': [('l1d_cache_wb', '')],                    # Arm 0x15
    # misses in flight, summed over cycles: fill-buffer occupancy
    'L1D_MISS_OCCUPANCY': [
        ('ls_alloc_mab_count', ''),                            # AMD Zen 4/5
        ('l1d_pend_miss.pending', '')],                        # Intel
    # accesses arriving at L2: L1 refills + L1 write-backs, and each part
    'L2D_CACHE': [('l2d_cache', '')],                          # Arm 0x16
    'L2D_CACHE_RD': [
        ('l2d_cache_rd', ''),                                  # Arm 0x50
        ('l2_request_g1.all_dc', ''),                          # AMD Zen
        ('l2_load_access', 'loads only'),                      # X60
        ('l2_rqsts.references', 'L2 prefetcher requests too')],   # Intel
    'L2D_CACHE_WR': [
        ('l2d_cache_wr', ''),                                  # Arm 0x51
        ('l2_store_access', 'store accesses: unverified as L1 write-backs')],
    'L2D_CACHE_REFILL': [
        ('l2d_cache_refill', ''),                              # Arm 0x17
        ('l2_lines_in.all', ''),                               # Intel
        ('l2_fill_rsp_src.all', ''),                           # AMD Zen 5
        ('l2_cache_req_stat.ic_dc_miss_in_l2',
         'demand misses only, instruction fetches included'),
        ('l2_load_miss', 'load misses only'),                  # X60
        ('generic:LL_READ_MISS', 'reads only')],     # where L2 is the last level
    'L2D_CACHE_WB': [
        ('l2d_cache_wb', ''),                                  # Arm 0x18
        ('l2_lines_out.non_silent', '')],                      # Intel
    'L2D_MISS_OCCUPANCY': [
        ('offcore_requests_outstanding.all_data_rd', 'data reads only')],
    'L3D_CACHE_REFILL': [
        ('l3d_cache_refill', ''),                              # Arm 0x2a
        ('ll_cache_miss_rd', 'reads only'),                    # Arm 0x37
        ('l2_fill_rsp_src.dram_io_near', 'lines from local DRAM'),  # AMD Zen 5
        ('generic:LL_READ_MISS', 'reads only')],
    # not predicted by the model
    'STALL_BACKEND': [
        ('stall_backend', ''),                                 # Arm 0x24
        ('stalled_cycle_backend', ''),                         # X60
        ('generic:STALLED_CYCLES_BACKEND', ''),
        ('de_no_dispatch_per_slot.backend_stalls',
         'dispatch slots, up to 8 per cycle')],                # AMD Zen 4/5
    'STALL_BACKEND_MEM': [
        ('stall_backend_mem', ''),                             # Arm 0x4005
        ('ex_no_retire.load_not_complete',
         'cycles in which retirement waits on a load')],       # AMD Zen
    'LOAD_QUEUE_STALL': [
        ('de_dispatch_stall_cycle_dynamic_tokens_part1.load_queue_rsrc_stall',
         'cycles dispatch waits for load-queue tokens'),       # AMD Zen 5
        ('eu_lsu_load_full', 'LSU load queue full; whether events or cycles is '
                             'not documented')],               # SpacemiT X60
    'L1D_SW_PREFETCH_REFILL': [('ls_sw_pf_dc_fills.all', '')],  # AMD Zen
    # who brought the lines in -- to attribute refills the model does not
    # predict, and C lines that were in L2 before the kernel asked
    'L1D_DEMAND_REFILL': [('ls_dmnd_fills_from_sys.all', '')],       # AMD Zen
    'L1D_HW_PREFETCH_REFILL': [('ls_hw_pf_dc_fills.all', ''),        # AMD Zen
                               ('l1d_prefetch_refill', '')],         # SpacemiT X60
    'L2_PF_DRAM_L2HW': [('l2_pf_miss_l2_l3.l2_hwpf',
                         'L2 prefetcher requests served by DRAM')],  # AMD Zen
    'L2_PF_DRAM_L1HW': [('l2_pf_miss_l2_l3.l1_dc_hwpf',
                         'L1 prefetcher requests served by DRAM')],  # AMD Zen
    'SW_PREFETCH_ISSUED': [('ls_pref_instr_disp.all',
                            'software prefetches dispatched')],      # AMD Zen
    'SW_PREFETCH_IN_FLIGHT': [('ls_inef_sw_pref.mab_mch_cnt',
                               'software prefetches whose line was already '
                               'in flight')],                        # AMD Zen
}

# Kernel generic events: the spec goto_bench takes, and perf's name for the
# event, which perf list shows only where the kernel supports it.
GENERIC = {
    'CYCLES':                 ('CYCLES', 'cpu-cycles'),
    'INSTRUCTIONS':           ('INSTRUCTIONS', 'instructions'),
    'L1D_READ_ACCESS':        ('L1D_READ_ACCESS', 'l1-dcache-loads'),
    'L1D_READ_MISS':          ('L1D_READ_MISS', 'l1-dcache-load-misses'),
    'L1D_WRITE_ACCESS':       ('L1D_WRITE_ACCESS', 'l1-dcache-stores'),
    'L1D_WRITE_MISS':         ('L1D_WRITE_MISS', 'l1-dcache-store-misses'),
    'LL_READ_ACCESS':         ('LL_READ_ACCESS', 'llc-loads'),
    'LL_READ_MISS':           ('LL_READ_MISS', 'llc-load-misses'),
    'STALLED_CYCLES_BACKEND': ('PMU:0:0x8', 'stalled-cycles-backend'),
}

# Arm's architected numbers, for when sysfs is unreadable but the CPU is Arm.
ARM_COMMON = {
    'l1d_cache_refill': 0x03, 'l1d_cache': 0x04, 'inst_retired': 0x08,
    'cpu_cycles': 0x11, 'l1d_cache_wb': 0x15, 'l2d_cache': 0x16,
    'l2d_cache_refill': 0x17, 'l2d_cache_wb': 0x18, 'stall_backend': 0x24,
    'l3d_cache_refill': 0x2a, 'll_cache_miss_rd': 0x37, 'l1d_cache_rd': 0x40,
    'l1d_cache_wr': 0x41, 'l2d_cache_rd': 0x50, 'l2d_cache_wr': 0x51,
    'stall_backend_mem': 0x4005,
}


def parse_format(text):
    """'config:0-7,32-35' -> ('config', [(0, 7), (32, 35)])"""
    field, bits = text.strip().split(':', 1)
    ranges = []
    for part in bits.split(','):
        lo, _, hi = part.partition('-')
        ranges.append((int(lo), int(hi or lo)))
    return field, ranges


def place(value, ranges):
    """Scatter value's low bits across the ranges, lowest range first."""
    out = 0
    for lo, hi in ranges:
        width = hi - lo + 1
        out |= (value & ((1 << width) - 1)) << lo
        value >>= width
    return out


# perf's own event terms: they configure the perf event, not the PMU's bits
PERF_TERMS = {'period', 'freq', 'name', 'metric-id', 'percore', 'call-graph',
              'stack-size', 'inherit', 'no-inherit', 'overwrite', 'no-overwrite'}


def encode(terms_text, formats):
    """
    'event=0x44,umask=0xff' with a PMU's formats -> {'config': ..}, or None
    unless the terms name one concrete event: not for a parameter ('?'), a
    range ('0..0xfff', as perf list shows a PMU's term syntax), a value too
    wide for its field, or a term the PMU does not define.
    """
    regs = {}
    for term in filter(None, (t.strip() for t in terms_text.split(','))):
        name, eq, val = term.partition('=')
        if name in PERF_TERMS:
            continue
        try:
            v = int(val, 0) if eq else 1
        except ValueError:
            return None
        if v < 0:
            return None
        if name in ('config', 'config1', 'config2'):
            regs[name] = regs.get(name, 0) | v
            continue
        if name not in formats:
            return None
        field, ranges = formats[name]
        if v >> sum(hi - lo + 1 for lo, hi in ranges):
            return None
        regs[field] = regs.get(field, 0) | place(v, ranges)
    return regs


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def sysfs_pmus(root):
    """{pmu: {'type': int, 'formats': {...}, 'events': {name: terms}}}"""
    pmus = {}
    for dev in sorted(glob.glob(os.path.join(root, 'bus/event_source/devices/*'))):
        typ = read(os.path.join(dev, 'type'))
        if typ is None:
            continue
        formats = {}
        for f in glob.glob(os.path.join(dev, 'format/*')):
            try:
                formats[os.path.basename(f)] = parse_format(read(f))
            except (ValueError, AttributeError):
                pass
        events = {}
        for e in glob.glob(os.path.join(dev, 'events/*')):
            name = os.path.basename(e)
            if '.' in name and name.rsplit('.', 1)[1] in ('scale', 'unit'):
                continue
            events[name] = read(e) or ''
        pmus[os.path.basename(dev)] = {'type': int(typ), 'formats': formats,
                                       'events': events}
    return pmus


def parse_perf_json(text):
    """
    The entries of `perf list -j` output.  perf can print its own errors into
    the same stream ('Error: failed to open tracing events directory' lands
    between the last entry and the closing bracket), so when the whole does
    not parse, every flat {...} object is parsed on its own.
    """
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    out = []
    for m in re.finditer(r'\{[^{}]*\}', text, re.S):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return out


def perf_list_events(json_file=None):
    """
    ([(name, 'pmu/terms/')], {every event name perf lists, lowercased},
    status) from `perf list -j`.  The set is None when perf is not there, or
    when it lists no legacy events at all (an output this script does not
    understand): then nothing can be ruled out.  status is 'ok', 'missing'
    (no perf), or 'old' (perf without -j, which came in perf 6.x).
    """
    if json_file:
        with open(json_file) as f:
            data = parse_perf_json(f.read())
    else:
        perf = shutil.which('perf')
        if not perf:
            return [], None, 'missing'
        try:
            out = subprocess.run([perf, 'list', '-j'], capture_output=True,
                                 text=True, timeout=300).stdout
        except (OSError, subprocess.SubprocessError):
            return [], None, 'old'
        data = parse_perf_json(out)
    if not data:
        return [], None, 'old'
    found, listed = [], set()
    for ev in data:
        name, enc = ev.get('EventName'), ev.get('Encoding')
        if name and enc and '/' in enc:
            found.append((name, enc))
        for field in ('EventName', 'EventAlias'):
            for alias in (ev.get(field) or '').split(' OR '):
                alias = alias.strip().lower()
                if '/' in alias:                       # cpu/L1-dcache-loads/
                    alias = alias.strip('/').rpartition('/')[2]
                if alias:
                    listed.add(alias)
    if not listed & {'cpu-cycles', 'cycles', 'instructions'}:
        listed = None
    return found, listed, 'ok'


# -- whether the kernel opens an event: perf_event_open(2), called directly,
# -- so that no perf binary is needed and the answer is the harness's own

class _perf_attr(ctypes.Structure):
    """struct perf_event_attr up to config2 (PERF_ATTR_SIZE_VER1)"""
    _fields_ = [('type', ctypes.c_uint32), ('size', ctypes.c_uint32),
                ('config', ctypes.c_uint64), ('sample_period', ctypes.c_uint64),
                ('sample_type', ctypes.c_uint64), ('read_format', ctypes.c_uint64),
                ('flags', ctypes.c_uint64), ('wakeup', ctypes.c_uint32),
                ('bp_type', ctypes.c_uint32), ('config1', ctypes.c_uint64),
                ('config2', ctypes.c_uint64)]


_NR_PERF_EVENT_OPEN = {'x86_64': 298, 'aarch64': 241, 'riscv64': 241,
                       'ppc64le': 319, 'ppc64': 319, 's390x': 331}
_CACHE = {'L1D': 0, 'L1I': 1, 'LL': 2, 'DTLB': 3, 'ITLB': 4, 'BPU': 5, 'NODE': 6}
_OP = {'READ': 0, 'WRITE': 1, 'PREFETCH': 2}
_RESULT = {'ACCESS': 0, 'MISS': 1}


def attr_of(spec):
    """goto_bench's event spec -> (perf type, config), None if not known"""
    if spec.startswith('PMU:'):
        _, t, c = spec.split(':')
        return int(t), int(c, 0)
    if spec.startswith('RAW:'):
        return 4, int(spec[4:], 0)
    if spec in ('CYCLES', 'INSTRUCTIONS'):
        return 0, 0 if spec == 'CYCLES' else 1
    parts = spec.split('_')
    if len(parts) == 3 and parts[0] in _CACHE and parts[1] in _OP and parts[2] in _RESULT:
        return 3, _CACHE[parts[0]] | _OP[parts[1]] << 8 | _RESULT[parts[2]] << 16
    return None


def _touch_memory():
    """a few milliseconds of loads, stores and misses to count"""
    s = 0
    for i in range(100000):
        s += i
    buf = bytearray(16 << 20)
    bytes(memoryview(buf)[::64])                 # a load per line
    buf[::64] = bytes(len(buf) // 64)            # a store per line
    return s


def try_count(etype, config):
    """(count over _touch_memory(), None), or (None, why the kernel refused)"""
    nr = _NR_PERF_EVENT_OPEN.get(platform.machine())
    if nr is None or not sys.platform.startswith('linux'):
        return None, 'perf_event_open not known here'
    attr = _perf_attr(type=etype, size=ctypes.sizeof(_perf_attr), config=config,
                      flags=1 | 1 << 5 | 1 << 6)   # disabled, exclude kernel, hv
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(nr, ctypes.byref(attr), 0, -1, -1, 0)
    if fd < 0:
        return None, os.strerror(ctypes.get_errno())
    try:
        fcntl.ioctl(fd, 0x2403, 0)                 # PERF_EVENT_IOC_RESET
        fcntl.ioctl(fd, 0x2400, 0)                 # PERF_EVENT_IOC_ENABLE
        _touch_memory()
        fcntl.ioctl(fd, 0x2401, 0)                 # PERF_EVENT_IOC_DISABLE
        return int.from_bytes(os.read(fd, 8), sys.byteorder), None
    finally:
        os.close(fd)


def make_check():
    """
    (check, None): check(spec, need_count) is None for an event this process
    can open -- and that counts something, if need_count -- else why not.
    (None, why) if not even the cycle counter opens: then nothing can be
    checked, and every pick stays unverified.
    """
    _, why = try_count(0, 0)
    if why:
        return None, why

    def check(spec, need_count):
        at = attr_of(spec)
        if at is None:
            return None
        n, why = try_count(*at)
        if why:
            return f'cannot be opened: {why}'
        if need_count and not n:
            return 'opens, but counted nothing'
        return None
    return check, None


def arm_core_type(pmus):
    """the type of the Arm core PMU, if there is one"""
    for name, info in pmus.items():
        if re.match(r'arm(v8|v9)?_|armv[89]', name) and 'cpu_cycles' in info['events']:
            return info['type']
    return None


def spec_for(pmu, info, regs):
    """The goto_bench spec for an encoded event, or None if not expressible."""
    if set(regs) - {'config'}:
        return None                      # needs config1/config2
    return f"PMU:{info['type']}:{regs.get('config', 0):#x}"


def core_pmu(pmus):
    """
    The core PMU's sysfs name.  perf's event tables call it default_core,
    whatever sysfs calls it: cpu on x86 and RISC-V, armv8_pmuv3_N on Arm.
    """
    if 'cpu' in pmus:
        return 'cpu'
    for name, info in pmus.items():
        if re.match(r'arm(v8|v9)?_|armv[89]|cpu_core', name) and \
                ('cpu_cycles' in info['events'] or name == 'cpu_core'):
            return name
    return None


def candidates(pmus, perf_events):
    """
    ({lowercased event name: (spec, where)} over every source, number of
    entries skipped as not one concrete event)
    """
    table, skipped = {}, 0
    for pmu, info in pmus.items():
        for name, terms in info['events'].items():
            regs = encode(terms, info['formats'])
            spec = regs is not None and spec_for(pmu, info, regs)
            if spec:
                table.setdefault(name.lower(), (spec, f"sysfs {pmu}/{name}"))
            else:
                skipped += 1
    core = core_pmu(pmus)
    for name, enc in perf_events:
        pmu, _, rest = enc.partition('/')
        if pmu == 'default_core' and core:
            pmu = core
        terms = rest.rpartition('/')[0] if '/' in rest else rest   # drop /u, /k ...
        info = pmus.get(pmu)
        if not info:
            continue
        regs = encode(terms, info['formats'])
        spec = regs is not None and spec_for(pmu, info, regs)
        if spec:
            table.setdefault(name.lower(), (spec, f"perf list {name} = {enc}"))
        else:
            skipped += 1
    return table, skipped


def is_arm(root):
    return os.path.isdir(os.path.join(root, 'bus/event_source/devices')) and any(
        p.startswith('armv8') or p.startswith('arm_')
        for p in os.listdir(os.path.join(root, 'bus/event_source/devices')))


def last_level(root):
    """The highest data or unified cache level of cpu0, from sysfs; None if
    sysfs does not say."""
    levels = []
    for d in glob.glob(os.path.join(root, 'devices/system/cpu/cpu0/cache/index*')):
        if read(os.path.join(d, 'type')) != 'Instruction':
            lv = read(os.path.join(d, 'level'))
            if lv and lv.isdigit():
                levels.append(int(lv))
    return max(levels) if levels else None


def choose(table, arm_fallback, listed, llc=None, check=None, arm_type=None):
    """
    ({canonical: (spec, where, note)}, None where nothing was found;
    {canonical: [candidates rejected, and why]}).  A generic event is taken
    only if perf lists it (listed), or if there is no perf list to tell; a
    generic last-level-cache event only for the level that is last here
    (llc; L3 if unknown).  With check, every candidate is opened first, and
    one the kernel refuses is passed over.  With arm_type, the architected
    numbers of events sysfs does not list (Arm lists only what its PMCEID
    registers advertise, never 0x40 and up: L1D_CACHE_RD ...) are tried on
    the core PMU, and kept only if they count -- only in 0x40..0xbf, the
    recommended implementation-defined range: below it, and at 0x4000 and up,
    sysfs lists what PMCEID advertises, so an absent event is not there (the
    kernel would still open it raw, to count nothing).
    """
    chosen, rejected = {}, {}
    for canon, cands in CANONICAL.items():
        pick = None
        for name, note in cands:
            cand, need = None, False
            if name.startswith('generic:'):
                spec, perf_name = GENERIC[name.split(':', 1)[1]]
                if listed is not None and perf_name not in listed:
                    continue                             # the kernel lacks it
                if spec.startswith('LL_') and not canon.startswith(f'L{llc or 3}D'):
                    continue                             # not this machine's LLC
                cand = (spec, "kernel's generic event" + (
                    '' if listed is not None else ', opened here' if check
                    else ', unverified (no perf list)'), note)
            elif name in table:
                cand = (*table[name], note)
            elif (arm_type is not None and check and name in ARM_COMMON
                  and 0x40 <= ARM_COMMON[name] <= 0xbf):
                cand = (f"PMU:{arm_type}:{ARM_COMMON[name]:#x}",
                        f"Arm architected number {ARM_COMMON[name]:#x}, counted here",
                        note)
                need = True
            elif arm_fallback and name in ARM_COMMON:
                cand = (f"RAW:{ARM_COMMON[name]:#x}",
                        "Arm architected number (sysfs absent)", note)
            if cand is None:
                continue
            why = check(cand[0], need) if check else None
            if why:
                rejected.setdefault(canon, []).append(f"{cand[0]}: {why}")
                continue
            pick = cand
            break
        chosen[canon] = pick
    return chosen, rejected


def cpuinfo(path):
    """The first value of each /proc/cpuinfo key."""
    info = {}
    for line in (read(path) or '').splitlines():
        k, _, v = line.partition(':')
        info.setdefault(k.strip(), v.strip())
    return info


def hints(pmus, perf_events, cpu, perf_status='ok'):
    """Why events may be missing on this machine."""
    out = []
    if (cpu.get('vendor_id') == 'AuthenticAMD' or 'ibs_op' in pmus) \
            and 'amd_l3' not in pmus:
        out.append("no amd_l3 PMU: the amd-uncore module is not loaded "
                   "(modprobe amd-uncore). perf's L3 latency metrics "
                   "(l3_read_miss_latency) need it, and need system-wide "
                   "counting besides, which goto_bench does not do; "
                   "L1D_MISS_OCCUPANCY / L1D_CACHE_REFILL measures the mean "
                   "L1 miss latency per thread instead")
    if 'mvendorid' in cpu or os.uname().machine.startswith('riscv'):
        vendor = [n for n, enc in perf_events
                  if enc.partition('/')[0] in ('cpu', 'default_core')]
        if not vendor and perf_status != 'ok':
            out.append("perf's event list could not be read here: its named "
                       "events are unknown (save `perf list -j` from a perf that "
                       "works and pass it with --perf-json)")
        elif not vendor:
            ids = '-'.join(cpu.get(k, '?') for k in ('mvendorid', 'marchid', 'mimpid'))
            out.append(f"perf has no event tables for this core ({ids}, "
                       "mvendorid-marchid-mimpid): its mapfile lacks the id, so "
                       "only generic events have names. The PMU itself works; "
                       "raw events (RAW:<code>) need the core's documentation, "
                       "and firmware (OpenSBI) that maps them to counters")
    return out


def main(argv=None):
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)    # quiet under | head
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--write', metavar='STEM',
                   help='write STEM.txt (events for goto_bench) and STEM.json')
    p.add_argument('--list', action='store_true',
                   help='list every cache, memory and stall event found')
    p.add_argument('--sysfs-root', default='/sys',
                   help='where sysfs is mounted (for testing)')
    p.add_argument('--no-perf', action='store_true', help='do not run perf list')
    p.add_argument('--perf-json', metavar='FILE',
                   help='read `perf list -j` output saved in FILE instead of running perf')
    p.add_argument('--include', metavar='REGEX',
                   help='also write every event whose name matches REGEX, under its '
                        'own name: measured, not modelled -- for exploring a core '
                        "(on K1: 'l2_a[rw]_channel|lsu|vlsu|l1d_(prefetch|excl|amr)')")
    p.add_argument('--no-verify', action='store_true',
                   help='do not open the picked events to check the kernel takes them')
    p.add_argument('--cpuinfo', default='/proc/cpuinfo', help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    pmus = sysfs_pmus(a.sysfs_root)
    perf_events, listed, perf_status = (([], None, 'skipped') if a.no_perf
                                        else perf_list_events(a.perf_json))
    table, skipped = candidates(pmus, perf_events)
    listed_pmus = ', '.join('%s (type %d)' % (k, v['type']) for k, v in pmus.items())
    print(f"# PMUs: {listed_pmus or 'none'}")
    print(f"# events: {len(table)} encodable"
          + (f", {len(perf_events)} from perf list" if perf_events else
             ", perf lists no named events here" if perf_status == 'ok' else
             ", perf's list could not be read (no 'perf list -j' before 6.x, "
             "or output that is not JSON): sysfs only"
             if perf_status == 'old' else
             ", perf not available" if perf_status == 'missing' else "")
          + (f"; {skipped} skipped as not one concrete event (term ranges, '?' "
             f"parameters, or values or terms the PMU cannot take)" if skipped else ""))
    for h in hints(pmus, perf_events, cpuinfo(a.cpuinfo), perf_status):
        print(f"# note: {h}")

    if a.list:
        pat = re.compile(r'cache|refill|miss|wb|writeback|fill|stall|mem|l1d|l2|'
                         r'l3|llc|lat|occup|mab|pend|outstanding|dispatch|retire',
                         re.I)
        for name in sorted(table):
            if pat.search(name):
                print(f"  {name:<44} {table[name][0]:<24} {table[name][1]}")
        return 0

    arm_fallback = not table and os.uname().machine == 'aarch64'
    check, why = (None, 'not asked to') if a.no_verify else make_check()
    if check is None and not a.no_verify:
        print(f"# note: this process cannot open the cycle counter ({why}): "
              f"no event could be checked, so the picks are unverified")
    chosen, rejected = choose(table, arm_fallback or (is_arm(a.sysfs_root) and not table),
                              listed, last_level(a.sysfs_root), check,
                              arm_core_type(pmus))
    for canon, pick in chosen.items():
        if pick:
            spec, where, note = pick
            print(f"  {canon:<22} {spec:<20} {where}" + (f"  [{note}]" if note else ""))
        else:
            print(f"  {canon:<22} not found")
        for r in rejected.get(canon, []):
            print(f"  {'':<22}   passed over {r}")
    if a.include:
        # exploration: every matching event under its own name, measured only
        pat, have, extra = re.compile(a.include, re.I), {p[0] for p in chosen.values() if p}, {}
        for name in sorted(table):
            spec, where = table[name]
            if not pat.search(name) or spec in have or name.upper() in chosen:
                continue
            why = check(spec, False) if check else None
            if why:
                print(f"  {name.upper():<22} passed over {spec}: {why}")
                continue
            extra[name.upper()] = (spec, where, '')
            have.add(spec)
        print(f"# included by --include: {len(extra)}, measured only")
        for canon, (spec, where, _) in extra.items():
            print(f"  {canon:<30} {spec:<14} {where}")
        chosen.update(extra)
    if a.write:
        with open(a.write + '.txt', 'w') as f:
            for canon, pick in chosen.items():
                if pick:
                    spec, where, note = pick
                    f.write(f"{spec:<24} # {canon}: {where}"
                            + (f" [{note}]" if note else "") + "\n")
        with open(a.write + '.json', 'w') as f:
            json.dump({c: p[0] for c, p in chosen.items() if p}, f, indent=2)
        print(f"# wrote {a.write}.txt and {a.write}.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
