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
import glob
import json
import os
import re
import signal
import shutil
import subprocess
import sys

# The model's events (model.py counters()), and what to look for.  Each
# candidate is a perf/sysfs event name (case-insensitive); the first one found
# wins.  'generic:' candidates are the counter backend's built-in names.
CANONICAL = {
    'CPU_CYCLES':       ['cpu_cycles', 'cycles', 'generic:CYCLES'],
    'INST_RETIRED':     ['inst_retired', 'instructions', 'generic:INSTRUCTIONS'],
    'L1D_CACHE':        ['l1d_cache',                         # Arm 0x04
                         'l1d.all_ref',
                         'generic:L1D_READ_ACCESS'],
    'L1D_CACHE_REFILL': ['l1d_cache_refill',                  # Arm 0x03
                         'l1d.replacement',                   # Intel
                         'ls_any_fills_from_sys.all',         # AMD Zen 4/5
                         'ls_refills_from_sys.all',           # AMD Zen 2/3
                         'generic:L1D_READ_MISS'],
    'L1D_CACHE_WB':     ['l1d_cache_wb'],                     # Arm 0x15
    'L2D_CACHE':        ['l2d_cache',                         # Arm 0x16
                         'l2_rqsts.references',               # Intel
                         'l2_request_g1.all_no_prefetch'],    # AMD
    'L2D_CACHE_REFILL': ['l2d_cache_refill',                  # Arm 0x17
                         'l2_lines_in.all',                   # Intel
                         'l2_cache_req_stat.ic_dc_miss_in_l2'],  # AMD
    'L2D_CACHE_WB':     ['l2d_cache_wb',                      # Arm 0x18
                         'l2_lines_out.non_silent'],          # Intel
    'L3D_CACHE_REFILL': ['l3d_cache_refill',                  # Arm 0x2a
                         'll_cache_miss_rd',                  # Arm 0x37
                         'generic:LL_READ_MISS'],
    'STALL_BACKEND':    ['stall_backend'],                    # Arm 0x24
}

# Arm's architected numbers, for when sysfs is unreadable but the CPU is Arm.
ARM_COMMON = {
    'l1d_cache_refill': 0x03, 'l1d_cache': 0x04, 'inst_retired': 0x08,
    'cpu_cycles': 0x11, 'l1d_cache_wb': 0x15, 'l2d_cache': 0x16,
    'l2d_cache_refill': 0x17, 'l2d_cache_wb': 0x18, 'stall_backend': 0x24,
    'l3d_cache_refill': 0x2a, 'll_cache_miss_rd': 0x37,
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


def perf_list_events():
    """[(name, 'pmu/terms/')] from `perf list -j`, when perf is installed."""
    perf = shutil.which('perf')
    if not perf:
        return []
    try:
        out = subprocess.run([perf, 'list', '-j'], capture_output=True,
                             text=True, timeout=60).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    found = []
    for ev in data:
        name, enc = ev.get('EventName'), ev.get('Encoding')
        if name and enc and '/' in enc:
            found.append((name, enc))
    return found


def spec_for(pmu, info, regs):
    """The goto_bench spec for an encoded event, or None if not expressible."""
    if set(regs) - {'config'}:
        return None                      # needs config1/config2
    return f"PMU:{info['type']}:{regs.get('config', 0):#x}"


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
    for name, enc in perf_events:
        pmu, _, rest = enc.partition('/')
        terms = rest.rpartition('/')[0] if '/' in rest else rest   # drop /u, /k ...
        info = pmus.get(pmu)
        if not info:
            continue
        regs = encode(terms, info['formats'])
        spec = regs is not None and spec_for(pmu, info, regs)
        if spec:
            table.setdefault(name.lower(), (spec, f"perf list {enc}"))
        else:
            skipped += 1
    return table, skipped


def is_arm(root):
    return os.path.isdir(os.path.join(root, 'bus/event_source/devices')) and any(
        p.startswith('armv8') or p.startswith('arm_')
        for p in os.listdir(os.path.join(root, 'bus/event_source/devices')))


def choose(table, arm_fallback):
    """{canonical: (spec, where)}, None where nothing was found."""
    chosen = {}
    for canon, cands in CANONICAL.items():
        pick = None
        for c in cands:
            if c.startswith('generic:'):
                pick = (c.split(':', 1)[1], "kernel's generic event")
                break
            if c in table:
                pick = table[c]
                break
            if arm_fallback and c in ARM_COMMON:
                pick = (f"RAW:{ARM_COMMON[c]:#x}", "Arm architected number (sysfs absent)")
                break
        chosen[canon] = pick
    return chosen


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
    a = p.parse_args(argv)

    pmus = sysfs_pmus(a.sysfs_root)
    perf_events = [] if a.no_perf else perf_list_events()
    table, skipped = candidates(pmus, perf_events)
    listed = ', '.join('%s (type %d)' % (k, v['type']) for k, v in pmus.items())
    print(f"# PMUs: {listed or 'none'}")
    print(f"# events: {len(table)} encodable"
          + (f", {len(perf_events)} from perf list" if perf_events else
             ", perf not available" if not a.no_perf else "")
          + (f"; {skipped} skipped as not one concrete event (term ranges, '?' "
             f"parameters, or values or terms the PMU cannot take)" if skipped else ""))

    if a.list:
        pat = re.compile(r'cache|refill|miss|wb|writeback|fill|stall|mem|l1d|l2|l3|llc', re.I)
        for name in sorted(table):
            if pat.search(name):
                print(f"  {name:<44} {table[name][0]:<24} {table[name][1]}")
        return 0

    arm_fallback = not table and os.uname().machine == 'aarch64'
    chosen = choose(table, arm_fallback or (is_arm(a.sysfs_root) and not table))
    for canon, pick in chosen.items():
        print(f"  {canon:<18} " + (f"{pick[0]:<24} {pick[1]}" if pick else "not found"))
    if a.write:
        with open(a.write + '.txt', 'w') as f:
            for canon, pick in chosen.items():
                if pick:
                    f.write(f"{pick[0]:<24} # {canon}: {pick[1]}\n")
        with open(a.write + '.json', 'w') as f:
            json.dump({c: p[0] for c, p in chosen.items() if p}, f, indent=2)
        print(f"# wrote {a.write}.txt and {a.write}.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
