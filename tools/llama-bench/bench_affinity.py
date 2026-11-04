#!/usr/bin/env python3
"""Select a benchmark main CPU without changing the launcher's affinity."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ENV_CPU = 'LLAMA_BENCH_MAIN_CPU'
CPU_LIMIT = 1024


def cpu_list(value):
    result = set()
    for item in value.strip().split(','):
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', item):
            raise ValueError('invalid CPU list: ' + value)
        bounds = [int(x) for x in item.split('-')]
        first, last = bounds[0], bounds[-1]
        if first > last or last >= 1048576:
            raise ValueError('CPU list outside supported range')
        result.update(range(first, last + 1))
    return result


def read(path):
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return None
        with path.open() as stream:
            value = stream.read(4 * 1024 * 1024 + 1)
        return value.strip() if len(value) <= 4 * 1024 * 1024 else None
    except (OSError, UnicodeError):
        return None


def snapshot(sys_root=Path('/sys'), proc_root=Path('/proc'), allowed=None):
    problems = []
    if allowed is None:
        try:
            allowed = os.sched_getaffinity(0)
        except (AttributeError, OSError) as error:
            allowed = set()
            problems.append('allowed mask unavailable: ' + str(error))
    allowed = set(allowed)
    root = sys_root / 'devices/system/cpu'
    try:
        online = cpu_list(read(root / 'online') or '')
    except ValueError:
        online = set()
        problems.append('online CPU list unavailable')
    if online and max(online) >= CPU_LIMIT:
        problems.append('CPU mask exceeds the benchmark helper limit')
    topology = {}
    for cpu in sorted(online):
        directory = root / ('cpu' + str(cpu)) / 'topology'
        try:
            package = int(read(directory / 'physical_package_id') or '-1')
            core = int(read(directory / 'core_id') or '-1')
            siblings = cpu_list(read(directory / 'thread_siblings_list') or '')
            if package < 0 or core < 0 or cpu not in siblings:
                raise ValueError('incomplete core identity')
            topology[str(cpu)] = dict(package=package, core=core, siblings=sorted(siblings))
        except ValueError:
            problems.append('topology unavailable for CPU ' + str(cpu))
    irqs = []
    interrupts = read(proc_root / 'interrupts')
    if interrupts is None:
        problems.append('/proc/interrupts unavailable')
    else:
        for line in interrupts.splitlines():
            match = re.match(r'\s*(\d+):', line)
            if not match or not re.search(r'kl\d+_dev\d+|kunlun|\bxpu\b', line, re.I):
                continue
            irq = int(match[1])
            directory = proc_root / 'irq' / str(irq)
            effective = read(directory / 'effective_affinity_list')
            try:
                cpus = cpu_list(effective or '')
            except ValueError:
                cpus = set()
                problems.append('effective affinity unavailable for IRQ ' + str(irq))
            irqs.append(dict(irq=irq, line=line.strip(), effective=sorted(cpus),
                             requested=read(directory / 'smp_affinity_list')))
    if not irqs:
        problems.append('no XPU IRQ identified')
    return dict(allowed=sorted(allowed), online=sorted(online), topology=topology,
                xpu_irqs=irqs, problems=problems)


def select(policy, state):
    policy = str(policy).strip()
    output = dict(state, policy=policy, selected_cpu=None, reason='disabled', warnings=[])
    allowed = set(state['allowed']) & set(state['online'])
    irq_cpus = {cpu for irq in state['xpu_irqs'] for cpu in irq['effective']}
    topology = state['topology']
    irq_cores = set()
    blocked = set(irq_cpus)
    for cpu in irq_cpus:
        item = topology.get(str(cpu))
        if item:
            irq_cores.add((item['package'], item['core']))
            blocked.update(item['siblings'])
    for cpu, item in topology.items():
        if (item['package'], item['core']) in irq_cores:
            blocked.add(int(cpu))
    output['excluded_irq_core_cpus'] = sorted(blocked)
    if policy in ('off', 'none', ''):
        return output
    if policy != 'auto':
        if not re.fullmatch(r'[0-9]+', policy):
            raise ValueError('main CPU must be off, auto, or an unsigned CPU number')
        cpu = int(policy)
        if cpu >= CPU_LIMIT or cpu not in set(state['allowed']):
            raise ValueError('explicit main CPU is outside the allowed/supported mask')
        output.update(selected_cpu=cpu, reason='explicit')
        if state['problems']:
            output['warnings'].append('incomplete topology/IRQ data; explicit selection retained')
        if cpu in blocked:
            output['warnings'].append('explicit CPU shares an XPU IRQ physical core')
        return output
    if state['problems'] or any(str(cpu) not in topology for cpu in allowed | irq_cpus):
        output['reason'] = 'auto unavailable: ' + '; '.join(state['problems'] or ['incomplete core map'])
        return output
    # One allowed logical CPU per physical core, excluding IRQ cores and siblings.
    cores = {}
    for cpu in sorted(allowed - blocked):
        if cpu >= CPU_LIMIT:
            continue
        item = topology[str(cpu)]
        key = (item['package'], item['core'])
        cores.setdefault(key, cpu)
    output['safe_candidates'] = sorted(cores.values())
    if not cores:
        output['reason'] = 'auto unavailable: no allowed physical core outside XPU IRQ cores'
        return output
    output.update(selected_cpu=min(cores.values()), reason='auto: lowest allowed representative of a non-IRQ physical core')
    return output


def prepare(policy, environment=None):
    state = select(policy, snapshot())
    env = dict(os.environ if environment is None else environment)
    env[ENV_CPU] = str(state['selected_cpu']) if state['selected_cpu'] is not None else 'off'
    return env, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-cpu', default='off')
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        env, state = prepare(args.main_cpu)
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(state, sort_keys=True)
    print('BENCH_AFFINITY_POLICY ' + text, file=sys.stderr, flush=True)
    if args.json_output:
        with args.json_output.open('x') as stream:
            stream.write(json.dumps(state, indent=2) + '\n')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.inspect:
        return
    if not command:
        parser.error('benchmark command required (or use --inspect)')
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
