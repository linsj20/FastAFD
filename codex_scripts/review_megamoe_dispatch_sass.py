#!/usr/bin/env python3
"""Inspect ordinary MegaMoE dispatch register regions in a source-scoped JIT cache.

This checks the linear region between dispatch register deallocation and the next
role's register change. It does not prove control flow, synchronization correctness,
or the budgets of native senders that execute before dispatch deallocation.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def inspect(sass):
    changes = list(re.finditer(r"USETMAXREG\.[^;]+;", sass))
    starts = [i for i, match in enumerate(changes)
              if re.search(r"DEALLOC\.CTAPOOL 0x(?:30|60)\s*;", match[0])]
    if len(starts) != 1 or starts[0] + 1 >= len(changes):
        raise ValueError("Expected one dispatch deallocation followed by another role")
    i = starts[0]
    budget = int(re.search(r"0x([0-9a-f]+)", changes[i][0])[1], 16)
    region = sass[changes[i].end():changes[i + 1].start()]
    return {
        "scope": "linear dispatch region; excludes any pre-deallocation sender",
        "start": changes[i][0], "end": changes[i + 1][0], "budget": budget,
        **inspect_region(region),
    }


def inspect_register_regions(sass):
    """Inspect each deallocation span, including TMA/MMA roles; not a CFG proof."""
    changes = list(re.finditer(r"USETMAXREG\.[^;]+;", sass))
    regions = []
    for current, following in zip(changes, changes[1:]):
        match = re.search(r"DEALLOC\.CTAPOOL 0x([0-9a-f]+)\s*;", current[0])
        if match:
            regions.append({
                "scope": "linear register region; control-flow paths require separate review",
                "start": current[0], "end": following[0],
                "budget": int(match[1], 16),
                **inspect_region(sass[current.end():following.start()]),
            })
    return regions


def inspect_reachable_registers(sass):
    """Follow nvdisasm branch labels from each deallocation until exit/reallocation.

    Indirect branches require nvdisasm BRANCH_TARGETS annotations. Unresolved
    transfers are reported explicitly; this does not prove runtime predicates.
    """
    instructions, labels = [], {}
    for line in sass.splitlines():
        label = re.match(r"\s*(\.L[\w.]+):", line)
        if label:
            labels[label[1]] = len(instructions)
        if re.search(r"/\*[0-9a-f]+\*/", line):
            instructions.append(line)
    reports = []
    for entry, line in enumerate(instructions):
        allocation = re.search(r"USETMAXREG\.DEALLOC\.CTAPOOL 0x([0-9a-f]+)", line)
        if not allocation:
            continue
        pending, visited, unresolved = [entry + 1], set(), set()
        operations = {}
        while pending:
            index = pending.pop()
            if index in visited or index >= len(instructions):
                continue
            current = instructions[index]
            if "USETMAXREG." in current:
                continue
            visited.add(index)
            match = re.search(r"\*/\s+(@!?[A-Z]+\d*\s+)?([A-Z][\w.]*)\s*(.*)", current)
            if not match:
                unresolved.add(current.strip())
                continue
            predicate, operation, operands = match.groups()
            operations[index] = operation
            transfer = operation.startswith(("BRA", "BRX", "JMP", "JMX", "CALL", "BSSY"))
            terminal = operation.startswith(("EXIT", "RET"))
            if transfer:
                targets = re.findall(r"\.L[\w.]+", operands)
                if not targets:
                    unresolved.add(current.strip())
                for target in targets:
                    if target in labels:
                        pending.append(labels[target])
                    else:
                        unresolved.add(target)
            conditional = bool(predicate or re.match(r"!?UP(?:\d+|T)\s*,", operands))
            if (not transfer and not terminal) or conditional or operation.startswith(("BSSY", "CALL")):
                pending.append(index + 1)
        reports.append({
            "scope": "Reachable explicit instructions between register changes; unresolved transfers require review",
            "entry": line.strip(), "budget": int(allocation[1], 16),
            "instructions_checked": len(visited), "unresolved_transfers": sorted(unresolved),
            "operation_counts": dict(Counter(operations.values())),
            **inspect_region("\n".join(instructions[i] for i in sorted(visited))),
        })
    return reports


def inspect_region(region):
    registers = [int(n) for n in re.findall(r"\bR(\d+)\b", region)]
    wide = [int(n) + 1 for n in re.findall(r"\bR(\d+)\.64\b", region)]
    for line in region.splitlines():
        instruction = re.search(r"\*/\s+(?:@!?[A-Z]+\d*\s+)?([A-Z][\w.]+)\s+(.*);", line)
        if not instruction:
            continue
        op, operands = instruction.groups()
        width = 8 if ".256" in op else 4 if ".128" in op else 2 if ".64" in op else 1
        data = re.findall(r"\bR(\d+)\b", operands)
        if ".256" in op and op.startswith("LDG"):
            # SM100 disassembly names two four-register groups for LDG.256.
            wide.extend(int(n) + 3 for n in data[:2])
        elif ".256" in op and op.startswith("STG"):
            wide.extend(int(n) + 3 for n in data[-2:])
        elif data and op.startswith(("LDG", "LDS", "LDC")):
            wide.append(int(data[0]) + width - 1)
        elif data and op.startswith(("STG", "STS")):
            wide.append(int(data[-1]) + width - 1)
    return {
        "highest_explicit_register": max(registers, default=-1),
        "highest_register_including_wide_memory_operands": max(registers + wide, default=-1),
        "local_memory_instructions": re.findall(r"\b(?:LDL|STL)[^;]*;", region),
    }


def inspect_allocation_dominance(main_sass, callee):
    """Check that every explicit path to a named call crosses register allocation.

    Accept only the ordinary unconditional TRY_ALLOC/retry sequence. Conditional
    branches and reconvergence targets are both explored; unresolved transfers
    fail the proof. Calls are assumed to return to their following instruction.
    The caller supplies only the main function, excluding callee bodies.
    """
    instructions, labels = [], {}
    for line in main_sass.splitlines():
        label = re.match(r"\s*(\.L[\w.]+):", line)
        if label:
            labels[label[1]] = len(instructions)
        match = re.search(r"/\*[0-9a-f]+\*/\s*(@!?[A-Z]+\d*\s+)?([A-Z][\w.]*)\s*(.*);", line)
        if match:
            instructions.append(match.groups())
    allocations = [i for i, (_, op, _) in enumerate(instructions) if op.startswith('USETMAXREG.TRY_ALLOC')]
    assert len(allocations) == 1
    allocation = allocations[0]
    predicate, _, operands = instructions[allocation]
    assert not predicate
    flag, budget = re.fullmatch(r'(UP\d+), (0x[0-9a-f]+)\s*', operands).groups()
    retry_predicate, retry_op, retry_operands = instructions[allocation + 1]
    retry_targets = re.findall(r'\.L[\w.]+', retry_operands)
    assert not retry_predicate and retry_op == 'BRA.U' and retry_operands.startswith('!' + flag + ',')
    assert len(retry_targets) == 1 and labels[retry_targets[0]] == allocation - 1
    retry_head = instructions[allocation - 1]
    assert not retry_head[0] and retry_head[1] in ('WARPSYNC.ALL', 'NOP')
    calls = {i for i, (_, op, operands) in enumerate(instructions) if op.startswith('CALL') and callee in operands}
    assert calls
    pending, visited, unresolved = [0], set(), set()
    while pending:
        i = pending.pop()
        if i in visited or i >= len(instructions) or i == allocation:
            continue
        visited.add(i)
        predicate, op, operands = instructions[i]
        transfer = op.startswith(('BRA', 'BRX', 'JMP', 'JMX', 'BSSY'))
        terminal = op.startswith(('EXIT', 'RET'))
        if transfer:
            targets = re.findall(r'\.L[\w.]+', operands)
            if not targets:
                unresolved.add(str((op, operands)))
            for target in targets:
                if target in labels:
                    pending.append(labels[target])
                else:
                    unresolved.add(target)
        conditional = bool(predicate or re.match(r'!?UP(?:\d+|T)\s*,', operands))
        if (not transfer and not terminal) or conditional or op.startswith('BSSY'):
            pending.append(i + 1)
    return dict(budget=int(budget, 16), retry_head_operation=retry_head[1], call_count=len(calls),
                calls_reachable_without_allocation=sorted(calls & visited),
                unresolved_transfers=sorted(unresolved),
                dominates_calls=not unresolved and not calls.intersection(visited))


def preparation_instructions(sass, source=None):
    """Optionally canonicalize the kernel name and proven assertion line arguments."""
    metadata = []
    if source is not None:
        kernels = re.findall(r"^\s*\.type\s+(_ZN9deep_gemm27sm100_fp8_fp4_mega_moe_impl\S+),@function$", sass, re.M)
        if len(kernels) != 1:
            raise ValueError("Expected exactly one MegaMoE entry symbol")
        # Retain each callee's own mangled name and every branch label/operand.
        sass = sass.replace(kernels[0], "MEGAMOE_ENTRY")
        metadata.append(dict(kernel_symbol=kernels[0]))
    functions = re.findall(r"^\s*\.type\s+(\S+),@function$", sass, re.M)
    instructions = [(m[1], re.sub(r"\s+", " ", m[2]).strip())
        for line in sass.splitlines()
        if (m := re.match(r"\s*/\*([0-9a-f]+)\*/\s*(.*?;)", line))]
    if source is not None:
        source_lines = Path(source).read_text().splitlines()
        for i, (address, instruction) in enumerate(instructions):
            match = re.fullmatch(r"IMAD.MOV.U32 (R\d+), RZ, RZ, (0x[0-9a-f]+) ;", instruction)
            if not match:
                continue
            register, immediate = match.groups()
            line = int(immediate, 16)
            if not 1 <= line <= len(source_lines) or not source_lines[line - 1].strip().startswith("DG_DEVICE_ASSERT("):
                continue
            # The assertion line argument is stored on the stack before call/trap.
            # Its offset varies with the caller's diagnostic argument frame.
            following = [op for _, op in instructions[i + 1:i + 17]]
            stores = [op for op in following
                      if re.fullmatch(r"STL \[R1\+0x[0-9a-f]+\], " + register + r" ;", op)]
            if len(stores) != 1 or not any(op.startswith("CALL.ABS.NOINC ") for op in following):
                continue
            store = stores[0]
            trap = next((j for j, op in enumerate(following) if op == "BPT.TRAP 0x1 ;"), None)
            if trap is None or following.index(store) >= trap:
                continue
            if any(re.match(r"(?:@\S+ )?(?:BRA|JMP|RET|EXIT)", op) for op in following[:trap]):
                continue
            if any(re.match(r"\S+ " + register + r"[, ]", op) for op in following[:following.index(store)]):
                continue
            statement = source_lines[line - 1].strip()
            instructions[i] = (address, instruction.replace(immediate, "ASSERT_LINE:" + statement))
            metadata.append(dict(address=address, line=line, statement=statement))
    return functions, instructions, metadata


def normalize_validation_helper_symbols(before, after, instructions):
    """Align local helper names only; callers still compare every instruction.

    Same-ordinal pairing preserves function layout. Only entry-scoped validation
    helpers may change names, and only exact symbolic call operands are replaced.
    Bodies, addresses, registers, immediates and unknown symbols remain intact.
    """
    if (not before or len(before) != len(after) or len(set(before)) != len(before)
            or len(set(after)) != len(after) or "MEGAMOE_ENTRY" not in before):
        raise ValueError("Validation-helper normalization requires matching distinct function coverage")
    prefix = "$MEGAMOE_ENTRY$_ZN9deep_gemm20validate_window_rowsI"
    renames = {}
    for old, new in zip(before, after):
        if old != new:
            if not old.startswith(prefix) or not new.startswith(prefix):
                raise ValueError(f"Unrecognized local function rename: {old} -> {new}")
            renames[new] = old
    # Simultaneous substitution prevents a swapped pair from collapsing names.
    def replace(match):
        symbol = match[1]
        return "`(" + renames.get(symbol, symbol) + ")"
    normalized = [(address, re.sub(r"`\(([^)]+)\)", replace, op)
                   if re.fullmatch(r"(?:@\S+ )?CALL\.REL\.NOINC `\([^)]+\) ;", op) else op)
                  for address, op in instructions]
    return [renames.get(name, name) for name in after], normalized, renames


def diagnostic_syscall_blocks(sass, instructions):
    """Identify terminal printf paths for exact before/after comparison.

    Require nvdisasm's syscall metadata and vprintf as the sole external symbol.
    This preserves existing opaque diagnostics, not a proof of their internals.
    All caller instructions from the preceding transfer through the trap remain
    in the comparison, including call targets, frame offsets and argument setup.
    """
    externs = re.findall(r"EIATTR_EXTERNS\n(.*?)(?=//----- nvinfo|\Z)", sass, re.S)
    if not externs or any(re.findall(r"index@\(([^)]+)\)", block) != ["vprintf"]
                          for block in externs):
        return {}
    sections = re.findall(r"EIATTR_SYSCALL_OFFSETS\n(.*?)(?=//----- nvinfo|\Z)", sass, re.S)
    offsets = {int(value, 16) for block in sections
               for value in re.findall(r"\.word\s+0x([0-9a-f]+)", block)}
    blocks = {}
    boundary = re.compile(r"(?:@\S+ )?(?:BRA|BRX|JMP|JMX|EXIT|RET|USETMAXREG|BPT)\b")
    for i, (address, op) in enumerate(instructions):
        if int(address, 16) not in offsets or not re.fullmatch(r"CALL\.ABS\.NOINC R\d+ ;", op):
            continue
        start = i
        while start > 0 and i - start < 64 and not boundary.match(instructions[start - 1][1]):
            start -= 1
        if start == 0 or not boundary.match(instructions[start - 1][1]):
            continue
        stop = i + 1
        while stop < len(instructions) and stop - i <= 32:
            current = instructions[stop][1]
            if current == "BPT.TRAP 0x1 ;":
                blocks[address] = instructions[start - 1:stop + 1]
                break
            if boundary.match(current) or current.startswith("@"):
                break
            if "CALL" in current and (int(instructions[stop][0], 16) not in offsets
                                      or not re.fullmatch(r"CALL\.ABS\.NOINC R\d+ ;", current)):
                break
            stop += 1
    return blocks


def compare_resource_bounds(before, after, normalized_instructions=None, preserve_diagnostics=True):
    """Check role budgets and new spill counts; performance still needs a GPU comparison."""
    maximum = "highest_register_including_wide_memory_operands"
    reports = []
    ops = normalized_instructions or [preparation_instructions(s)[1] for s in (before, after)]
    diagnostics = [diagnostic_syscall_blocks(s, op) for s, op in zip((before, after), ops)]
    if preserve_diagnostics and diagnostics[0] != diagnostics[1]:
        raise ValueError("Changed external diagnostic call paths")
    if not preserve_diagnostics:
        # Scheduling changes move assertion blocks and their registers. Each
        # accepted call must still be a relocated vprintf in a straight-line
        # block ending in TRAP; retain equal instruction-kind/count coverage.
        shapes = [Counter(tuple(re.sub(r'^@!?\w+\s+', '', op).split()[0]
                                for _, op in block) for block in diagnostic.values())
                  for diagnostic in diagnostics]
        if shapes[0] != shapes[1]:
            raise ValueError("Changed terminal diagnostic instruction coverage")
    for sass, diagnostic in zip((before, after), diagnostics):
        roles = inspect_reachable_registers(sass)
        if not roles:
            raise ValueError("No role register regions")
        for role in roles:
            if role[maximum] >= role["budget"]:
                raise ValueError(f"Register budget exceeded: {role}")
            unresolved = [transfer for transfer in role["unresolved_transfers"]
                          if not ("CALL.REL.NOINC" in transfer and "validate_window_rows" in transfer)
                          and not ((m := re.match(r"/\*([0-9a-f]+)\*/", transfer))
                                   and m[1] in diagnostic)]
            if unresolved:
                raise ValueError(f"Unresolved role control flow: {role}")
        initial = inspect_region(sass[:sass.index("USETMAXREG.")])
        if initial[maximum] >= 128:
            raise ValueError("Initial register budget exceeded")
        callees = []
        declarations = list(re.finditer(r"^\s*\.type\s+(\S+),@function$", sass, re.M))
        for i, declaration in enumerate(declarations):
            if "validate_window_rows" not in declaration[1]:
                continue
            stop = declarations[i + 1].start() if i + 1 < len(declarations) else len(sass)
            region = inspect_region(sass[declaration.end():stop])
            if region[maximum] >= 128:
                raise ValueError("Metadata callee register budget exceeded")
            callees.append(region)
        allocation = list(re.finditer(r"USETMAXREG\.TRY_ALLOC[^;]+;", sass))
        if len(allocation) != 1:
            raise ValueError("Expected one epilogue allocation")
        budget = int(re.search(r"0x([0-9a-f]+)", allocation[0][0])[1], 16)
        stop = next((d.start() for d in declarations if d.start() > allocation[0].end()), len(sass))
        epilogue = inspect_region(sass[allocation[0].end():stop])
        if epilogue[maximum] >= budget:
            raise ValueError("Epilogue register budget exceeded")
        regions = [initial, *roles, *callees, epilogue]
        reports.append(dict(role_budgets=[r["budget"] for r in roles],
                            epilogue_budget=budget, maxima=[r[maximum] for r in regions],
                            preserved_external_diagnostic_calls=list(diagnostic),
                            local_memory=[r["local_memory_instructions"] for r in regions]))
    if (reports[0]["role_budgets"] != reports[1]["role_budgets"] or
            reports[0]["epilogue_budget"] != reports[1]["epilogue_budget"] or
            len(reports[0]["local_memory"]) != len(reports[1]["local_memory"])):
        raise ValueError("Changed role/resource coverage")
    for old, new in zip(reports[0]["local_memory"], reports[1]["local_memory"]):
        old_ops = Counter(op.split()[0] for op in old)
        new_ops = Counter(op.split()[0] for op in new)
        if new_ops - old_ops:
            raise ValueError(f"New local-memory instructions: {old_ops} -> {new_ops}")
    return reports


def compare_preparations(comparisons, removed, nvdisasm, output, normalize_source_metadata=False,
                         resource_gate=False, normalize_validation_symbols=False):
    """Prove cleanup preserves retained configurations and disassembled instructions."""
    report = output / "preservation_review.json"
    report.write_text(json.dumps(dict(status="INCOMPLETE")) + "\n")
    if normalize_validation_symbols and not normalize_source_metadata:
        raise ValueError("Validation-helper normalization requires source metadata normalization")
    checks = []
    for before_path, after_path in comparisons:
        before = json.loads((Path(before_path) / "manifest.json").read_text())
        after = json.loads((Path(after_path) / "manifest.json").read_text())
        if before["configurations"] != after["configurations"]:
            raise ValueError(f"Changed preparation coverage: {after_path}")
        entries = [{tuple(e["cases"]): e for e in manifest["specializations"]}
                   for manifest in (before, after)]
        if not entries[0] or entries[0].keys() != entries[1].keys():
            raise ValueError(f"Changed specialization coverage: {after_path}")
        for cases, old in entries[0].items():
            new = entries[1][cases]
            configs = [dict(re.findall(r"(\w+)=([^,)]+)", e["config"])) for e in (old, new)]
            for name in removed:
                if name in configs[1] or configs[0].pop(name, "0") != "0":
                    raise ValueError(f"Removed option was active or remains present: {name}, {cases}")
            if not configs[0] or configs[0] != configs[1]:
                raise ValueError(f"Changed retained configuration: {cases}")
            instructions, functions, metadata, disassembly = [], [], [], []
            for label, entry, manifest in (("before", old, before), ("after", new, after)):
                sass = subprocess.check_output(
                    [nvdisasm, str(Path(entry["target"]) / "kernel.cubin")], text=True)
                (output / f"{len(checks)}_{label}.sass").write_text(sass)
                disassembly.append(sass)
                source = None
                if normalize_source_metadata:
                    generated = (Path(entry["target"]) / "kernel.cu").read_text()
                    headers = re.findall(r'#include [<"]deep_gemm/impls/(sm100_fp8_fp4_mega_moe(?:_all_push)?\.cuh)[>"]', generated)
                    if len(headers) != 1:
                        raise ValueError("Expected one explicit MegaMoE implementation include")
                    header_root = (Path(manifest["header_root"]) if manifest.get("header_root") else
                                   Path(manifest["source"]) / "python/minisgl/kernel/csrc/deepgemm/deep_gemm")
                    source = header_root / "include/deep_gemm/impls" / headers[0]
                names, ops, changes = preparation_instructions(sass, source)
                functions.append(names)
                instructions.append(ops)
                metadata.append(changes)
            renames = {}
            if normalize_validation_symbols:
                functions[1], instructions[1], renames = normalize_validation_helper_symbols(
                    functions[0], functions[1], instructions[1])
            identical = bool(instructions[0]) and instructions[0] == instructions[1] and functions[0] == functions[1]
            resources = compare_resource_bounds(*disassembly, normalized_instructions=instructions) if resource_gate else None
            if not identical and not resource_gate:
                raise ValueError(f"Changed compiled instructions: {cases}; inspect saved SASS")
            checks.append(dict(cases=cases, before=old["target"], after=new["target"],
                               instructions=len(instructions[1]), identical_instructions=identical,
                               source_metadata=metadata, validation_helper_symbol_renames=renames,
                               resource_comparison=resources))
    if not checks:
        raise ValueError("No preparation comparisons")
    review = dict(status=("PASS_RESOURCE_BOUNDS_GPU_COMPARISON_REQUIRED" if resource_gate else
                         "PASS_RETAINED_INSTRUCTIONS_EXCEPT_SOURCE_AND_LOCAL_SYMBOLS" if normalize_validation_symbols else
                         "PASS_RETAINED_INSTRUCTIONS_EXCEPT_SOURCE_METADATA" if normalize_source_metadata
                          else "PASS_IDENTICAL_CONFIGURATIONS_AND_INSTRUCTIONS"),
                  removed_inactive_fields=removed, specializations=len(checks), checks=checks)
    report.write_text(json.dumps(review, indent=2) + "\n")
    print(review["status"], len(checks))


def review_scheduling(argv):
    parser = argparse.ArgumentParser(description='Review worker/row scheduling against a compiled control')
    parser.add_argument('root', type=Path)
    parser.add_argument('--control', default='control')
    parser.add_argument('--resource-policy', help='Arm that must implement the balanced resource-derived policy')
    parser.add_argument('--output', type=Path, help='New review directory; preserves previous failed evidence')
    args = parser.parse_args(argv)
    root = args.root
    plan = json.loads((root / 'plan.json').read_text())
    ep = plan['ep']
    arms = [a for a in plan['arm_prefixes'] if a != 'pull']
    if args.control not in arms or (args.resource_policy and args.resource_policy not in arms):
        raise ValueError('Unknown scheduling arm')
    out = args.output or root / 'scheduling_review'
    out.mkdir(exist_ok=False)
    (out / 'review.json').write_text('{"status":"INCOMPLETE"}\n')
    models = sorted({g['model'] for g in plan['groups']})
    sass, checks, resource_checks = {}, [], {}
    scheduling = {'num_dispatch_expert_ctas', 'all_push_metadata_row_partitions'}
    for model in models:
        prepared = {a: json.loads((root / f'preparation/{a}/{model}/manifest.json').read_text()) for a in arms}
        manifests = {a: json.loads((root / model / a / 'manifest.json').read_text()) for a in arms}
        for case in manifests[args.control]['cases']:
            cid = case['case_id']
            contract = dict(manifests[args.control]['contract'], **case.get('capture_contract', {}))
            experts = contract['experts'] // ep
            entries, configs = {}, {}
            for arm in arms:
                group = [g for g in plan['groups'] if g['model'] == model and g['arm'] == arm
                         and cid in g['env']['MEGAMOE_SWEEP_CASES'].split(',')]
                if len(group) != 1 or manifests[arm] != manifests[args.control]:
                    raise ValueError('Mismatched scheduling cases/contracts')
                env = group[0]['env']
                if not Path(prepared[arm]['source']).samefile(env['FASTAFD_SOURCE_REPO']):
                    raise ValueError('Preparation source differs from execution source')
                found = [e for e in prepared[arm]['specializations'] if e['mode'] == 'ordinary'
                         and any(c.endswith('/' + cid + '/selected') for c in e['cases'])]
                if len(found) != 1:
                    raise ValueError('Expected one prepared selected kernel')
                entry = entries[arm] = found[0]
                cfg = configs[arm] = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', entry['config'])}
                if plan.get('routing') == 'normal':
                    required = (case['rows_per_rank'] + case['global_tokens'] * contract['top_k']
                                + experts * (cfg['block_m'] - 1))
                    if cfg['block_m'] > 192 or contract['bucket'] < required:
                        raise ValueError('Normal-route capacity does not cover arbitrary expert skew')
                w, p = cfg['num_dispatch_expert_ctas'], cfg['all_push_metadata_row_partitions']
                full, push = cfg['all_push_full_tma'], cfg['num_dispatch_push_ctas']
                expected_w = (int(env['DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS']) if cfg['block_m'] > 32
                              else min(experts, 2 * ep))
                expected_p = int(env['DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS']) if full else 1
                if ((w, p) != (expected_w, expected_p) or w <= 0 or not 1 <= p <= 8 or w % p
                        or 2 * w + push + 1 > contract['num_sms']
                        or (w > experts and w % experts) or (w // p > experts and (w // p) % experts)
                        or cfg['smem_size'] + 1024 > 232448):
                    raise ValueError(f'Invalid worker/resource geometry: {model}/{cid}/{arm}')
                if arm == args.resource_policy and full:
                    capacity = min(2 * experts, (contract['num_sms'] - push - 1) // 2)
                    resource_w = max(n for n in range(1, capacity + 1) if experts % n == 0 or n % experts == 0)
                    if (w, p) != (resource_w, 2 if resource_w >= 2 * experts else 1):
                        raise ValueError('Resource policy differs from prospective rule')
                target = entry['target']
                if not Path(target).resolve().is_relative_to(Path(env['DG_JIT_CACHE_DIR']).resolve()):
                    raise ValueError('Prepared target outside execution cache')
                if target not in sass:
                    sass[target] = subprocess.check_output(['/usr/local/cuda/bin/nvdisasm',
                                                           str(Path(target) / 'kernel.cubin')], text=True)
                    (out / f'kernel_{len(sass)}.sass').write_text(sass[target])
            baseline = configs[args.control]
            for arm in arms:
                if ({k: v for k, v in configs[arm].items() if k not in scheduling}
                        != {k: v for k, v in baseline.items() if k not in scheduling}):
                    raise ValueError('Geometry differs beyond worker/row scheduling')
                before, after = entries[args.control]['target'], entries[arm]['target']
                if baseline['block_m'] <= 32 and before != after:
                    raise ValueError('Small kernel target changed')
                pair = before, after
                if pair not in resource_checks:
                    resource_checks[pair] = compare_resource_bounds(sass[before], sass[after],
                                                                   preserve_diagnostics=False)
                checks.append(dict(model=model, case=cid, arm=arm, target=after,
                                   identical_control_target=before == after, config=configs[arm]))
    report = dict(status='PASS_SCHEDULING_GEOMETRY_RESOURCES_SMALL_TARGETS', ep=ep,
                  resource_policy=args.resource_policy, checks=checks,
                  resource_checks=[dict(before=a, after=b, resources=v)
                                   for (a, b), v in resource_checks.items()])
    (out / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], len(checks), flush=True)


def review_rows(argv):
    parser = argparse.ArgumentParser(description='Review row-partition resources and topology against matching compiled controls')
    parser.add_argument('root', type=Path)
    args = parser.parse_args(argv)
    r = args.root
    scope = json.loads((r / 'scope.json').read_text())
    ep = scope['ep']
    totals = scope['totals']
    assert ep in (2, 4, 8, 16, 32, 64) and totals
    assert all((ts and len(set(ts)) == len(ts) and all((t > 0 and t % ep == 0 for t in ts)) for ts in totals.values()))
    d = r / 'resource_review'
    d.mkdir(exist_ok=True)
    cached = {}
    for (model, case_totals) in totals.items():
        for arm in ['control', 'workers', 'rows']:
            manifest = json.loads((r / f'preparation/{arm}/{model}/manifest.json').read_text())
            for total in case_totals:
                e = next((e for e in manifest['specializations'] if any((f'ep{ep}_tokens{total}/' in c for c in e['cases']))))
                target = e['target']
                if target not in cached:
                    cached[target] = subprocess.check_output(['/usr/local/cuda/bin/nvdisasm', str(Path(target) / 'kernel.cubin')], text=True)
                (d / f'{model}_{total}_{arm}.sass').write_text(cached[target])
    maximum = 'highest_register_including_wide_memory_operands'
    out = []
    for model in totals:
        manifests = {a: json.loads((r / f'preparation/{a}/{model}/manifest.json').read_text()) for a in ['control', 'workers', 'rows']}
        for (arm, manifest) in manifests.items():
            for total in totals[model]:
                e = next((e for e in manifest['specializations'] if any((f'ep{ep}_tokens{total}/' in c for c in e['cases']))))
                cfg = {k: int(v) for (k, v) in re.findall('(\\w+)=(\\d+)', e['config'])}
                assert cfg['all_push_balance_epilogue'] == 0
                assert cfg['all_push_return_wide'] == cfg['all_push_return_wg_sync'] == 0
                assert cfg['smem_size'] + 1024 <= 232448
                peers = [next((x for x in mf['specializations'] if any((f'ep{ep}_tokens{total}/' in c for c in x['cases'])))) for mf in manifests.values()]
                unchanged = len({x['target'] for x in peers}) == 1
                if unchanged:
                    assert all((x['config'] == e['config'] for x in peers)) and cfg['all_push_full_tma'] == 0
                target = Path(e['target'])
                sass = (d / f'{model}_{total}_{arm}.sass').read_text()
                regions = inspect_reachable_registers(sass)
                for reg in regions:
                    assert reg[maximum] < reg['budget'] and (unchanged or not reg['local_memory_instructions']), (model, total, arm, reg)
                    assert all(('CALL.REL.NOINC' in t and 'validate_window_rows' in t for t in reg['unresolved_transfers']))
                labels = list(re.finditer('^\\$[^\\n]*validate_window_rows[^\\n]*:\\s*$', sass, re.M))
                assert len(labels) == 2
                callee = []
                for label in labels:
                    end = re.search('^\\s*\\.type\\s', sass[label.end():], re.M)
                    assert end
                    reg = inspect_region(sass[label.end():label.end() + end.start()])
                    assert reg[maximum] < 128 and (unchanged or not reg['local_memory_instructions'])
                    callee.append(reg[maximum])
                initial = inspect_region(sass[:sass.index('USETMAXREG.')])
                assert initial[maximum] < 128 and (unchanged or not initial['local_memory_instructions'])
                alloc = list(re.finditer('USETMAXREG\\.TRY_ALLOC[^;]+;', sass))
                assert len(alloc) == 1
                end = sass.rfind('.type', alloc[0].end(), labels[0].start())
                epi = inspect_region(sass[alloc[0].end():end])
                assert epi[maximum] < (168 if cfg['all_push_dedicated_dispatch'] else 208)
                assert unchanged or (len(epi['local_memory_instructions']) == 3 and all((t.startswith('STL') for t in epi['local_memory_instructions'])))
                out.append(dict(model=model, total=total, arm=arm, config=e['config'], unchanged_control_target=str(target) if unchanged else None, local_memory=dict(initial=initial['local_memory_instructions'], roles=[x['local_memory_instructions'] for x in regions], epilogue=epi['local_memory_instructions']), initial=initial[maximum], caller=[x[maximum] for x in regions], callee=callee, epilogue=epi[maximum], existing_assert_stores=3))
    assert len(out) == 3 * sum(map(len, totals.values()))
    (d / 'gates.json').write_text(json.dumps(dict(status='PASS_CPU_RESOURCES_METADATA_ROW_PARTITIONS', ordinary=out, partition_experiment=True), indent=2) + '\n')
    print('PASS_METADATA_ROW_PARTITIONS_CPU_RESOURCES', len(out))
    changes = []
    for model in totals:
        manifests = {a: json.loads((r / f'preparation/{a}/{model}/manifest.json').read_text()) for a in ['control', 'workers', 'rows']}
        for total in totals[model]:
            entries = {a: next((e for e in mf['specializations'] if any((f'ep{ep}_tokens{total}/' in c for c in e['cases'])))) for (a, mf) in manifests.items()}
            configs = {a: {k: int(v) for (k, v) in re.findall('(\\w+)=(\\d+)', e['config'])} for (a, e) in entries.items()}
            wide = configs['control']['all_push_full_tma'] != 0
            partitions = {a: cfg.pop('all_push_metadata_row_partitions') for (a, cfg) in configs.items()}
            workers = {a: cfg.pop('num_dispatch_expert_ctas') for (a, cfg) in configs.items()}
            assert partitions == dict(control=1, workers=1, rows=2 if wide else 1)
            contract = json.loads((r / model / 'control/manifest.json').read_text())['contract']
            base = min(contract['experts'] // ep, 2 * ep)
            requested_workers = 2 * base
            assert workers == dict(control=base, workers=requested_workers if wide else base, rows=requested_workers if wide else base)
            if wide:
                assert requested_workers // 2 == base, (model, total, base, requested_workers)
            assert configs['control'] == configs['workers'] == configs['rows'], (model, total, configs)
            if not wide:
                assert len({e['target'] for e in entries.values()}) == 1
            changes.append(dict(model=model, total=total, block_m=configs['control']['block_m'], workers=workers, partitions=partitions))
    (d / 'geometry.json').write_text(json.dumps(changes, indent=2) + '\n')
    print('PASS_ROW_PARTITION_GEOMETRY', json.dumps(changes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="?", type=Path, help="Ordinary source-scoped jit/cache directory")
    parser.add_argument("--compare-preparations", type=Path,
                        help="JSON pairs of before/after preparation directories; require identical code")
    parser.add_argument("--removed-config", action="append", default=[],
                        help="Removed configuration field, required inactive in the reference")
    parser.add_argument("--normalize-source-metadata", action="store_true",
                        help="Compare renamed entry symbols and proven assertion source-line arguments semantically")
    parser.add_argument("--normalize-validation-helper-symbols", action="store_true",
                        help="Align same-ordinal local validation-helper names; still compare every body and call instruction")
    parser.add_argument("--resource-gate", action="store_true",
                        help="Gate changed code on register budgets/new spills; requires a later matched GPU comparison")
    parser.add_argument("--nvdisasm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--include-instrumented", action="store_true",
                        help="Also inspect IKET builds, labeled separately in the results")
    args = parser.parse_args()
    if bool(args.cache) == bool(args.compare_preparations):
        parser.error("Choose a cache or --compare-preparations")
    if args.removed_config and not args.compare_preparations:
        parser.error("--removed-config requires --compare-preparations")
    if args.normalize_validation_helper_symbols and not (args.compare_preparations and args.normalize_source_metadata):
        parser.error("--normalize-validation-helper-symbols requires --compare-preparations and --normalize-source-metadata")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compare_preparations:
        compare_preparations(json.loads(args.compare_preparations.read_text()),
                             args.removed_config, args.nvdisasm, args.output, args.normalize_source_metadata,
                             args.resource_gate, args.normalize_validation_helper_symbols)
        return
    results = []
    for source in sorted(args.cache.glob("kernel.sm100_fp8_fp4_mega_moe.*/kernel.cu")):
        text = source.read_text()
        arguments = text.split("sm100_fp8_fp4_mega_moe_impl<", 1)[1].split(">);", 1)[0].strip()
        values = [v.strip() for v in arguments.split(",")]
        if args.bucket is not None and int(values[0]) != args.bucket:
            continue
        if args.ranks is not None and int(values[19]) != args.ranks:
            continue
        cubin = source.with_suffix(".cubin")
        sass = subprocess.check_output([args.nvdisasm, str(cubin)], text=True)
        instrumented = "__iket_meta_info" in sass
        if instrumented and not args.include_instrumented:
            continue
        destination = args.output / f"kernel_{len(results)}_sass.txt"
        destination.write_text(sass)
        results.append({"template_arguments": arguments, "cubin": str(cubin),
                        "sass": str(destination), "iket_build": instrumented,
                        **inspect(sass)})
    if not results:
        raise ValueError("No ordinary kernels matched the requested geometry")
    (args.output / "compiled_review.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(json.dumps({**{key: value for key, value in result.items()
                             if key not in {"template_arguments", "local_memory_instructions"}},
                          "local_memory_instruction_count": len(result["local_memory_instructions"])}))
    for result in results:
        if result["highest_register_including_wide_memory_operands"] >= result["budget"]:
            print("MANUAL_CONTROL_FLOW_REVIEW: linear span includes registers above the "
                  "dispatch budget; other warp roles may legitimately use them.")


if __name__ == "__main__":
    if sys.argv[1:2] == ["review-scheduling"]:
        review_scheduling(sys.argv[2:])
    elif sys.argv[1:2] == ["review-rows"]:
        review_rows(sys.argv[2:])
    else:
        main()
