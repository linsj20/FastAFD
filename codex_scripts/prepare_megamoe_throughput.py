#!/usr/bin/env python3
"""Prepare a mirrored EP throughput comparison on the execution filesystem."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sweep_policy_difference(a, b):
    """Only scheduling flags proven to share the existing symmetric-buffer layout."""
    ignored = {"MEGAMOE_SWEEP_ROOT"}
    changed = {key for key in a.keys() | b.keys() if key not in ignored and a.get(key) != b.get(key)}
    allowed = {"DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS", "DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS"}
    if not changed or not changed <= allowed or any(key not in a or key not in b for key in changed):
        raise ValueError(f"Unsupported policy/environment differences: {sorted(changed)}")
    return sorted(changed)


def normal_route_bucket(total, ep, experts, top_k, max_tile=192):
    """Capacity for all routes landing locally plus padding of every local expert."""
    if min(total, ep, experts, top_k, max_tile) <= 0 or total % ep or experts % ep or top_k > experts:
        raise ValueError('Invalid normal-routing shape')
    required = total // ep + total * top_k + experts // ep * max_tile
    return (required + 383) // 384 * 384


def prepare_normal(argv):
    parser = argparse.ArgumentParser(description='Prepare normal-route graphs with measured ordinary settings')
    parser.add_argument('sweep', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--totals', required=True, help='Measured totals, with the small control first')
    parser.add_argument('--preparations', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args(argv)
    old, root = args.sweep.absolute(), args.output.absolute()
    plan, summary = read(old / 'plan.json'), read(old / 'summary.json')
    totals = [int(t) for t in args.totals.split(',')]
    ep, arms = plan['ep'], plan['arm_prefixes']
    sources = {g['env']['FASTAFD_SOURCE_REPO'] for g in plan['groups']}
    if (root.exists() or summary['status'] != 'PASS' or len(sources) != 1
            or not totals or len(set(totals)) != len(totals) or min(totals) != totals[0]
            or arms[0] != 'control' or arms[-1] != 'pull' or args.candidate not in arms[1:-1]):
        raise ValueError('Require a new output, passing single-source sweep and small-first measured totals')
    source = sources.pop()
    for group in plan['groups']:
        if (group['control_case'] != f'ep{ep}_tokens{totals[0]}'
                or group['env']['DG_JIT_CACHE_DIR'] != str(old / 'cache/jit')):
            raise ValueError('Require the measured small control and a sweep-local cache')
    scripts = Path(__file__).resolve().parent
    models = sorted({g['model'] for g in plan['groups']})
    specs, manifests = {}, {}
    for model in models:
        original = read(old / model / 'control/manifest.json')
        cases = {c['global_tokens']: c for c in original['cases']}
        if not set(totals) <= cases.keys():
            raise ValueError('Normal-route totals must already be measured')
        manifest = dict(original, cases=[])
        for total in totals:
            case = json.loads(json.dumps(cases[total]))
            contract = dict(original['contract'], **case.get('capture_contract', {}))
            if total // ep * contract['top_k'] == 24:
                raise ValueError('This shape triggers additional fixed-route smoke cases; explicit preparation required')
            for arm in arms:
                prepared = read(args.preparations / arm / ('' if arm == 'pull' else model) / 'manifest.json')
                found = [e for e in prepared['specializations'] if e['mode'] == 'ordinary'
                         and any(c.endswith('/' + case['case_id'] + '/selected') for c in e['cases'])]
                # Pull may contain both models with the same case label; both
                # must satisfy the conservative tile bound.
                if (not found or not Path(prepared['source']).samefile(source)
                        or any(int(re.search(r'\bblock_m=(\d+)', e['config'])[1]) > 192 for e in found)):
                    raise ValueError('Unprepared source/shape or unsupported tile above192')
            case['capture_contract'] = dict(case.get('capture_contract', {}), bucket=normal_route_bucket(
                total, ep, contract['experts'], contract['top_k']))
            manifest['cases'].append(case)
        manifests[model] = manifest
        specs[model] = totals
    # Retain execution settings exactly; only source-independent output/cache paths change.
    def moved(value):
        return json.loads(json.dumps(value).replace(str(old) + '/', str(root) + '/'))
    groups, runs, commands = [], [], []
    for group in moved(plan['groups']):
        selected = [c for c in manifests[group['model']]['cases']
                    if c['case_id'] in group['env']['MEGAMOE_SWEEP_CASES'].split(',')]
        if not selected:
            continue
        group['env']['MEGAMOE_SWEEP_CASES'] = ','.join(c['case_id'] for c in selected)
        groups.append(group)
        write(Path(group['env']['MEGAMOE_SWEEP_ROOT']) / 'manifest.json', manifests[group['model']])
        for case in selected:
            c = dict(manifests[group['model']]['contract'], **case['capture_contract'])
            path = root / 'runs' / group['model'] / group['arm'] / case['case_id']
            path.mkdir(parents=True)
            env = dict(group['env'], OUTPUT_ROOT=str(path), PYTHONPATH=source + '/python',
                       PYTHONUNBUFFERED='1', MINISGL_DEEPGEMM_BUILD_DIR=str(root / 'cache/native'),
                       TVM_FFI_CACHE_DIR=str(root / 'cache/tvm'), DG_JIT_IKET_INCLUDE='',
                       MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE='fp4', MEGAMOE_SMOKE_RANK_MODE='run',
                       MEGAMOE_SMOKE_PROFILE_ONLY='1', MEGAMOE_SMOKE_FIXED_PERFECT_ONLY='0',
                       MEGAMOE_SMOKE_PERFECT_ROUTING='0', MEGAMOE_SMOKE_SPECIALIZATION_MODES='1:1:1:1',
                       MEGAMOE_SMOKE_ROWS=str(case['rows_per_rank']), MEGAMOE_SMOKE_BUCKET=str(c['bucket']),
                       MEGAMOE_SMOKE_SMS=str(c['num_sms']), MEGAMOE_SMOKE_NUM_EXPERTS=str(c['experts']),
                       MEGAMOE_SMOKE_HIDDEN=str(c['hidden']), MEGAMOE_SMOKE_INTERMEDIATE=str(c['intermediate']),
                       MEGAMOE_SMOKE_TOP_K=str(c['top_k']), MEGAMOE_SMOKE_WARMUPS='10',
                       MEGAMOE_SMOKE_ITERATIONS='1000', MEGAMOE_SMOKE_CUDA_PROFILER_RANGE='0',
                       MEGAMOE_SMOKE_OUTPUT=str(path / 'result.json'),
                       MEGAMOE_RANK_TIMING_OUTPUT=str(path / 'rank_timings.json'),
                       MEGAMOE_RANK_TIMING_MODULE_ROOT=source + '/codex_scripts')
            for key in ('MEGAMOE_SMOKE_GLOBAL_TOKENS', 'MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX',
                        'MEGAMOE_SMOKE_WEIGHT_CACHE_INPUT', 'MEGAMOE_SMOKE_WEIGHT_CACHE_OUTPUT'):
                env.pop(key, None)
            if env['MINISGL_MEGAMOE_PAYLOAD_MODE'] == 'all_push':
                env['MEGAMOE_RESULT_REFERENCE_DIR'] = str(root / 'reference' / group['model'] / case['case_id'])
            commands.append(['env'] + [f'{k}={v}' for k, v in sorted(env.items())] +
                            ['bash', source + '/scripts/experiments/afd/oci_hsg/megamoe_same_rank_smoke.sbatch'])
            runs.append(dict(model=group['model'], total=case['global_tokens'], arm=group['arm'], path=str(path), env=env))
    write(root / 'plan.json', dict(ep=ep, arm_prefixes=arms, groups=groups, routing='normal'))
    write(root / 'scope.json', dict(ep=ep, source=source, specs=specs, arm_prefixes=arms, runs=runs,
          routing='normal', candidate=args.candidate, small_control=totals[0], calls_per_arm=1000,
          capacity='All global routes local plus192 padded rows per local expert; tiles above192 rejected'))
    write(root / 'commands.json', commands)
    (root / 'logs').mkdir()
    cpu = [moved(c) for c in read(old / 'cpu_prepare/commands.json')
           if any(v.endswith('/precompile_megamoe_sweep.py') or 'BARRIER_CPU_READY' in v for v in c)]
    donors = list((old / 'cache/native/lib').glob('_minisgl_deepgemm.*.so'))
    if len(donors) != 1 or not cpu:
        raise ValueError('Missing compatible native donor or compile commands')
    cpu.insert(0, ['python3', '-B', str(scripts / 'seed_native_cache.py'), str(donors[0]), '--copy',
                  '--cache-root', str(root / 'cache/native'), '--output', str(root / 'native_seed.json')])
    env = groups[0]['env']
    cpu += [['env', 'PYTHONDONTWRITEBYTECODE=1', env['FASTAFD_VENV'] + '/bin/python',
             str(scripts / 'megamoe_header_smoke.py'), 'prepare-routers', '--source', source,
             '--cache-root', str(root / 'cache/tvm'), '--output', str(root / 'router_modules.json')],
            ['python3', '-B', str(scripts / 'review_megamoe_dispatch_sass.py'), 'review-scheduling', str(root)],
            ['python3', '-B', str(scripts / 'run_megamoe_sweep_group.py'), 'validate', str(root / 'plan.json')]]
    write(root / 'cpu_prepare/commands.json', cpu)
    for phase in ('cpu_prepare', 'gpu', 'analysis'):
        submit = moved(read(old / phase / 'submit.json'))
        submit = [v for v in submit if not v.startswith('--dependency=')]
        if phase == 'gpu':
            submit = [v for v in submit if not v.startswith('--wrap=')]
            submit += ['--wrap=' + shlex.join(['python3', '-B', str(Path(__file__).resolve()), 'run-graphs', str(root)])]
        else:
            write(root / phase / 'image_commands.json', moved(read(old / phase / 'image_commands.json')))
        write(root / phase / 'submit.json', submit)
        write(root / phase / 'env.json', dict(OUTPUT_ROOT=str(root / phase)))
    write(root / 'analysis/commands.json', [['python3', '-B', str(scripts / 'analyze_megamoe_throughput.py'),
                                           'review-normal', str(root)]])
    print('NORMAL_PLAN_PREPARED', root, len(runs), flush=True)


def prepare_sweep_calibration(argv):
    parser = argparse.ArgumentParser(description="Prepare AA/AB/BB from a completed ordinary sweep")
    parser.add_argument("sweep", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preparations", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cases", required=True, help="Comma-separated model:total pairs")
    parser.add_argument("--entry", choices=("synchronized", "queued"), default="synchronized")
    args = parser.parse_args(argv)
    sweep, root = args.sweep.absolute(), args.output.absolute()
    labels = [args.control, args.candidate]
    cases = [(model, int(total)) for model, total in (x.split(":") for x in args.cases.split(","))]
    if (root.exists() or len(set(labels)) != 2 or not cases or len(set(cases)) != len(cases)
            or any(not re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in labels)):
        raise ValueError("Require a new output, distinct arm labels and distinct cases")
    plan, summary = read(sweep / "plan.json"), read(sweep / "summary.json")
    ep = plan["ep"]
    if summary["status"] != "PASS" or ep not in (2, 4, 8, 16, 32, 64):
        raise ValueError("Requires a passing ordinary sweep")
    selected = {}
    evidence = []
    for model, total in cases:
        case = f"ep{ep}_tokens{total}"
        measured = summary["cases"][f"{model}/{case}"]
        if total <= 0 or total % ep or not set(labels) <= measured["paired_means"].keys():
            raise ValueError(f"Invalid or unmeasured case: {model}/{case}")
        arms, contracts, targets = {}, {}, {}
        for label in labels:
            groups = [g for g in plan["groups"] if g["model"] == model and g["arm"] == label
                      and case in g["env"]["MEGAMOE_SWEEP_CASES"].split(",")]
            if len(groups) != 1:
                raise ValueError(f"Expected one measured group: {model}/{case}/{label}")
            env = arms[label] = dict(groups[0]["env"])
            required = dict(DG_JIT_REQUIRE_CACHE="1", MINISGL_MEGAMOE_PAYLOAD_MODE="all_push",
                            MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT="joined", MEGAMOE_SMOKE_GRAPH_BATCH="1",
                            MEGAMOE_SMOKE_RESET_DISPATCH_PAYLOAD="1", MEGAMOE_SMOKE_PREPARE_BARRIER="collective",
                            MEGAMOE_SMOKE_BALANCE_RANKS="1", MEGAMOE_RESULT_MODE="compare",
                            FASTAFD_IKET_CAPTURE_PHASE="normal", MEGAMOE_SMOKE_TIMING_SCOPE="pipeline")
            if any(env.get(key) != value for key, value in required.items()):
                raise ValueError(f"Unsupported ordinary capture contract: {model}/{case}/{label}")
            manifest = read(Path(env["MEGAMOE_SWEEP_ROOT"]) / "manifest.json")
            entries = [x for x in manifest["cases"] if x["case_id"] == case]
            if len(entries) != 1 or entries[0]["ep"] != ep or entries[0]["global_tokens"] != total:
                raise ValueError("Case manifest mismatch")
            contract = contracts[label] = dict(manifest["contract"], **entries[0].get("capture_contract", {}))
            preparation = read(args.preparations / label / model / "manifest.json")
            if not Path(preparation["source"]).samefile(env["FASTAFD_SOURCE_REPO"]):
                raise ValueError("Preparation source differs from measured source")
            targets[label] = {}
            combine_check = env.get('MEGAMOE_SMOKE_COMPARE_COMBINE', '0')
            if combine_check not in ('0', '1'):
                raise ValueError('Invalid completed-sweep combine check')
            for mode in (('selected', 'combine_reference') if combine_check == '1' else ('selected',)):
                matches = [x for x in preparation["specializations"] if x["mode"] == "ordinary"
                           and any(name.endswith(f"/{case}/{mode}") for name in x["cases"])]
                if len(matches) != 1:
                    raise ValueError(f"Missing prepared specialization: {model}/{case}/{label}/{mode}")
                target = Path(matches[0]["target"])
                if (not (target / "kernel.cubin").is_file()
                        or not target.resolve().is_relative_to(Path(env["DG_JIT_CACHE_DIR"]).resolve())):
                    raise ValueError("Missing cubin or mismatched JIT cache")
                targets[label][mode] = matches[0]
            for rank in range(ep):
                reference = Path(env["MEGAMOE_RESULT_REFERENCE_DIR"]) / (
                    f"fixed_bucket{contract['bucket']}_rows{total // ep}_sms{contract['num_sms']}_rank{rank}.pt")
                if not reference.is_file():
                    raise FileNotFoundError(reference)
            for key in ("FASTAFD_SOURCE_REPO", "FASTAFD_VENV", "FASTAFD_IKET_DEEPGEMM_CACHE_ROOT",
                        "FASTAFD_IKET_TVM_CACHE_ROOT", "DG_JIT_CACHE_DIR"):
                if not Path(env[key]).is_dir():
                    raise FileNotFoundError(env[key])
        changed = sweep_policy_difference(arms[labels[0]], arms[labels[1]])
        if contracts[labels[0]] != contracts[labels[1]]:
            raise ValueError("Arms do not share the same shape/capacity/routing contract")
        for mode in targets[labels[0]]:
            configs = [re.sub(r"\b(num_dispatch_expert_ctas|all_push_metadata_row_partitions)=\d+",
                              r"\1=*", targets[label][mode]["config"]) for label in labels]
            if configs[0] != configs[1]:
                raise ValueError("Prepared geometry differs beyond worker/row scheduling")
        selected[model, total] = arms, contracts[labels[0]], changed
        evidence.append(dict(model=model, total=total, changed_environment=changed, targets=targets))
    sources = {env["FASTAFD_SOURCE_REPO"] for arms, _, _ in selected.values() for env in arms.values()}
    if len(sources) != 1:
        raise ValueError("Calibration requires one frozen source")
    source = sources.pop()
    for sub in ("workflow", "gpu", "analysis", "logs"):
        (root / sub).mkdir(parents=True)
    scripts = Path(__file__).resolve().parent
    for name in ("prepare_megamoe_throughput.py", "interleave_megamoe_smoke.py",
                 "analyze_megamoe_throughput.py", "compare_megamoe_rank_timelines.py",
                 "collect_cuda_rank_timings.py"):
        shutil.copyfile(scripts / name, root / "workflow" / name)
    runs, commands = [], []
    for order, pairs in (("abba", ("aa", "ab", "bb")), ("baab", ("bb", "ab", "aa"))):
        for pair in pairs:
            for model, total in cases:
                arms, contract, changed = selected[model, total]
                order_labels = labels if order == "abba" else labels[::-1]
                policies = dict(zip(labels, [labels[1] if pair == "bb" else labels[0],
                                            labels[0] if pair == "aa" else labels[1]]))
                variants = {label: {key: arms[policies[label]][key] for key in changed} for label in order_labels}
                env = dict(arms[policies[order_labels[0]]])
                path = root / "runs" / pair / model / order / f"ep{ep}_tokens{total}"
                path.mkdir(parents=True)
                env.update(OUTPUT_ROOT=str(path), PYTHONPATH=source + "/python", PYTHONUNBUFFERED="1",
                           MINISGL_DEEPGEMM_BUILD_DIR=env["FASTAFD_IKET_DEEPGEMM_CACHE_ROOT"],
                           TVM_FFI_CACHE_DIR=env["FASTAFD_IKET_TVM_CACHE_ROOT"], DG_JIT_IKET_INCLUDE="",
                           MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE="fp4", MEGAMOE_SMOKE_RANK_MODE="run",
                           MEGAMOE_SMOKE_PROFILE_ONLY="1", MEGAMOE_SMOKE_FIXED_PERFECT_ONLY="1",
                           MEGAMOE_SMOKE_PRODUCTION_ROUTES="1",
                           MEGAMOE_SMOKE_ROWS=str(total // ep), MEGAMOE_SMOKE_GLOBAL_TOKENS=str(total),
                           MEGAMOE_SMOKE_BUCKET=str(contract["bucket"]), MEGAMOE_SMOKE_SMS=str(contract["num_sms"]),
                           MEGAMOE_SMOKE_NUM_EXPERTS=str(contract["experts"]),
                           MEGAMOE_SMOKE_HIDDEN=str(contract["hidden"]),
                           MEGAMOE_SMOKE_INTERMEDIATE=str(contract["intermediate"]),
                           MEGAMOE_SMOKE_TOP_K=str(contract["top_k"]), MEGAMOE_SMOKE_WARMUPS="10",
                           MEGAMOE_SMOKE_ITERATIONS="1000", MEGAMOE_SMOKE_CUDA_PROFILER_RANGE="0",
                           MEGAMOE_INTERLEAVED_ENTRY=args.entry, MEGAMOE_INTERLEAVED_VARIANTS=json.dumps(variants),
                           MEGAMOE_SMOKE_OUTPUT=str(path / "result.json"),
                           MEGAMOE_INTERLEAVED_OUTPUT=str(path / "interleaved.json"),
                           MEGAMOE_SMOKE_TEST_SCRIPT=str(root / "workflow/interleave_megamoe_smoke.py"))
                for key in ("MEGAMOE_RANK_TIMING_OUTPUT", "MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX",
                            "MEGAMOE_SMOKE_WEIGHT_CACHE_INPUT", "MEGAMOE_SMOKE_WEIGHT_CACHE_OUTPUT"):
                    env.pop(key, None)
                commands.append(["env"] + [f"{key}={value}" for key, value in sorted(env.items())] +
                                ["bash", source + "/scripts/experiments/afd/oci_hsg/megamoe_same_rank_smoke.sbatch"])
                runs.append(dict(model=model, total=total, pair=pair, order=order, path=str(path), env=env))
    count = len(runs)
    scope = dict(source=source, ep=ep, control=labels[0], candidate=labels[1], routing="balanced",
                 entry_protocol=args.entry, runs=runs, cases=cases, pairs=["aa", "ab", "bb"],
                 calls_per_arm=1000, warmups_per_arm=10,
                 expected_correctness=dict(RESULT_COMPARE=count * ep, REPLAY_EXACT=count * ep,
                                           INTERLEAVED_EXACT=count * ep),
                 reuse=dict(sweep=str(sweep), preparations=str(args.preparations)),
                 scope="Same-process graph timing including preparation. AA/BB identical-policy controls; "
                       "AB policy comparison. All samples retained; no pull/ordinary kernel score replacement.")
    write(root / "scope.json", scope)
    write(root / "commands.json", commands)
    write(root / "analysis/commands.json", [["python3", "-B",
          str(root / "workflow/analyze_megamoe_throughput.py"), "review-interleaved", str(root)]])
    template = read(sweep / "gpu/submit.json")
    template = [arg for arg in template if not arg.startswith(("--dependency=", "--wrap=", "--output=",
                                                               "--error=", "--time=", "--job-name="))]
    if "--qos=short" not in template or "--partition=batch" not in template:
        raise ValueError("GPU template must use HSG short")
    runner = root / "workflow/prepare_megamoe_throughput.py"
    template += ["--time=00:30:00", f"--job-name=megamoe-calibration-{ep}",
                 f"--output={root}/gpu/capture-%j.out", f"--error={root}/gpu/capture-%j.err",
                 "--wrap=" + shlex.join(["python3", "-B", str(runner), "run-calibration", str(root)])]
    write(root / "gpu/submit.json", template)
    write(root / "gpu/env.json", dict(OUTPUT_ROOT=str(root / "gpu")))
    write(root / "plan_review.json", dict(status="PASS_REUSE_COMPLETED_SWEEP_KERNELS_AND_REFERENCES",
          processes=count, rank_samples=count * ep * 2000, new_kernel_builds=0, evidence=evidence,
          correctness_expected=scope["expected_correctness"], source=source))
    print(json.dumps(dict(root=str(root), processes=count, rank_samples=count * ep * 2000)), flush=True)


def run_calibration(argv):
    parser = argparse.ArgumentParser(description="Run prepared graph commands in one shared container step")
    parser.add_argument("root", type=Path)
    parser.add_argument("--rank-worker", action="store_true")
    args = parser.parse_args(argv)
    root, plan = args.root, read(args.root / "scope.json")
    if int(os.environ["SLURM_NTASKS"]) != plan["ep"]:
        raise ValueError("Allocation does not match calibration EP")
    if args.rank_worker:
        rank = os.environ["SLURM_PROCID"]
        temp = root / "temp" / ("rank-" + rank)
        temp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(temp)
        os.environ["MEGAMOE_SMOKE_DEVICE_INDEX"] = os.environ["SLURM_LOCALID"]
        with (root / "logs" / ("rank-" + rank + ".log")).open("x") as log:
            for command in read(root / "commands.json"):
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        print("CALIBRATION_RANK_COMPLETE", rank, flush=True)
        return
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                                    text=True).splitlines()
    per_node, remainder = divmod(plan["ep"], len(nodes))
    images = {run["env"]["FASTAFD_IMAGE"] for run in plan["runs"]}
    if (remainder or not 1 <= per_node <= 4 or len(images) != 1
            or len({re.sub(r"-T\d+$", "", node) for node in nodes}) != 1):
        raise ValueError("Require one image and a single matching NVL72 allocation")
    env = dict(os.environ, MASTER_ADDR=nodes[0])
    env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run(["srun", f"--nodes={len(nodes)}", f"--ntasks={plan['ep']}",
                    f"--ntasks-per-node={per_node}", "--kill-on-bad-exit=1",
                    "--container-image=" + images.pop(), "--container-mount-home",
                    "--container-mounts=/lustre:/lustre", "--no-container-remap-root",
                    "env", "-u", "CUDA_VISIBLE_DEVICES", "python3", "-B", str(Path(__file__).resolve()),
                    "run-calibration", str(root), "--rank-worker"], env=env, check=True)


def prepare_calibration(argv):
    parser = argparse.ArgumentParser(description="Reuse validated interleaved kernels for matched AA/AB/BB calibration")
    parser.add_argument("normal", type=Path)
    parser.add_argument("interleaved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", required=True, help="Comma-separated model:total pairs")
    args = parser.parse_args(argv)
    normal, base, root = args.normal.absolute(), args.interleaved.absolute(), args.output.absolute()
    cases = [(m, int(t)) for m, t in (x.split(":") for x in args.cases.split(","))]
    assert cases and len(set(cases)) == len(cases) and not root.exists()
    plan = read(base / "scope.json")
    prior = read(base / "analysis/review.json")
    normal_review = read(normal / "analysis/review.json")
    assert prior["status"] == "PASS_INTERLEAVED_COVERAGE_AND_EXACT_OUTPUTS"
    assert normal_review["status"] == "PASS_NORMAL_ROUTE_COMPLETENESS"
    ep = plan["ep"]
    selected = {}
    for model, total in cases:
        for order in ["abba", "baab"]:
            run = next(x for x in plan["runs"] if (x["model"], x["total"], x["order"]) == (model, total, order))
            assert run["env"]["MEGAMOE_INTERLEAVED_ENTRY"] == "queued"
            assert run["env"]["DG_JIT_REQUIRE_CACHE"] == "1"
            assert Path(run["env"]["MEGAMOE_RESULT_REFERENCE_DIR"]).is_dir()
            selected[model, total, order] = run
    for sub in ["gpu", "analysis", "workflow", "logs"]:
        (root / sub).mkdir(parents=True)
    wrapper = root / "workflow/interleave_megamoe_smoke.py"
    wrapper.write_text((base / "workflow/interleave_megamoe_smoke.py").read_text())
    runs, commands = [], []
    for order, pairs in [("abba", ["aa", "ab", "bb"]), ("baab", ["bb", "ab", "aa"])]:
        for pair in pairs:
            for model, total in cases:
                original = selected[model, total, order]
                env = dict(original["env"])
                variants = json.loads(env["MEGAMOE_INTERLEAVED_VARIANTS"])
                a, b = variants["control"], variants["workers"]
                assert a.keys() == b.keys() == {"DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS"}
                values = {"control": b if pair == "bb" else a, "workers": a if pair == "aa" else b}
                variants = {label: values[label] for label in variants}
                path = root / "runs" / pair / model / order / f"ep{ep}_tokens{total}"
                path.mkdir(parents=True)
                env.update(OUTPUT_ROOT=str(path), MEGAMOE_SMOKE_OUTPUT=str(path / "result.json"),
                           MEGAMOE_INTERLEAVED_OUTPUT=str(path / "interleaved.json"),
                           MEGAMOE_SMOKE_TEST_SCRIPT=str(wrapper),
                           MEGAMOE_INTERLEAVED_VARIANTS=json.dumps(variants))
                commands.append(["env"] + [f"{k}={v}" for k, v in sorted(env.items())] +
                                ["bash", str(Path(plan["source"]) / "scripts/experiments/afd/oci_hsg/megamoe_same_rank_smoke.sbatch")])
                runs.append(dict(model=model, total=total, order=order, pair=pair, path=str(path), env=env))
    count = len(runs)
    assert count == len(cases) * 6
    write(root / "commands.json", commands)
    write(root / "scope.json", dict(source=plan["source"], ep=ep, runs=runs, cases=cases,
          processes=count, rank_samples=count * ep * 2000,
          scope="AA and BB compare two graphs of the same kernel policy; AB compares policies. Queued entry, both orders, all samples retained. No pull comparison or kernel change.",
          reuse=dict(normal=str(normal), interleaved=str(base))))
    logged = "import json,os,subprocess,sys;from pathlib import Path;r=Path(sys.argv[1]);f=(r/'logs'/('rank-'+os.environ['SLURM_PROCID']+'.log')).open('w');[subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT) for c in json.load(open(r/'commands.json'))];f.close()"
    write(root / "gpu/commands.json", [["python3", "-B", "-c", logged, str(root)]])
    review = (base / "analysis/review.py").read_text()
    old_counts = prior["correctness"]
    new_counts = dict(RESULT_COMPARE=count * ep * 9, REPLAY_EXACT=count * ep, INTERLEAVED_EXACT=count * ep)
    for key in new_counts:
        before = repr(key) + ":" + str(old_counts[key])
        assert before in review, before
        review = review.replace(before, repr(key) + ":" + str(new_counts[key]))
    before = "rank_samples=" + str(prior["rank_samples"])
    assert before in review
    review = review.replace(before, "rank_samples=" + str(count * ep * 2000))
    before = "entry_protocol=v['entry_protocol'],model=run['model']"
    assert before in review
    review = review.replace(before, "pair=run['pair']," + before)
    (root / "analysis/review.py").write_text(review)
    write(root / "analysis/commands.json", [["python3", "-B", str(root / "analysis/review.py"), str(root)]])
    for phase in ["gpu", "analysis"]:
        submit = read(normal / phase / "submit.json")
        wrap = next(x.removeprefix("--wrap=") for x in submit if x.startswith("--wrap="))
        template_input = Path(shlex.split(wrap)[-1])
        expected_input = normal if phase == "gpu" else normal / "analysis/commands.json"
        assert template_input.samefile(expected_input), template_input
        # Slurm templates may retain a lexical alias different from resolve().
        # Replace the path actually present in the template, then verify targets.
        template_root = template_input if phase == "gpu" else template_input.parent.parent
        submit = [x.replace(str(template_root), str(root)) for x in submit if not x.startswith("--dependency=")]
        submit = ["--time=00:15:00" if x.startswith("--time=") else
                  "--job-name=megamoe-calibration-" + phase if x.startswith("--job-name=") else x for x in submit]
        wrap = next(x.removeprefix("--wrap=") for x in submit if x.startswith("--wrap="))
        expected_input = root if phase == "gpu" else root / "analysis/commands.json"
        assert shlex.split(wrap)[-1] == str(expected_input)
        assert all(x.split("=", 1)[1].startswith(str(root) + "/") for x in submit
                   if x.startswith(("--output=", "--error=")))
        write(root / phase / "submit.json", submit)
        env = read(normal / phase / "env.json")
        env["OUTPUT_ROOT"] = str(root / phase)
        write(root / phase / "env.json", env)
        (root / phase / "submit.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(submit) + "\n")
    write(root / "plan_review.json", dict(status="PASS_REUSE_VALIDATED_KERNELS_AND_REFERENCES",
          processes=count, rank_samples=count * ep * 2000, correctness_expected=new_counts,
          cases=cases, pairs=["aa", "ab", "bb"], orders=["abba", "baab"], new_kernel_builds=0))
    print((root / "plan_review.json").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--ep", type=int, choices=(2, 4, 8, 16, 32, 64), default=8)
    p.add_argument("--models", help="Comma-separated models present in the selected baseline EP plan")
    p.add_argument("--control-total", type=int,
                   help="Shared small-control token count; required outside EP8/16")
    p.add_argument("--totals", default="1024,2048,4096,8192", help="Large target token counts")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--cache-root", type=Path, help="Explicit prepared cache for unchanged native/kernel source")
    p.add_argument("--model-env", type=Path,
                   help="JSON model-to-environment overrides applied to every all-push arm")
    p.add_argument("--model-variants", type=Path,
                   help="JSON arm-to-model-to-environment overrides, e.g. explicit topology-derived worker counts")
    p.add_argument("--profile-env", type=Path,
                   help="Recorded all-push IKET environment; append separate first-pass large profiles")
    p.add_argument("--pull-profile-env", type=Path,
                   help="Recorded pull IKET environment; append separate first-pass large profiles")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--push-ctas", help="Comma-separated large-tile sender counts; eight is the control")
    selection.add_argument("--variants", type=Path,
                           help="JSON mapping ordered arm names to environment overrides, starting with control")
    args = p.parse_args()
    root, source = args.root.absolute(), args.source.absolute()
    totals = [int(t) for t in args.totals.split(",")]
    if len(set(totals)) != len(totals) or any(t <= 0 or t % args.ep for t in totals):
        p.error("--totals requires distinct positive multiples of EP")
    push_ctas = [int(n) for n in args.push_ctas.split(",")] if args.push_ctas else []
    if len(set(push_ctas)) != len(push_ctas) or any(n <= 0 or n == 8 for n in push_ctas):
        p.error("--push-ctas requires distinct positive candidates other than eight")
    template = json.loads((args.baseline / f"ep{args.ep}/plan.json").read_text())
    available_models = [group["model"] for group in template["groups"]]
    models = args.models.split(",") if args.models else available_models
    if (not models or len(set(available_models)) != len(available_models)
            or len(set(models)) != len(models) or set(models) - set(available_models)):
        p.error("--models requires distinct models present in the baseline EP plan")
    template["groups"] = [group for group in template["groups"] if group["model"] in models]
    if args.control_total is not None:
        if args.control_total <= 0 or args.control_total % args.ep:
            p.error("--control-total must be a positive multiple of EP")
        control_rows = {model: args.control_total // args.ep for model in models}
    elif args.ep not in (8, 16):
        p.error("EP2/4/32/64 requires an explicit --control-total")
    else:
        default_rows = {"qwen3": 32, "qwen35": 32} if args.ep == 8 else {"qwen3": 12, "qwen35": 57}
        if set(models) - set(default_rows):
            p.error("Unknown default small control; specify --control-total")
        control_rows = {model: default_rows[model] for model in models}
    baseline_source = template["groups"][0]["env"]["FASTAFD_SOURCE_REPO"]

    def source_arg(value):
        return value.replace(baseline_source, str(source))

    model_env = json.loads(args.model_env.read_text()) if args.model_env else {}
    if (not isinstance(model_env, dict) or set(model_env) - set(control_rows)
            or any(not isinstance(env, dict) or any(not key.startswith("DG_MEGAMOE_ALL_PUSH_") for key in env)
                   for env in model_env.values())):
        p.error("--model-env requires known models and DG_MEGAMOE_ALL_PUSH_ overrides")
    model_env = {model: {key: str(value) for key, value in env.items()} for model, env in model_env.items()}
    groups, manifests = [], []
    variants = {"control": {"DG_MEGAMOE_ALL_PUSH_WIDE": str(int(bool(push_ctas)))}}
    variants.update({f"push{n}": {"DG_MEGAMOE_ALL_PUSH_CTAS": str(n)} for n in push_ctas}
                    if push_ctas else {"wide": {}})
    if args.variants:
        variants = json.loads(args.variants.read_text())
    if (not variants or next(iter(variants)) != "control"
            or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name.endswith("_repeat")
                   for name in variants)):
        p.error("Variants require distinct arm names beginning with control")
    variants["pull"] = {"MINISGL_MEGAMOE_PAYLOAD_MODE": "pull"}
    defaults = dict(MINISGL_MEGAMOE_PAYLOAD_MODE="all_push", DG_MEGAMOE_ALL_PUSH_WIDE="1",
                    DG_MEGAMOE_ALL_PUSH_CTAS="8", DG_MEGAMOE_ALL_PUSH_WAVES="0",
                    DG_MEGAMOE_ALL_PUSH_FULL_TMA="0")
    defaults["DG_MEGAMOE_ALL_PUSH_PRIORITIZE_EXPERTS"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_ROTATE_COMPUTE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_COMPUTE_WARMUP_ROUNDS"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_COMPUTE_BUSY_SENDERS"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_UNIFIED_TMA"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_DEDICATED_DISPATCH"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_COMPACT_TILES"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_ACCUM_STAGES"] = "2"
    defaults["DG_MEGAMOE_ALL_PUSH_WAVE_BLOCKS_PER_SM"] = "2"
    defaults["DG_MEGAMOE_ALL_PUSH_COMPACT_STORE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_LINEAR_SOURCE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_CACHE_DESTINATIONS"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_INTERLEAVE_M"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_NARROW_SF"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_COALESCED_SF"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_PREFETCH_SEND"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_ROW_RELEASE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_BLOCK_M"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_MIDDLE_WINDOW_END"] = "2048"
    defaults["DG_MEGAMOE_ALL_PUSH_HALF_WARP"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_FIRST_WINDOW_END"] = "1024"
    defaults["DG_MEGAMOE_ALL_PUSH_STICKY_VALIDATION"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_FULL_ROW_COMBINE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_WEIGHT_CACHE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_WARP_COMBINE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_PREFIX_SWEEP"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_PRODUCER_PREFIX"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_SENDER_LDG"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_BALANCE_EPILOGUE"] = "0"
    defaults["DG_MEGAMOE_ALL_PUSH_METADATA_SINGLE_JOIN"] = "0"
    variants = {name: dict(defaults, **{k: str(v) for k, v in overrides.items()})
                for name, overrides in variants.items()}
    model_variants = read(args.model_variants) if args.model_variants else {}
    if (not isinstance(model_variants, dict) or set(model_variants) - (set(variants) - {"pull"})
            or any(not isinstance(overrides, dict) or set(overrides) - set(models)
                   or any(not isinstance(env, dict)
                          or any(not key.startswith("DG_MEGAMOE_ALL_PUSH_") for key in env)
                          for env in overrides.values())
                   for overrides in model_variants.values())):
        p.error("--model-variants requires known all-push arms/models and DG_MEGAMOE_ALL_PUSH_ settings")

    def settings_for(arm, model):
        settings = dict(variants[arm])
        if arm != "pull":
            settings.update(model_env.get(model, {}))
            settings.update({key: str(value) for key, value in
                             model_variants.get(arm, {}).get(model, {}).items()})
        return settings

    if any(int(settings_for(arm, model)["DG_MEGAMOE_ALL_PUSH_COMPACT_TILES"]) not in (0, 1)
           for arm in variants if arm != "pull"
           for model in control_rows):
        p.error("COMPACT_TILES supports 0 or 1; automatic M64 selection failed normal-routing validation")
    if args.profile_env and any(
            any(settings_for(arm, model)[key] != "0" for key in (
                "DG_MEGAMOE_ALL_PUSH_UNIFIED_TMA", "DG_MEGAMOE_ALL_PUSH_NARROW_SF"))
            for arm in variants if arm != "pull"
            for model in control_rows) and json.loads(args.profile_env.read_text()).get(
                "MEGAMOE_IKET_ROLE_CONTRACT") != "all_push_pipeline":
        p.error("Unified TMA/narrow SF profiling requires the all_push_pipeline role contract")
    if any(v["MINISGL_MEGAMOE_PAYLOAD_MODE"] != ("pull" if name == "pull" else "all_push")
           for name, v in variants.items()):
        p.error("Only the pull arm may select pull payloads")
    prefixes = list(variants)
    arms = prefixes + [name + "_repeat" for name in reversed(prefixes)]
    cache = args.cache_root.absolute() if args.cache_root else root / "cache"
    profile_env = json.loads(args.profile_env.read_text()) if args.profile_env else None
    pull_profile_env = json.loads(args.pull_profile_env.read_text()) if args.pull_profile_env else None
    root.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        for group in template["groups"]:
            model = group["model"]
            variant = settings_for(arm.removesuffix("_repeat"), model)
            env = {key: source_arg(value) for key, value in group["env"].items()}
            manifest = json.loads((Path(env["MEGAMOE_SWEEP_ROOT"]) / "manifest.json").read_text())
            control_total = control_rows[model] * args.ep
            control_case = f"ep{args.ep}_tokens{control_total}"
            c = manifest["contract"]
            existing = {case["global_tokens"]: case for case in manifest["cases"]
                        if case["ep"] == args.ep}
            manifest["cases"] = [existing.get(total, dict(
                case_id=f"ep{args.ep}_tokens{total}", ep=args.ep,
                rows_per_rank=total // args.ep, global_tokens=total,
                capture_contract=dict(bucket=c["bucket"])))
                for total in totals if total != control_total]
            manifest["cases"].insert(0, dict(case_id=control_case, ep=args.ep,
                rows_per_rank=control_rows[model], global_tokens=control_total,
                capture_contract=dict(bucket=c["bucket"])))
            for case in manifest["cases"]:
                # Both paths use the same capacity, including wider tile padding.
                expected = case["global_tokens"] * c["top_k"] / c["experts"]
                block = 16 if expected <= 8.5 else 32 if expected <= 32.5 else 96 if expected <= 64.5 else 128 if expected <= 96.5 else 192
                count = -(-case["global_tokens"] * c["top_k"] // c["experts"])
                blocks = {block}
                if expected > 16.5:
                    for name, candidate in variants.items():
                        if candidate["MINISGL_MEGAMOE_PAYLOAD_MODE"] != "all_push":
                            continue
                        settings = settings_for(name, model)
                        if not int(settings["DG_MEGAMOE_ALL_PUSH_WIDE"]):
                            continue
                        compact = int(settings["DG_MEGAMOE_ALL_PUSH_COMPACT_TILES"])
                        if compact and expected > 96.5:
                            blocks.add(min((192, 160, 128), key=lambda tile: (-(-count // tile) * tile, -tile)))
                        explicit_tile = int(settings["DG_MEGAMOE_ALL_PUSH_BLOCK_M"])
                        if explicit_tile == 64 or (expected > 32.5 and explicit_tile in (80, 96)):
                            blocks.add(explicit_tile)
                        elif explicit_tile and expected > 96.5:
                            if explicit_tile not in (128, 160, 192, 256):
                                raise ValueError(f"Unsupported large tile: {explicit_tile}")
                            blocks.add(explicit_tile)
                padded = max(-(-count // tile) * tile for tile in blocks) * (c["experts"] // args.ep)
                capacity = -(-(case["rows_per_rank"] + padded) // 384) * 384
                case["capture_contract"]["bucket"] = max(case["capture_contract"]["bucket"], capacity)
            destination = root / model / arm
            write(destination / "manifest.json", manifest)
            if arm == "control":
                manifests.append(str(destination / "manifest.json"))
            env.update(
                FASTAFD_SOURCE_REPO=str(source),
                FASTAFD_IKET_DEEPGEMM_CACHE_ROOT=str(cache / "native"),
                FASTAFD_IKET_TVM_CACHE_ROOT=str(cache / "tvm"),
                DG_JIT_CACHE_DIR=str(cache / "jit"),
                OUTPUT_ROOT=str(root), MEGAMOE_SWEEP_ROOT=str(destination),
                MEGAMOE_SWEEP_CASES=",".join(c["case_id"] for c in manifest["cases"]),
                MEGAMOE_BASELINE_ITERATIONS=str(args.iterations),
                MEGAMOE_RESULT_REFERENCE_DIR=str(root / "reference" / model),
                MEGAMOE_RESULT_MODE="record" if arm == "control" else "compare",
            )
            env.update(variant)
            # Pull retains its own producer arithmetic; graph replay remains exact.
            if variant["MINISGL_MEGAMOE_PAYLOAD_MODE"] == "pull":
                env["MEGAMOE_SMOKE_RESET_DISPATCH_PAYLOAD"] = "0"
                env["MEGAMOE_SMOKE_PREPARE_BARRIER_ONLY"] = "1"
                env.pop("MEGAMOE_RESULT_REFERENCE_DIR")
                env.pop("MEGAMOE_RESULT_MODE")
            groups.append(dict(model=model, arm=arm, control_case=control_case, env=env))
    ordered = []
    for small in (True, False):
        for group in groups:
            env = dict(group["env"])
            selected = [case for case in env["MEGAMOE_SWEEP_CASES"].split(",")
                        if (case == group["control_case"]) == small]
            if selected:
                env["MEGAMOE_SWEEP_CASES"] = ",".join(selected)
                ordered.append(dict(group, stage="small_control" if small else "large_target", env=env))
    if profile_env is not None or pull_profile_env is not None:
        for group in list(ordered):
            env = group["env"]
            selected_profile = (profile_env if env["MINISGL_MEGAMOE_PAYLOAD_MODE"] == "all_push"
                                else pull_profile_env)
            if (group["stage"] != "large_target" or group["arm"].endswith("_repeat")
                    or selected_profile is None):
                continue
            env = dict(env, **selected_profile)
            env["FASTAFD_IKET_CAPTURE_PHASE"] = "profile"
            if env["MINISGL_MEGAMOE_PAYLOAD_MODE"] == "all_push":
                env.update(MEGAMOE_RESULT_MODE="compare",
                           MEGAMOE_IKET_PUSHER_CTAS=env["DG_MEGAMOE_ALL_PUSH_CTAS"])
                env["MEGAMOE_IKET_MIDDLE_WINDOW_END"] = env["DG_MEGAMOE_ALL_PUSH_MIDDLE_WINDOW_END"]
                env["MEGAMOE_IKET_FIRST_WINDOW_END"] = env["DG_MEGAMOE_ALL_PUSH_FIRST_WINDOW_END"]
                if env.get("MEGAMOE_IKET_THREE_WINDOW_READY") == "1":
                    # Recorded profile settings may reference an older decoder.
                    env["MEGAMOE_IKET_WINDOW_DECODER_ROOT"] = str(source / "codex_scripts")
                if env.get("MEGAMOE_IKET_ROLE_CONTRACT") == "all_push_pipeline":
                    env["MEGAMOE_IKET_SF_WORK_WARPS"] = (
                        "4" if env["DG_MEGAMOE_ALL_PUSH_NARROW_SF"] == "0"
                        else env["DG_MEGAMOE_ALL_PUSH_NARROW_SF"])
                    limits = json.loads(env["FASTAFD_IKET_EVENT_REGISTER_LIMITS"])
                    if env["DG_MEGAMOE_ALL_PUSH_UNIFIED_TMA"] == "1":
                        limits.update({name: 40 for name in ("MegaIndicesReady", "MegaPayloadReady",
                            "MegaGatherIssued", "MegaWindowWaitStart", "MegaWindowReady")})
                    if env["DG_MEGAMOE_ALL_PUSH_DEDICATED_DISPATCH"] == "1":
                        limits.update(MegaMetadataStart=128, MegaMetadataAcquired=128)
                        limits.update({name: 168 for name in ("MegaComputeEnd", "MegaCombineStart",
                            "MegaCombineEnd", "MegaLocalReduceStart", "MegaLocalReduceEnd",
                            "MegaEpiL1Wait", "MegaEpiL1Ready", "MegaEpiL2Wait", "MegaEpiL2Ready")})
                    env["FASTAFD_IKET_EVENT_REGISTER_LIMITS"] = json.dumps(limits)
                # The analyzer derives per-specialization worker counts from the
                # compiled tile and validates this requested large-tile count.
                if "DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS" in env:
                    env["MEGAMOE_IKET_EXPERT_CTAS"] = env["DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS"]
            ordered.append(dict(group, stage="profile", env=env))
    write(root / "plan.json", dict(ep=args.ep, arm_prefixes=prefixes, groups=ordered))
    base_commands = [[source_arg(value) for value in command]
                     for command in json.loads((args.baseline / "cpu_prepare/commands.json").read_text())]
    commands = []
    compile_variants = []
    for arm, variant in variants.items():
        if (model_env or model_variants) and variant["MINISGL_MEGAMOE_PAYLOAD_MODE"] == "all_push":
            compile_variants.extend((f"{arm}/{Path(manifest).parent.parent.name}",
                                     settings_for(arm, Path(manifest).parent.parent.name),
                                     [manifest]) for manifest in manifests)
        else:
            compile_variants.append((arm, variant, manifests))
    for arm, variant, compile_manifests in compile_variants:
        command = base_commands[0]
        start = command.index(str(source / "codex_scripts/precompile_megamoe_sweep.py"))
        end = command.index("--source")
        cmd = command[:start + 1] + compile_manifests + command[end:]
        cmd = [v.replace(str(args.baseline), str(root)) for v in cmd]
        cmd = [v.replace(str(root / "cache"), str(cache)) for v in cmd]
        selected_profile = (profile_env if variant["MINISGL_MEGAMOE_PAYLOAD_MODE"] == "all_push"
                            else pull_profile_env)
        if selected_profile is not None:
            cmd = [v for v in cmd if v != "--ordinary-only"]
        cmd[1:1] = [f"{k}={v}" for k, v in variant.items()]
        cmd += ["--payload-mode", variant["MINISGL_MEGAMOE_PAYLOAD_MODE"],
                "--metadata-output", str(root / "preparation" / arm)]
        commands.append(cmd)
    commands.append([v.replace(str(args.baseline), str(root)).replace(str(root / "cache"), str(cache))
                     for v in base_commands[1]])
    commands.append(["python3", "-B", str(source / "codex_scripts/run_megamoe_sweep_group.py"), "validate", str(root / "plan.json")])
    write(root / "cpu_prepare/commands.json", commands)
    image = [[source_arg(value) for value in command]
             for command in json.loads((args.baseline / "cpu_prepare/image_commands.json").read_text())]
    write(root / "cpu_prepare/image_commands.json", [[v.replace(str(args.baseline), str(root)) for v in c] for c in image])
    # Analysis uses a CPU allocation after capture, never holds the GPU job.
    write(root / "analysis/commands.json", [["python3", "-B",
        str(source / "codex_scripts/analyze_megamoe_throughput.py"), str(root)]])
    write(root / "analysis/image_commands.json", [[
        v.replace(str(args.baseline / "cpu_prepare"), str(root / "analysis"))
        for v in c] for c in image])
    for phase, original in [("cpu_prepare", args.baseline / "cpu_prepare/submit.json"),
                            ("gpu", args.baseline / f"ep{args.ep}/submit.json"),
                            ("analysis", args.baseline / "cpu_prepare/submit.json")]:
        submit = [source_arg(value) for value in json.loads(original.read_text())]
        submit = [v.replace(str(args.baseline / f"ep{args.ep}"), str(root)).replace(str(args.baseline), str(root))
                  for v in submit if not v.startswith("--dependency=")]
        submit = ["--job-name=megamoe-throughput-" + phase if v.startswith("--job-name=") else v for v in submit]
        if phase == "gpu":
            submit = [v + " --shared-container" if v.startswith("--wrap=") else v for v in submit]
        if phase != "gpu":
            submit = ["--qos=cpu-short" if v.startswith("--qos=") else v for v in submit]
        if phase == "analysis":
            submit = [v.replace(str(root / "cpu_prepare"), str(root / "analysis")) for v in submit]
        write(root / phase / "submit.json", submit)
        write(root / phase / "env.json", dict(OUTPUT_ROOT=str(root / phase)))
        (root / phase / "submit.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(submit) + "\n")
    print(root, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["normal"]:
        prepare_normal(sys.argv[2:])
    elif sys.argv[1:2] == ["calibrate-sweep"]:
        prepare_sweep_calibration(sys.argv[2:])
    elif sys.argv[1:2] in (["run-calibration"], ["run-graphs"]):
        run_calibration(sys.argv[2:])
    elif sys.argv[1:2] == ["calibrate"]:
        prepare_calibration(sys.argv[2:])
    else:
        main()
