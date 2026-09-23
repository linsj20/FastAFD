"""Original installed TRT-LLM DeepGEMM: workload and six-point trace collection.

The ordinary process uses the image verbatim. Only profile processes use an
include-path overlay containing the same kernel with guarded IKET markers.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def workload(root, case_id, mode):
    import datetime
    import torch
    import torch.distributed as dist
    from tensorrt_llm._torch.modules.fused_moe.quantization import _import_deep_gemm
    dg = _import_deep_gemm()
    from tensorrt_llm.deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

    manifest = json.loads((root / "manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["case_id"] == case_id)
    contract = dict(manifest["contract"], **case.get("capture_contract", {}))
    iterations = manifest.get("ordinary_iterations", 3) if mode == "normal" else 3
    warmups = manifest.get("ordinary_warmups", 1)
    rank, local_rank = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_LOCALID"])
    ep = case["ep"]
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=ep,
                            timeout=datetime.timedelta(minutes=10), device_id=torch.device("cuda", local_rank))
    torch.manual_seed(1234 + rank)
    h, inter, experts, topk = (contract[k] for k in ("hidden", "intermediate", "experts", "top_k"))
    dg.set_num_sms(contract["num_sms"])
    buffer = dg.get_symm_buffer_for_mega_moe(dist.group.WORLD, experts,
        contract["bucket"], topk, h, inter, mma_type="fp8xfp4")

    def weights(n, k):
        groups = experts // ep
        w = torch.empty((groups, n, k // 2), dtype=torch.int8, device="cuda")
        sf = torch.empty((groups, n, k // 32), dtype=torch.float32, device="cuda")
        for i in range(groups):
            x = (torch.randn((n, k), device="cuda") * .125).to(torch.float8_e4m3fn).to(torch.bfloat16)
            w[i], sf[i] = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
        return w, dg.transform_sf_into_required_layout(sf, n, k, (1, 32), groups)

    weight_file = root / "cache" / f"ep{ep}" / f"weights-rank{rank}.pt"
    if weight_file.exists():
        transformed = torch.load(weight_file, map_location="cuda", weights_only=True)
    else:
        assert mode == "normal", "profile requires ordinary weights"
        transformed = dg.transform_weights_for_mega_moe(weights(2 * inter, h), weights(h, inter))
        torch.save(tuple(tuple(t.cpu() for t in pair) for pair in transformed), weight_file)
    # Separate input RNG from the one-time weight creation/cache path.
    torch.manual_seed(4321 + rank)
    total = case["global_tokens"]
    count = total // ep + (rank < total % ep)
    prefix = rank * (total // ep) + min(rank, total % ep)
    x = torch.randn((max(count, 1), h), device="cuda", dtype=torch.bfloat16) * .125
    xq, xsf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    buffer.x[:count].copy_(xq[:count])
    buffer.x_sf[:count].copy_(xsf[:count])
    seq = (torch.arange(count * topk, device="cuda") + prefix * topk) % experts
    ids = ((seq % ep) * (experts // ep) + seq // ep).reshape(count, topk)
    buffer.topk_idx[:count].copy_(ids)
    buffer.topk_weights[:count].fill_(1 / topk)
    counts = torch.bincount(ids.flatten(), minlength=experts)
    dist.all_reduce(counts)
    assert int(counts.sum()) == total * topk and int(counts.max() - counts.min()) <= 1
    y = torch.empty((count, h), dtype=torch.bfloat16, device="cuda")
    def run():
        dg.fp8_fp4_mega_moe(y, *transformed, buffer)

    run()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(y).all()), "nonfinite original output"
    case_root = root / "cases" / case_id
    reference = case_root / f"reference-rank{rank}.pt"
    if mode == "normal":
        torch.save(y.cpu(), reference)
    else:
        expected = torch.load(reference, map_location="cuda", weights_only=True)
        assert torch.equal(y.view(torch.int16), expected.view(torch.int16)), "instrumented output differs"
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    if mode == "normal" and rank == 0:
        torch.cuda.profiler.start()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    if mode == "normal" and rank == 0:
        torch.cuda.profiler.stop()
    expected = torch.load(reference, map_location="cuda", weights_only=True)
    assert torch.equal(y.view(torch.int16), expected.view(torch.int16)), "replay output differs"
    elapsed = torch.tensor([start.elapsed_time(end) * 1000 / iterations], dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        write(case_root / (mode + ".json"), dict(status="ok", case=case,
              kernel="official TRT-LLM DeepGEMM FP8xFP4", prepared_routes=False,
              correctness="all ranks finite; replay byte exact against first ordinary output",
              graph_mean_max_rank_us=float(elapsed[0]), iterations=iterations))
        print(f"CASE_{mode.upper()}_OK {case_id}", flush=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


def analyze(root, ep):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("ordinary_only"):
        for case in (c for c in manifest["cases"] if c["ep"] == ep):
            path = root / "cases" / case["case_id"]
            result = json.loads((path / "normal.json").read_text())
            assert result["status"] == "ok" and result["case"] == case
            assert result["iterations"] == manifest["ordinary_iterations"]
            report = path / "nsys/megamoe.nsys-rep"
            out = subprocess.check_output(["nsys", "stats", "--force-export=true",
                "--report", "cuda_gpu_kern_sum", "--format", "csv", str(report)], text=True)
            (path / "nsys/kernel_summary.csv").write_text(out)
        write(root / f"ep{ep}/complete.json", dict(ep=ep, status="ok", ordinary_only=True))
        return
    samples = []
    for case in (c for c in manifest["cases"] if c["ep"] == ep):
        path = root / "cases" / case["case_id"]
        for mode in ("normal", "profile"):
            assert json.loads((path / (mode + ".json")).read_text())["status"] == "ok"
        traces = []
        for file in (path / "iket").rglob("*.json"):
            data = json.loads(file.read_text())
            if isinstance(data, dict) and "launches" in data and "locationTable" in data:
                traces.append((file, data))
        assert len(traces) == 1, f"expected one trace for {case['case_id']}"
        trace = traces[0][1]
        def indexed(table, index):
            return table[index] if isinstance(table, list) else table[str(index)]
        launches = list(trace["launches"])
        graphs = trace.get("graphLaunches", {})
        assert isinstance(graphs, dict), "unexpected graph trace schema"
        for key in sorted(graphs, key=lambda key: int(key.rsplit(":", 1)[-1])):
            group = graphs[key]
            launches.extend(group.get("launches", []) if isinstance(group, dict) else group)
        mega = [l for l in launches if any(indexed(trace["stringTable"], m["markerNameIdx"]) == "MegaDispatchStart" for m in l.get("markers", []))]
        assert len(mega) >= 3, "missing measured graph launches"
        case_samples = []
        for ordinal, launch in enumerate(mega[-3:]):
            lives = launch["warpLifetimes"]
            base, end = min(int(w["startTs"]) for w in lives), max(int(w["endTs"]) for w in lives)
            row = dict(case_id=case["case_id"], ep=ep, global_tokens=case["global_tokens"], sample=ordinal,
                       kernel_envelope_us=(end-base)/1000)
            for phase in ("Dispatch", "Compute", "Combine"):
                for bound, reducer in (("Start", min), ("End", max)):
                    name = "Mega" + phase + bound
                    times = [int(m["timestamp"]) for m in launch["markers"] if indexed(trace["stringTable"], m["markerNameIdx"]) == name]
                    assert times and base <= min(times) <= max(times) <= end, name
                    row[f"{phase.lower()}_{bound.lower()}_us"] = (reducer(times)-base)/1000
                row[phase.lower()+"_span_us"] = row[phase.lower()+"_end_us"] - row[phase.lower()+"_start_us"]
                assert row[phase.lower()+"_span_us"] >= 0
            row["exposed_dispatch_us"] = row["compute_start_us"]-row["dispatch_start_us"]
            row["exposed_combine_us"] = max(0, row["combine_end_us"]-row["compute_end_us"])
            case_samples.append(row)
        # Export on a CPU allocation, after the GPU allocation has exited.
        report = path / "nsys" / "megamoe.nsys-rep"
        out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(report)], check=True, capture_output=True, text=True).stdout
        (path / "nsys" / "kernel_summary.csv").write_text(out)
        start = out.index('"Time (%)"') if '"Time (%)"' in out else out.index("Time (%),")
        kernels = [r for r in csv.DictReader(out[start:].splitlines()) if "sm100_fp8_fp4_mega_moe_impl" in r.get("Name", "")]
        assert len(kernels) == 1 and int(kernels[0]["Instances"]) == 3, kernels
        nsys = {k: float(kernels[0][k]) / 1000 for k in ("Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)")}
        write(path / "decomposition.json", dict(case=case, samples=case_samples, nsys_kernel_us=nsys,
              compute_start_semantics="operands_ready_proxy", dispatch_includes_original_routing=True,
              phase_spans_overlap=True, trace=str(traces[0][0])))
        samples.extend(case_samples)
    with (root / f"ep{ep}" / "decomposition.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    write(root / f"ep{ep}" / "complete.json", dict(ep=ep, cases=len(samples)//3, samples=len(samples), status="ok"))


def collect(root):
    """Audit compact captures and export the shared comparison-report schema.

    Preserve raw captures. Derive exposed communication from interval overlap,
    keeping dispatch-to-compute gaps separate from the dispatch phase itself.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("ordinary_only"):
        records = []
        iterations = manifest["ordinary_iterations"]
        for case in manifest["cases"]:
            folder = root / "cases" / case["case_id"]
            result = json.loads((folder / "normal.json").read_text())
            assert result["status"] == "ok" and result["case"] == case
            assert result["iterations"] == iterations and result["prepared_routes"] is False
            assert not (folder / "profile.json").exists()
            stats = folder / "nsys/kernel_summary.csv"
            lines = stats.read_text().splitlines()
            start = next(i for i, line in enumerate(lines) if "Med (ns)" in line and "Instances" in line)
            kernels = [r for r in csv.DictReader(lines[start:]) if "sm100_fp8_fp4_mega_moe_impl" in r["Name"]]
            assert len(kernels) == 1 and int(kernels[0]["Instances"]) == iterations
            value = float(kernels[0]["Med (ns)"]) / 1000
            assert math.isfinite(value) and value > 0
            records.append(dict(case_id=case["case_id"], ep=case["ep"],
                global_tokens=case["global_tokens"], rows_per_rank=case["rows_per_rank"],
                nsys_kernel_median_us=value, ordinary_samples=iterations,
                artifact=str(stats.relative_to(root))))
        with (root / "ordinary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        write(root / "ordinary_audit.json", dict(complete=True, missing=[],
            artifact_contracts_checked=len(records), ordinary_samples=iterations,
            rank=0, metric="nsys_kernel_median_us", instrumented=False,
            sample_policy="Unfiltered ordinary kernel medians", backend="official TRT-LLM image"))
        return
    sequence = json.loads((root / "sequence.json").read_text())
    assert sequence["status"] == "complete"
    eps = manifest["contract"]["eps"]
    assert [(j["ep"], j["mode"]) for j in sequence["jobs"]] == [(ep, mode) for ep in eps for mode in ("gpu", "analyze")]
    for i, job in enumerate(sequence["jobs"]):
        assert job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
        if i:
            assert sequence["jobs"][i-1]["ended"] <= job["submitted"]
    for ep in eps:
        complete = json.loads((root / f"ep{ep}" / "complete.json").read_text())
        count = sum(c["ep"] == ep for c in manifest["cases"])
        assert complete == dict(ep=ep, cases=count, samples=3*count, status="ok")
        racks = {node.rsplit("-T", 1)[0] for node in (root / f"ep{ep}" / "nodes.txt").read_text().splitlines()}
        assert len(racks) == 1, f"EP{ep} spans racks"

    records, all_samples = [], []
    for case in manifest["cases"]:
        folder = root / "cases" / case["case_id"]
        result = json.loads((folder / "decomposition.json").read_text())
        assert result["case"] == case
        assert result["compute_start_semantics"] == "operands_ready_proxy"
        assert result["dispatch_includes_original_routing"] and result["phase_spans_overlap"]
        for mode in ("normal", "profile"):
            report = json.loads((folder / (mode + ".json")).read_text())
            assert report["case"] == case and report["status"] == "ok" and report["iterations"] == 3
            assert report["prepared_routes"] is False
        text = (folder / "nsys/kernel_summary.csv").read_text()
        start = text.index('"Time (%)"') if '"Time (%)"' in text else text.index("Time (%),")
        kernels = [r for r in csv.DictReader(text[start:].splitlines()) if "sm100_fp8_fp4_mega_moe_impl" in r["Name"]]
        assert len(kernels) == 1 and int(kernels[0]["Instances"]) == 3
        for name, value in result["nsys_kernel_us"].items():
            assert math.isfinite(value) and value > 0
            assert math.isclose(float(kernels[0][name])/1000, value, abs_tol=1e-9)
        assert [s["sample"] for s in result["samples"]] == [0, 1, 2]
        samples = []
        for raw in result["samples"]:
            assert (raw["case_id"], raw["ep"], raw["global_tokens"]) == (case["case_id"], case["ep"], case["global_tokens"])
            ds, de, cs, ce, bs, be, end = (raw[k] for k in (
                "dispatch_start_us", "dispatch_end_us", "compute_start_us", "compute_end_us",
                "combine_start_us", "combine_end_us", "kernel_envelope_us"))
            assert all(math.isfinite(x) for x in (ds, de, cs, ce, bs, be, end))
            assert 0 <= ds <= cs <= ce <= be <= end and ds <= de <= ce and cs <= bs <= be
            s = dict(raw, measured_ordinal=raw["sample"], initial_interval_us=ds,
                     pre_compute_dispatch_exposure_us=min(de, cs)-ds,
                     dispatch_to_compute_gap_us=max(0, cs-de),
                     post_compute_combine_exposure_us=be-max(bs, ce),
                     compute_to_combine_gap_us=max(0, bs-ce), post_combine_tail_us=end-be)
            for phase in ("dispatch", "compute", "combine"):
                assert math.isclose(s[phase+"_span_us"], s[phase+"_end_us"]-s[phase+"_start_us"], abs_tol=1e-9)
            for phase in ("dispatch", "combine"):
                s[phase+"_compute_overlap_us"] = max(0, min(s[phase+"_end_us"], ce)-max(s[phase+"_start_us"], cs))
            closure = sum(s[k] for k in ("initial_interval_us", "pre_compute_dispatch_exposure_us",
                "dispatch_to_compute_gap_us", "compute_span_us", "compute_to_combine_gap_us",
                "post_compute_combine_exposure_us", "post_combine_tail_us"))
            assert math.isclose(closure, end, abs_tol=1e-8)
            samples.append(s)
        row = dict(case_id=case["case_id"], ep=case["ep"], rows_per_rank=case["rows_per_rank"],
                   global_tokens=case["global_tokens"], equivalent_input_count=len(case["equivalent_inputs"]),
                   artifact=str((folder / "decomposition.json").relative_to(root)))
        for name, key in (("min", "Min (ns)"), ("median", "Med (ns)"), ("max", "Max (ns)")):
            row[f"nsys_kernel_{name}_us"] = result["nsys_kernel_us"][key]
        for metric in ("dispatch_span_us", "compute_span_us", "combine_span_us", "kernel_envelope_us",
                       "pre_compute_dispatch_exposure_us", "post_compute_combine_exposure_us",
                       "initial_interval_us", "dispatch_to_compute_gap_us", "compute_to_combine_gap_us",
                       "post_combine_tail_us", "dispatch_compute_overlap_us", "combine_compute_overlap_us"):
            values = [s[metric] for s in samples]
            for stat, function in (("min", min), ("median", median), ("max", max)):
                row[f"{metric}_{stat}"] = function(values)
        records.append(row)
        all_samples.extend(samples)
    assert len({r["case_id"] for r in records}) == len(manifest["cases"])
    for name, rows in (("decomposition.csv", records), ("all_samples.csv", all_samples)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = dict(complete=True, artifact_contracts_checked=len(records), expected_cases=len(records),
                 samples_checked=len(all_samples), equivalent_inputs_checked=sum(r["equivalent_input_count"] for r in records),
                 missing=[], source_format="trtllm-original", phase_samples=3, baseline_iterations=3,
                 checks=["serial completed GPU and CPU jobs", "one rack per EP", "exact input contracts",
                         "ordinary and marked success", "three ordinary kernel samples and unit conversion",
                         "three retained IKET samples", "six finite ordered boundaries", "per-sample phase closure"])
    write(root / "audit_summary.json", audit)
    print(json.dumps(audit))


def sequence(root):
    """Durable serial submission; fail closed on capture or analysis failure."""
    script = root / "control" / "trtllm_megamoe_sweep.sbatch"
    manifest = json.loads((root / "manifest.json").read_text())
    state = dict(status="running", jobs=[])
    state_file = root / "sequence.json"
    assert (root / "preparation" / "complete").exists()
    assert not state_file.exists(), "sequence already submitted; inspect recorded jobs"
    write(state_file, state)
    try:
        for ep in manifest["contract"]["eps"]:
            for mode in ("gpu", "analyze"):
                nodes = (ep + 3) // 4 if mode == "gpu" else 1
                args = ["sbatch", "--parsable", f"--nodes={nodes}",
                        f"--job-name=trt-original-q3-ep{ep}-{mode}",
                        f"--output={root}/ep{ep}/{mode}-%j.out", f"--error={root}/ep{ep}/{mode}-%j.err"]
                if mode == "gpu":
                    args += ["--qos=short", "--time=02:00:00", f"--segment={nodes}",
                             f"--ntasks={ep}", f"--ntasks-per-node={min(ep,4)}", "--cpus-per-task=8", "--gres=gpu:4", "--exclusive"]
                else:
                    args += ["--partition=cpu", "--qos=cpu-normal", "--time=01:00:00", "--ntasks=1", "--cpus-per-task=16", "--mem=64G"]
                job = subprocess.check_output(args + [str(script), mode, str(root), str(ep)], text=True).strip().split(";")[0]
                record = dict(ep=ep, mode=mode, job_id=job, state="SUBMITTED", submitted=time.time())
                state["jobs"].append(record)
                write(state_file, state)
                print(json.dumps(record), flush=True)
                while True:
                    time.sleep(300)
                    result = subprocess.check_output(["sacct", "-n", "-X", "-j", job, "-o", "JobIDRaw,State,ExitCode", "-P"], text=True)
                    rows = [line.split("|") for line in result.splitlines() if line.startswith(job+"|")]
                    if not rows:
                        continue
                    status, code = rows[0][1:3]
                    if status in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"):
                        continue
                    record.update(state=status, exit_code=code, ended=time.time())
                    write(state_file, state)
                    assert status == "COMPLETED" and code == "0:0", f"job {job}: {status} {code}"
                    break
            assert (root / f"ep{ep}" / "complete.json").exists(), f"EP{ep} analysis incomplete"
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        write(state_file, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("workload")
    w.add_argument("root", type=Path)
    w.add_argument("case_id")
    w.add_argument("mode", choices=("normal", "profile"))
    a = sub.add_parser("analyze")
    a.add_argument("root", type=Path)
    a.add_argument("ep", type=int)
    s = sub.add_parser("sequence")
    s.add_argument("root", type=Path)
    c = sub.add_parser("collect")
    c.add_argument("root", type=Path)
    args = p.parse_args()
    if args.command == "workload":
        workload(args.root, args.case_id, args.mode)
    elif args.command == "analyze":
        analyze(args.root, args.ep)
    elif args.command == "collect":
        collect(args.root)
    else:
        sequence(args.root)
