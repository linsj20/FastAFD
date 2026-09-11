#!/usr/bin/env python3
"""Explain FMHA/step-span differences from paired exported Nsight SQLite files."""

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def graphs(path):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        grouped = defaultdict(list)
        for row in db.execute(
            "SELECT k.start,k.end,k.streamId,k.correlationId,s.value AS name "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
            "ON k.demangledName=s.id WHERE k.graphId IS NOT NULL ORDER BY k.start"
        ):
            grouped[row["correlationId"]].append(dict(row))
    return sorted(grouped.values(), key=lambda rows: min(r["start"] for r in rows))


def select(rows, name):
    return [r for r in rows if name in r["name"]]


def span(rows):
    return min(r["start"] for r in rows), max(r["end"] for r in rows)


def duration(rows):
    return sum(r["end"] - r["start"] for r in rows)


def union_duration(rows):
    end = -1
    total = 0
    for r in sorted(rows, key=lambda r: r["start"]):
        total += max(0, r["end"] - max(r["start"], end))
        end = max(end, r["end"])
    return total


def averages(rows):
    return {k: mean(r[k] for r in rows) for k in rows[0] if k != "step"}


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary_gpu_work(model_path, agraphs, mgraphs, shift_low, shift_high, output):
    """Count GPU work and CPU submission lead in ordinary inter-graph intervals."""
    with sqlite3.connect(model_path) as db:
        db.row_factory = sqlite3.Row
        kernels = [dict(r) for r in db.execute(
            "SELECT k.start,k.end,k.correlationId,k.streamId,s.value AS name "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id"
        )]
        copies = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
        memsets = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET")]
        api = {r["correlationId"]: dict(r) for r in db.execute(
            "SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )}
    activity = sorted(kernels + copies + memsets, key=lambda r: r["start"])
    useful = [r for r in activity if not any(
        name in r.get("name", "") for name in ("wait_ready_kernel", "wait_turn_kernel")
    )]
    rows, initial = [], []
    for index in range(2, len(mgraphs)):
        previous_end, current_start = span(mgraphs[index - 1])[1], span(mgraphs[index])[0]
        current_api = api[mgraphs[index][0]["correlationId"]]
        row = dict(step=index + 1, interval_us=(current_start - previous_end) / 1e3)
        events, example = [], []
        for kind, items in (("kernel", kernels), ("copy", copies), ("memset", memsets)):
            selected = [r for r in items if r["end"] > previous_end and r["start"] < current_start]
            clipped = [dict(start=max(previous_end, r["start"]),
                            end=min(current_start, r["end"])) for r in selected]
            events.extend(clipped)
            row[kind + "_count"] = len(selected)
            row[kind + "_sum_us"] = duration(clipped) / 1e3
            if kind == "copy":
                row["copy_bytes"] = sum(r["bytes"] for r in selected)
                for copy_kind in (1, 2, 8):
                    row["copy_kind_" + str(copy_kind) + "_count"] = sum(
                        r["copyKind"] == copy_kind for r in selected)
            for r in selected:
                example.append(dict(kind=kind, start_us=(r["start"] - previous_end) / 1e3,
                                    duration_us=(r["end"] - r["start"]) / 1e3,
                                    name=r.get("name"), bytes=r.get("bytes"),
                                    copy_kind=r.get("copyKind"), stream=r["streamId"]))
        row["gpu_active_union_us"] = union_duration(events) / 1e3
        row["no_recorded_gpu_work_us"] = row["interval_us"] - row["gpu_active_union_us"]
        row["graph_api_return_lead_us"] = (previous_end - current_api["end"]) / 1e3
        rows.append(row)
        if index == 2:
            (output / "step3_boundary_events.json").write_text(
                json.dumps(sorted(example, key=lambda r: r["start_us"]), indent=2) + "\n")
        attention_start = span(agraphs[index])[0]
        first_fmha = select(agraphs[index], "fmhaSm")[0]["start"]
        relevant = [r for r in useful if r["end"] + shift_high > attention_start
                    and r["start"] + shift_low < first_fmha]
        shifts = {shift_low, shift_high}
        for r in relevant:
            for edge in (r["start"], r["end"]):
                for boundary in (attention_start, first_fmha):
                    if shift_low <= boundary - edge <= shift_high:
                        shifts.add(boundary - edge)
        values = [union_duration([
            dict(start=max(attention_start, r["start"] + shift),
                 end=min(first_fmha, r["end"] + shift))
            for r in relevant if r["end"] + shift > attention_start
            and r["start"] + shift < first_fmha
        ]) / 1e3 for shift in shifts]
        initial.append(dict(step=index + 1, wait_us=(first_fmha - attention_start) / 1e3,
                            non_wait_gpu_work_low_us=min(values),
                            non_wait_gpu_work_high_us=max(values)))
    write_csv(output / "boundary_gpu_work.csv", rows)
    write_csv(output / "initial_wait_gpu_work.csv", initial)
    return dict(intergraph_mean=averages(rows), initial_wait_mean=averages(initial),
                minimum_graph_api_return_lead_us=min(r["graph_api_return_lead_us"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attention", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("metric", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    agraphs, mgraphs = graphs(args.attention), graphs(args.model)
    metric = json.loads(args.metric.read_text())
    assert len(agraphs) == len(mgraphs) == 16, "Expected warmup plus 15 graph launches"
    assert metric["latency_decode_step_ids"] == list(range(2, 17))
    steps, gaps, lower, upper = [], [], [], []
    for step, (arows, mrows) in enumerate(zip(agraphs[1:], mgraphs[1:]), 2):
        astart, aend = span(arows)
        mstart, mend = span(mrows)
        attn = select(arows, "fmhaSm")
        apub = select(arows, "quantize_publish_o_fp8_kernel")
        awaiters = select(arows, "wait_ready_kernel")
        mpub = select(mrows, "publish_qkv_kernel")
        mwaiters = select(mrows, "wait_ready_kernel")
        moe = select(mrows, "mega_moe")
        head = select(mrows, "nvjet_")
        embedding = select(mrows, "::index_kernel<")
        assert len(attn) == len(apub) == len(awaiters) == len(mpub) == len(mwaiters)
        assert len(attn) % 2 == 0 and len(head) == 1 and len(embedding) == 2
        expected = metric["per_step_cuda_execution_critical_ms"][step - 2]
        assert abs(max(aend - astart, mend - mstart) / 1e6 - expected) < 1e-9
        gap_ns = [nxt["start"] - prev["end"] for prev, nxt in zip(attn, attn[1:])]
        row = dict(
            step=step,
            attention_graph_ms=(aend - astart) / 1e6,
            fmha_sum_ms=duration(attn) / 1e6,
            fmha_union_ms=union_duration(attn) / 1e6,
            mb0_fmha_mean_us=duration(attn[::2]) / len(attn[::2]) / 1e3,
            mb1_fmha_mean_us=duration(attn[1::2]) / len(attn[1::2]) / 1e3,
            initial_wait_ms=(attn[0]["start"] - astart) / 1e6,
            between_fmha_net_ms=sum(gap_ns) / 1e6,
            between_fmha_positive_ms=sum(max(0, gap) for gap in gap_ns) / 1e6,
            final_publication_tail_ms=(aend - attn[-1]["end"]) / 1e6,
            publication_sum_ms=duration(apub) / 1e6,
            model_graph_ms=(mend - mstart) / 1e6,
            input_embedding_ms=duration(embedding) / 1e6,
            model_bootstrap_first_q_ready_ms=(mpub[0]["end"] - mstart) / 1e6,
            output_head_ms=duration(head) / 1e6,
            output_after_last_moe_ms=(mend - max(r["end"] for r in moe)) / 1e6,
            model_between_graph_ms=(mstart - span(mgraphs[step - 2])[1]) / 1e6,
        )
        assert abs(row["attention_graph_ms"] - sum(row[k] for k in (
            "fmha_sum_ms", "initial_wait_ms", "between_fmha_net_ms",
            "final_publication_tail_ms"
        ))) < 1e-9
        steps.append(row)
        for index, (prev, nxt) in enumerate(zip(attn, attn[1:])):
            waiter = max((r for r in awaiters if r["streamId"] == nxt["streamId"]
                          and r["start"] < nxt["start"]), key=lambda r: r["start"])
            publication = min((r for r in apub if r["streamId"] == prev["streamId"]
                               and r["start"] > prev["start"]), key=lambda r: r["start"])
            gaps.append(dict(
                step=step, layer_zero_based=index // 2,
                direction="mb0_to_mb1" if index % 2 == 0 else "mb1_to_mb0",
                previous_stream=prev["streamId"], next_stream=nxt["streamId"],
                gap_us=(nxt["start"] - prev["end"]) / 1e3,
                q_wait_end_relative_us=(waiter["end"] - prev["end"]) / 1e3,
                q_wait_to_fmha_us=(nxt["start"] - waiter["end"]) / 1e3,
                same_mb_roundtrip_to_q_ready_us=(
                    (waiter["end"] - attn[index - 1]["end"]) / 1e3 if index else None
                ),
                previous_publish_start_us=(publication["start"] - prev["end"]) / 1e3,
                previous_publish_end_us=(publication["end"] - prev["end"]) / 1e3,
            ))
        # Bound a constant model->attention trace shift using both message directions.
        # Publication starts before its peer's ready waiter can finish. This is a
        # causal envelope, not an assertion that independent host clocks align.
        lower.extend(a["start"] - m["end"] for a, m in zip(apub, mwaiters))
        upper.extend(a["end"] - m["start"] for a, m in zip(awaiters, mpub))
    shift_low, shift_high = max(lower) - 2000, min(upper) + 2000
    assert shift_low <= shift_high, "No consistent cross-trace causal envelope"
    boundary, producer = [], []
    for step in range(3, 17):
        arows, mrows, previous = agraphs[step - 1], mgraphs[step - 1], mgraphs[step - 2]
        astart, _ = span(arows)
        first_head = select(previous, "nvjet_")[0]
        boundary.append({
            "step": step,
            **{name + suffix: (time - astart + offset) / 1e3
               for name, time in [("previous_head_start", first_head["start"]),
                                  ("previous_model_end", span(previous)[1]),
                                  ("current_model_start", span(mrows)[0])]
               for suffix, offset in [("_low_us", shift_low), ("_high_us", shift_high)]}
        })
    for step in range(2, 17):
        arows, mrows = agraphs[step - 1], mgraphs[step - 1]
        attn, mpub = select(arows, "fmhaSm"), select(mrows, "publish_qkv_kernel")
        for index in range(2, len(attn), 2):
            publication = mpub[index + 1]
            previous_moe = max((r for r in select(mrows, "mega_moe")
                                if r["streamId"] == publication["streamId"]
                                and r["start"] < publication["start"]),
                               key=lambda r: r["start"])
            producer.append(dict(
                step=step, layer_zero_based=index // 2,
                moe_end_low_us=(previous_moe["end"] - attn[index]["end"] + shift_low) / 1e3,
                moe_end_high_us=(previous_moe["end"] - attn[index]["end"] + shift_high) / 1e3,
                qkv_publication_us=(publication["end"] - publication["start"]) / 1e3,
            ))
    directions = {}
    for direction in ("mb0_to_mb1", "mb1_to_mb0"):
        rows = [r for r in gaps if r["direction"] == direction]
        late = [r for r in rows if r["q_wait_end_relative_us"] > 0]
        directions[direction] = dict(
            transitions=len(rows), q_wait_finishes_after_previous_attention=len(late),
            mean_gap_us=mean(r["gap_us"] for r in rows),
            median_gap_us=median(r["gap_us"] for r in rows),
            mean_same_mb_roundtrip_to_q_ready_us=mean(
                r["same_mb_roundtrip_to_q_ready_us"] for r in rows
                if r["same_mb_roundtrip_to_q_ready_us"] is not None),
            mean_q_wait_to_fmha_when_q_late_us=(mean(r["q_wait_to_fmha_us"] for r in late)
                                             if late else None),
            next_attention_before_previous_publication_ends=sum(
                r["gap_us"] < r["previous_publish_end_us"] for r in rows),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("per_step", steps), ("attention_transitions", gaps),
                       ("boundary_envelope", boundary), ("producer_envelope", producer)]:
        write_csv(args.output / (name + ".csv"), rows)
    summary = dict(
        case_id=metric["case_id"], reports=metric["profiled_roles"],
        all_15_steps_mean=averages(steps), steps_3_to_16_mean=averages(steps[1:]),
        transitions=directions,
        model_to_attention_shift_envelope_us=[shift_low / 1e3, shift_high / 1e3],
        boundary_mean_envelope=averages(boundary),
        producer_mean_envelope=averages(producer),
        producer_moe_certainly_finished_before_mb0_attention_ends=sum(
            r["moe_end_high_us"] < 0 for r in producer),
        producer_transitions=len(producer),
        caveats=[
            "The reported step metric is maximum representative GPU graph span, not host request latency.",
            "Kernel-end/start timestamps show sub-microsecond cross-stream overlaps; net gaps preserve the sum identity.",
            "Cross-trace alignment uses ordered Q/O handshakes and a 2us endpoint margin, not raw host-clock subtraction.",
            "Steps 3-16 are diagnostic only; all original 15 accepted samples remain in the performance result.",
        ],
    )
    summary["boundary_gpu_work"] = boundary_gpu_work(
        args.model, agraphs, mgraphs, shift_low, shift_high, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
