#!/usr/bin/env python3
"""Compare direct MegaMoE latency traces, preserving samples and marker semantics."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def location(marker):
    loc = marker["location"]
    return tuple(loc["ctaId"]), loc["warpId"]


def bounds(values):
    return dict(first=min(values), median=median(values), last=max(values))


def startup_markers(markers, cta, origin):
    """Expose startup work on a specific pair, including dispatch participants."""
    pair_x = cta[0] & ~1
    events = {}
    names = {
        "GenphaseStart", "GenphaseIndexScatter", "GenphaseIndexReady",
        "GenphasePayloadSendStart", "GenphasePayloadDrained",
        "GenphasePayloadSenderDone", "GenphaseTailPublishStart", "GenphasePayloadStoresIssued",
        "GenphasePayloadValidateStart", "GenphaseRemainderValidateStart",
        "GenphaseFullPayloadIssue", "GenphaseEnd", "MegaDispatchEnd",
        "GenphaseSfStoresIssued", "GenphaseMetadataStoresIssued",
        "GenphaseSfReadyIssue", "GenphaseFirstPayloadIssue",
        "MegaMetadataAcquired", "MegaIndicesReady", "MegaPayloadReady",
        "MegaGatherSfStart", "MegaGatherSfReady", "MegaGatherIssued",
        "MegaTileReady", "MegaOperandsWait", "MegaMmaIssue", "MegaComputeStart",
        "MegaFirstFc1AccObserved", "MegaFirstFc1EpiObserved",
        "MegaRemainderWaitStart", "MegaRemainderReady",
    }
    for marker in markers:
        x, y, z = location(marker)[0]
        if ((x & ~1, y, z) == (pair_x, cta[1], cta[2])
                and marker["name"] in names):
            events.setdefault(marker["name"], []).append(marker["time_us"] - origin)
    return {name: bounds(values) for name, values in events.items()}


def sender_publication_intervals(markers, origin):
    """Pair each sender's markers; avoid subtracting unrelated global extrema."""
    events = {name: {} for name in (
        "GenphasePayloadSendStart", "GenphasePayloadDrained",
        "GenphaseTailPublishStart", "GenphaseEnd",
    )}
    for marker in markers:
        if marker["name"] in events:
            events[marker["name"]].setdefault(location(marker), []).append(marker["time_us"])
    published = events["GenphaseTailPublishStart"]
    if not published:
        return []  # Older captures do not have this measurement boundary.
    for name in ("GenphasePayloadSendStart", "GenphasePayloadDrained"):
        if events[name].keys() != published.keys():
            raise ValueError(f"Unpaired sender publication markers: {name}")
    if not published.keys() <= events["GenphaseEnd"].keys():
        raise ValueError("Sender publication lacks a retirement marker")
    result = []
    for key in sorted(published):
        start = min(events["GenphasePayloadSendStart"][key])
        drained = min(events["GenphasePayloadDrained"][key])
        publication = min(published[key])
        end = max(events["GenphaseEnd"][key])
        if not start <= drained <= publication <= end:
            raise ValueError(f"Reversed sender publication boundaries: {key}")
        result.append(dict(
            cta=list(key[0]), warp=key[1],
            send_start_from_metadata_us=start - origin,
            drained_from_metadata_us=drained - origin,
            tail_publish_start_from_metadata_us=publication - origin,
            retirement_from_metadata_us=end - origin,
            send_start_to_drained_us=drained - start,
            drained_to_tail_publish_start_us=publication - drained,
            drained_to_retirement_us=end - drained,
            tail_publish_start_to_retirement_us=end - publication,
            raw_marker_counts={name: len(points[key]) for name, points in events.items()},
        ))
    return result


def sender_store_retirement(markers, origin):
    issued = [m for m in markers if m["name"] == "GenphasePayloadStoresIssued"]
    result = []
    for m in issued:
        key = location(m)
        ends = [e["time_us"] for e in markers
                if e["name"] == "GenphaseEnd" and location(e) == key]
        if not ends or max(ends) < m["time_us"]:
            raise ValueError("Missing/reversed generic-store retirement")
        result.append(dict(cta=list(key[0]), warp=key[1],
                           stores_issued_from_metadata_us=m["time_us"] - origin,
                           retirement_from_metadata_us=max(ends) - origin,
                           stores_issued_to_retirement_us=max(ends) - m["time_us"]))
    return result


def local_readiness_publications(markers, origin, metadata_materialized=True):
    """Pair local work with publication issue, without claiming completion."""
    names = ("GenphaseSfStoresIssued", "GenphaseMetadataStoresIssued",
             "GenphaseSfReadyIssue", "GenphasePayloadValidateStart",
             "GenphaseFirstPayloadIssue")
    events = {name: {} for name in names}
    for marker in markers:
        if marker["name"] in events:
            cta, warp = location(marker)
            events[marker["name"]].setdefault(cta, {}).setdefault(warp, []).append(
                marker["time_us"] - origin)
    sf_issues = events["GenphaseSfReadyIssue"]
    sf_stores = events["GenphaseSfStoresIssued"]
    metadata_stores = events["GenphaseMetadataStoresIssued"]
    if sf_issues.keys() != sf_stores.keys():
        raise ValueError("Incomplete local SF publication markers")
    if metadata_materialized:
        if sf_issues.keys() != metadata_stores.keys():
            raise ValueError("Incomplete local metadata store markers")
    elif metadata_stores:
        raise ValueError("Unexpected local metadata stores for direct producer reads")
    sf = []
    for cta in sorted(sf_issues):
        issue = min(t for times in sf_issues[cta].values() for t in times)
        stores = {warp: bounds(times) for warp, times in sf_stores[cta].items()}
        last_store = max(value["last"] for value in stores.values())
        metadata = (bounds([t for times in metadata_stores[cta].values() for t in times])
                    if metadata_materialized else None)
        if last_store > issue:
            raise ValueError(f"SF publication precedes worker store markers: {cta}")
        sf.append(dict(
            cta=list(cta), sf_store_phase_ends_by_warp=stores,
            metadata_stores_from_metadata_us=metadata,
            sf_publication_issue_from_metadata_us=issue,
            last_sf_phase_end_to_publication_issue_us=issue - last_store,
            metadata_stores_minus_sf_publication_issue_us=(
                metadata["last"] - issue if metadata is not None else None),
        ))
    activation = []
    for cta, warps in sorted(events["GenphaseFirstPayloadIssue"].items()):
        if cta not in events["GenphasePayloadValidateStart"]:
            raise ValueError(f"First A publication lacks validation start: {cta}")
        start = min(t for times in events["GenphasePayloadValidateStart"][cta].values() for t in times)
        issue = min(t for times in warps.values() for t in times)
        starts = events["GenphasePayloadValidateStart"][cta]
        for warp, times in warps.items():
            if warp in starts and min(times) < min(starts[warp]):
                raise ValueError(f"First A publication precedes its warp's validation: {cta}, {warp}")
        activation.append(dict(
            cta=list(cta), validation_start_from_metadata_us=start,
            first_A_publication_issue_from_metadata_us=issue,
            validation_start_to_first_A_publication_issue_us=issue - start,
        ))
        if len(warps) > 1:
            # A per-warp publication certifies only its owned rows. Preserve
            # each issue; the earliest is not a full-tile-ready timestamp.
            activation[-1]["publication_issues_by_warp_from_metadata_us"] = {
                warp: bounds(times) for warp, times in warps.items()}
            activation[-1]["last_A_publication_issue_from_metadata_us"] = max(
                t for times in warps.values() for t in times)
            activation[-1]["phase_start_scope"] = "marked warp only; cross-warp issue offsets may be signed"
    result = dict(sf=sf, activation=activation)
    starts = [m for m in markers if m["name"] == "GenphaseRemainderValidateStart"]
    if starts:
        publications = [m for m in markers if m["name"] == "GenphaseFullPayloadIssue"]
        begin, ends = {}, {}
        for items, target in ((starts, begin), (publications, ends)):
            for m in items:
                target.setdefault(location(m), []).append(m["time_us"])
        if begin.keys() != ends.keys():
            raise ValueError("Unpaired remainder validation/publication locations")
        intervals = []
        for (cta, warp), times in sorted(begin.items()):
            publications = ends[(cta, warp)]
            if min(publications) < max(times):
                raise ValueError("Remainder publication precedes validation")
            # Lane-predicated publications can emit multiple raw records per
            # warp. Retain their bounds/counts, as the profiler does.
            intervals.append(dict(cta=list(cta), warp=warp,
                                  start_from_metadata_us=min(times) - origin,
                                  publication_from_metadata_us=max(publications) - origin,
                                  start_bounds_from_metadata_us=bounds([t - origin for t in times]),
                                  publication_bounds_from_metadata_us=bounds([t - origin for t in publications]),
                                  raw_start_count=len(times), raw_publication_count=len(publications),
                                  validation_to_publication_us=max(publications) - min(times)))
        result["remainder"] = intervals
    return result


def first_pair_gather(markers, cta, first_issue, origin):
    pair = cta[0] & ~1
    observed = {}
    for m in markers:
        key = location(m)
        if (key[0][0] & ~1, *key[0][1:]) != (pair, *cta[1:]):
            continue
        if m["name"] in ("MegaIndicesReady", "MegaPayloadReady", "MegaGatherIssued"):
            observed.setdefault(key, {})[m["name"]] = m["time_us"]
    rows = []
    for key, points in sorted(observed.items()):
        rows.append(dict(cta=list(key[0]), warp=key[1],
                         markers_from_metadata_us={k: v - origin for k, v in points.items()}))
    issued = [p["MegaGatherIssued"] for p in observed.values() if "MegaGatherIssued" in p]
    return dict(observed_warps=len(observed), warps=rows,
                last_issue_marker_to_mma_us=first_issue - max(issued) if issued else None)


def remainder_wait_intervals(markers, origin):
    events = {name: {} for name in ("MegaRemainderWaitStart", "MegaRemainderReady")}
    for marker in markers:
        if marker["name"] in events:
            key = location(marker)
            if key in events[marker["name"]]:
                raise ValueError(f"Duplicate remainder marker at {key}")
            events[marker["name"]][key] = marker["time_us"]
    starts, ends = events.values()
    if starts.keys() != ends.keys():
        raise ValueError("Unpaired remainder-wait markers")
    result = []
    for (cta, warp), start in sorted(starts.items()):
        end = ends[(cta, warp)]
        if end < start:
            raise ValueError("Reversed remainder-wait interval")
        result.append(dict(cta=list(cta), warp=warp,
                           start_from_metadata_us=start - origin,
                           ready_from_metadata_us=end - origin,
                           wait_us=end - start))
    return result


def first_fc1_epilogue_observations(markers, origin):
    events = {name: {} for name in (
        "MegaFirstFc1AccObserved", "MegaFirstFc1EpiObserved", "GenphaseEnd",
    )}
    for m in markers:
        if m["name"] in events:
            key = location(m)[0]
            events[m["name"]].setdefault(key, []).append(m["time_us"])
    ready, epilogue, startup = events.values()
    if ready.keys() != epilogue.keys():
        raise ValueError("Unpaired first-FC1 accumulator/epilogue observations")
    result = []
    for cta in sorted(ready):
        observed, consumed = min(ready[cta]), min(epilogue[cta])
        end = max(startup[cta]) if cta in startup else None
        result.append(dict(
            cta=list(cta), ready_observed_from_metadata_us=observed - origin,
            epilogue_observed_from_metadata_us=consumed - origin,
            # Independent observers can be scheduled in either order. This is
            # signed observation skew, not exact hardware completion latency.
            epilogue_minus_ready_observation_us=consumed - observed,
            startup_end_from_metadata_us=None if end is None else end - origin,
            ready_observed_before_startup_end=end is not None and observed < end,
        ))
    return result


def summarize(data):
    metadata_materialized = data.get("interpretation", {}).get("receiver_metadata_materialized", True)
    if not isinstance(metadata_materialized, bool):
        raise ValueError("receiver_metadata_materialized must be boolean")
    samples = []
    for sample in data["samples"]:
        markers = sample["source_markers"]
        origin = sample["metadata_transfer_start_us"]
        issue = [m for m in markers if m["name"] == "MegaMmaIssue"]
        actual = bool(issue)
        if not actual:
            issue = [m for m in markers if m["name"] == "MegaComputeStart"]
        first = min(issue, key=lambda m: m["time_us"])
        cta = location(first)[0]
        # Pair each leader's first MMA with the final accumulator observation
        # from either CTA. This separates start-time skew from pipeline duration.
        pair_starts, pair_ends, pair_returns, pair_startup_ends = {}, {}, {}, {}
        pair_operands = {}
        for marker in issue:
            x, y, z = location(marker)[0]
            key = (x & ~1, y, z)
            pair_starts[key] = min(pair_starts.get(key, float("inf")), marker["time_us"])
        for marker in markers:
            if marker["name"] in ("MegaIndicesReady", "MegaPayloadReady",
                                  "MegaGatherSfReady", "MegaGatherIssued"):
                x, y, z = location(marker)[0]
                key = (x & ~1, y, z)
                pair_operands.setdefault(key, {}).setdefault(marker["name"], []).append(
                    marker["time_us"] - origin)
            if marker["name"] == "GenphaseEnd":
                x, y, z = location(marker)[0]
                key = (x & ~1, y, z)
                pair_startup_ends[key] = max(pair_startup_ends.get(key, float("-inf")), marker["time_us"])
            if marker["name"] == "MegaComputeEnd":
                x, y, z = location(marker)[0]
                key = (x & ~1, y, z)
                pair_ends[key] = max(pair_ends.get(key, float("-inf")), marker["time_us"])
            elif marker["name"] == "MegaCombineStart":
                x, y, z = location(marker)[0]
                key = (x & ~1, y, z)
                pair_returns[key] = min(pair_returns.get(key, float("inf")), marker["time_us"])
        if pair_starts.keys() != pair_ends.keys():
            raise ValueError("First-MMA and compute-end CTA pairs do not match")
        if pair_starts.keys() != pair_returns.keys():
            raise ValueError("First-MMA and first FC2 epilogue CTA pairs do not match")
        fc1_observations = first_fc1_epilogue_observations(markers, origin)
        pair_fc1_ready = {}
        for observation in fc1_observations:
            x, y, z = observation["cta"]
            key = (x & ~1, y, z)
            if key not in pair_starts:
                raise ValueError("First-FC1 observer has no matching MMA pair")
            # Use the later CTA observation. This is an upper bound on the
            # first accumulator's completion, not exact MMA hardware latency.
            pair_fc1_ready[key] = max(
                pair_fc1_ready.get(key, float("-inf")),
                observation["ready_observed_from_metadata_us"],
            )
        pair_spans = [dict(
            leader_cta=list(key),
            first_issue_from_metadata_us=start - origin,
            first_return_epilogue_from_metadata_us=pair_returns[key] - origin,
            compute_end_from_metadata_us=pair_ends[key] - origin,
            issue_to_compute_end_us=pair_ends[key] - start,
            issue_to_first_return_epilogue_us=pair_returns[key] - start,
            first_return_epilogue_to_compute_end_us=pair_ends[key] - pair_returns[key],
        ) for key, start in sorted(pair_starts.items())]
        for pair in pair_spans:
            key = tuple(pair["leader_cta"])
            if key in pair_operands:
                operands = {name: bounds(values) for name, values in pair_operands[key].items()}
                pair["first_stage_observations_from_metadata_us"] = operands
                if "MegaPayloadReady" in operands and "MegaGatherSfReady" in operands:
                    # These are different consumers' observations, not the exact
                    # publication or arrival times of the two operands.
                    pair["last_A_observation_minus_last_SF_observation_us"] = (
                        operands["MegaPayloadReady"]["last"] - operands["MegaGatherSfReady"]["last"])
                if "MegaGatherIssued" in operands:
                    pair["last_gather_marker_to_mma_us"] = (
                        pair["first_issue_from_metadata_us"] - operands["MegaGatherIssued"]["last"])
            if key in pair_fc1_ready:
                pair["first_fc1_ready_observed_from_metadata_us"] = pair_fc1_ready[key]
                elapsed = pair_fc1_ready[key] - pair["first_issue_from_metadata_us"]
                if elapsed < 0:
                    raise ValueError("First-FC1 accumulator observed before first MMA")
                pair["issue_to_first_fc1_ready_observation_us"] = elapsed
            if key in pair_startup_ends:
                pair["local_startup_end_from_metadata_us"] = pair_startup_ends[key] - origin
                # Negative means this pair issued MMA while at least one of
                # its own startup workers was still running. This is a local
                # overlap observation, not an inferred network transfer time.
                pair["first_issue_minus_local_startup_end_us"] = pair_starts[key] - pair_startup_ends[key]
        if any(p["issue_to_compute_end_us"] < 0 for p in pair_spans):
            raise ValueError("Compute end precedes first MMA in a CTA pair")
        last_start = max(pair_spans, key=lambda p: p["first_issue_from_metadata_us"])
        last_finish = max(pair_spans, key=lambda p: p["compute_end_from_metadata_us"])
        sender_intervals = sender_publication_intervals(markers, origin)
        producer = next((p for p in data.get("producer_metadata_samples", [])
                         if p["rank"] == sample["rank"] and
                         p["measured_ordinal"] == sample["measured_ordinal"]), None)
        producer_bounds, sf_early_ctas = {}, 0
        if producer:
            events = producer["markers"]
            metadata_ns = min(m["timestamp_ns"] for m in events
                              if m["name"] == "RouteMetadataTransferStart")
            for name in sorted({m["name"] for m in events}):
                producer_bounds[name] = bounds([
                    (m["timestamp_ns"] - metadata_ns) / 1000.0
                    for m in events if m["name"] == name])
            sf_ends, metadata_ends = {}, {}
            for m in events:
                target = (sf_ends if m["name"] == "RouteSfTransferIssued" else
                          metadata_ends if m["name"] == "RouteMetadataIssued" else None)
                if target is not None:
                    key = location(m)[0]
                    target[key] = max(target.get(key, 0), m["timestamp_ns"])
            sf_early_ctas = sum(end < metadata_ends[key]
                               for key, end in sf_ends.items() if key in metadata_ends)
        operation_markers = {}
        for name in ("MegaMetadataAcquired", "MegaPayloadLoadStart", "MegaPullIssued",
                     "MegaPullArrived", "MegaGatherSfStart"):
            values = [m["time_us"] - origin for m in markers if m["name"] == name]
            if values:
                operation_markers[name] = bounds(values)
        waits = {}
        for phase in ("L1", "L2"):
            starts = {location(m): m["time_us"] for m in markers
                      if m["name"] == f"MegaEpi{phase}Wait"}
            ends = {location(m): m["time_us"] for m in markers
                    if m["name"] == f"MegaEpi{phase}Ready"}
            if starts.keys() != ends.keys():
                raise ValueError(f"Unpaired epilogue markers: {phase}")
            if starts:
                values = [ends[key] - start for key, start in starts.items()]
                if min(values) < 0:
                    raise ValueError(f"Reversed epilogue wait: {phase}")
                waits[phase] = bounds(values)
        samples.append(dict(
            ordinal=sample["measured_ordinal"], rank=sample["rank"],
            first_fc1_epilogue_observations=fc1_observations,
            remainder_wait_intervals=remainder_wait_intervals(markers, origin),
            pair_issue_to_first_fc1_ready_observation_us=bounds([
                p["issue_to_first_fc1_ready_observation_us"] for p in pair_spans
                if "issue_to_first_fc1_ready_observation_us" in p
            ]) if pair_fc1_ready else None,
            first_issue_is_actual=actual, first_issue_cta=list(cta),
            mma_release_from_metadata_us=bounds([m["time_us"] - origin for m in issue]),
            kernel_entry_from_metadata_us=-origin,
            first_issue_from_kernel_us=first["time_us"],
            first_issue_from_metadata_us=first["time_us"] - origin,
            first_issue_from_selector_us=first["time_us"] + sample["producer_entry_to_megamoe_entry_us"],
            compute_end_from_metadata_us=sample["compute_end_us"] - origin,
            issue_to_compute_end_us=sample["compute_end_us"] - first["time_us"],
            pair_issue_to_compute_end_us=bounds([p["issue_to_compute_end_us"] for p in pair_spans]),
            # CombineStart is the first FC2 epilogue, after its accumulator wait;
            # it is not the start of FC2 MMA or the end of FC1 computation.
            pair_issue_to_first_return_epilogue_us=bounds([p["issue_to_first_return_epilogue_us"] for p in pair_spans]),
            pair_first_return_epilogue_to_compute_end_us=bounds([p["first_return_epilogue_to_compute_end_us"] for p in pair_spans]),
            last_finishing_pair=last_finish,
            last_starting_pair=last_start,
            last_starting_pair_from_metadata_us=startup_markers(markers, last_start["leader_cta"], origin),
            last_finishing_pair_from_metadata_us=startup_markers(markers, last_finish["leader_cta"], origin),
            compute_pairs=pair_spans,
            mma_before_local_startup_end_pair_count=sum(
                p.get("first_issue_minus_local_startup_end_us", 0) < 0 for p in pair_spans),
            sender_publication_intervals=sender_intervals,
            sender_store_retirement=sender_store_retirement(markers, origin),
            local_readiness_publications=local_readiness_publications(markers, origin, metadata_materialized),
            first_pair_gather=first_pair_gather(markers, cta, first["time_us"], origin),
            producer_from_metadata_us=producer_bounds,
            operation_markers_from_metadata_us=operation_markers,
            producer_sf_issued_before_metadata_issued_cta_count=sf_early_ctas,
            kernel_end_from_metadata_us=sample["metadata_start_to_megamoe_end_us"],
            critical_pair_from_metadata_us=startup_markers(markers, cta, origin),
            epilogue_wait_us=waits,
            genphase_from_metadata_us={
                name: {k: value - origin if k.endswith("_us") else value
                       for k, value in event.items()}
                for name, event in (sample["genphase_markers_relative_to_megamoe"] or {}).items()
            },
        ))
    return dict(kernel_us=data["nsys_kernel_us"], samples=samples)


def read_ordinary_passes(path, result):
    """Validate an ordered timing sidecar against the profiled arms."""
    passes = json.loads(path.read_text())
    if not isinstance(passes, dict) or not result.keys() <= passes.keys():
        raise ValueError("Ordinary timing input must include every profiled arm")
    fields = ("mean", "median", "min", "max")
    for name, values in passes.items():
        if not isinstance(values, dict) or any(
            key not in values or isinstance(values[key], bool)
            or not isinstance(values[key], (int, float))
            or not math.isfinite(values[key]) for key in fields
        ):
            raise ValueError(f"Invalid ordinary timing values: {name}")
        if not (0 < values["min"] <= values["mean"] <= values["max"]
                and values["min"] <= values["median"] <= values["max"]):
            raise ValueError(f"Invalid ordinary timing bounds: {name}")
        if name in result and any(not math.isclose(
                values[key], result[name]["kernel_us"][key], rel_tol=1e-9,
                abs_tol=1e-9) for key in fields):
            raise ValueError(f"Ordinary timing input disagrees with capture: {name}")
    return passes


def markdown_report(result, notes, ordinary_passes=None):
    """Keep ordinary timings separate and preserve each instrumented capture."""
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    def interval(value):
        if not value:
            return None
        first = value.get("first_us", value.get("first"))
        last = value.get("last_us", value.get("last"))
        return cell(first) if first == last else f"{cell(first)}–{cell(last)}"

    names = list(result)
    lines = [
        "**MegaMoE latency comparison**", "",
        "Times are µs. Each capture uses its own producer metadata-transfer start as zero. "
        "Ranges are the first–last observed markers in that capture; they are not transfer durations. "
        "Missing markers are shown as —. All captures and outliers are retained.", "",
        "**Ordinary kernel timing**", "",
        "| Arm | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    timings = ({name: arm["kernel_us"] for name, arm in result.items()}
               if ordinary_passes is None else ordinary_passes)
    for name, values in timings.items():
        lines.append("| " + " | ".join([cell(name)] + [
            cell(values[key]) for key in ("mean", "median", "min", "max")
        ]) + " |")
    if ordinary_passes is not None:
        lines += ["", "Ordinary passes appear in the input's execution order; each row's "
                  "statistics belong to that pass. Medians are not pooled across repeats."]
    lines += ["", "The following tables are instrumented. Do not add their overlapping phases "
              "or subtract them from the ordinary kernel timings above."]
    captures = {name: {(s["rank"], s["ordinal"]): s for s in arm["samples"]}
                for name, arm in result.items()}
    keys = sorted({key for samples in captures.values() for key in samples})
    for rank, ordinal in keys:
        samples = [captures[name].get((rank, ordinal)) for name in names]
        lines += ["", f"**Capture {ordinal}, rank {rank}**", "",
                  "| Milestone / duration | " + " | ".join(map(cell, names)) + " |",
                  "|---|" + "---:|" * len(names)]

        def row(label, get):
            values = [get(s) if s is not None else None for s in samples]
            if any(v is not None for v in values):
                lines.append("| " + cell(label) + " | " + " | ".join(map(cell, values)) + " |")

        row("Metadata transfer starts", lambda s: 0)
        row("Kernel entry", lambda s: s["kernel_entry_from_metadata_us"])
        for event, label in (
            ("RouteMetadataIssued", "Producer metadata stores issued"),
            ("RouteSfTransferStart", "Producer SF transfer starts"),
            ("RouteSfTransferIssued", "Producer SF stores issued"),
            ("RouteMetadataPublished", "Producer metadata-ready publication marker"),
        ):
            row(label, lambda s, event=event: interval(s["producer_from_metadata_us"].get(event)))
        for event, label in (
            ("Start", "Startup workers enter"),
            ("TopkReceived", "Receiver route metadata acquired"),
            ("SizesReady", "Sizes-ready publication issue"),
            ("IndexScatter", "SF/index phase starts (see ownership notes)"),
            ("SfStoresIssued", "SF worker stores issued (variant warp scope)"),
            ("MetadataStoresIssued", "Index/score worker stores issued"),
            ("SfReadyIssue", "Local SF readiness publication issue"),
            ("PayloadSendStart", "Payload send phase starts"),
            ("PayloadStoresIssued", "Payload stores issued"),
            ("PayloadValidateStart", "Receiver activation validation starts"),
            ("FirstPayloadIssue", "First-window A readiness publication issue"),
            ("RemainderValidateStart", "Receiver remainder validation starts"),
            ("FullPayloadIssue", "Remaining A readiness publication issue (see scope)"),
            ("End", "Startup workers retire"),
        ):
            if event == "PayloadSendStart":
                row("Payload source-load phase starts (variant scope)", lambda s:
                    interval(s["operation_markers_from_metadata_us"].get("MegaPayloadLoadStart")))
            row(label, lambda s, event=event: interval(s["genphase_from_metadata_us"].get(event)))
        for event, label in (
            ("MegaMetadataAcquired", "Compute CTAs: metadata acquired"),
            ("MegaPullIssued", "Pull issue markers"),
            ("MegaPullArrived", "Pull completion observations"),
        ):
            row(label, lambda s, event=event:
                interval(s["operation_markers_from_metadata_us"].get(event)))
        for event, label in (
            ("MegaIndicesReady", "First MMA pair: index-path marker (variant scope)"),
            ("MegaPayloadReady", "First MMA pair: activation readiness observed"),
            ("MegaGatherSfStart", "First MMA pair: SF gather starts"),
            ("MegaGatherSfReady", "First MMA pair: SF readiness observed"),
            ("MegaGatherIssued", "First MMA pair: activation load issued"),
        ):
            row(label, lambda s, event=event: interval(s["critical_pair_from_metadata_us"].get(event)))
        row("First MMA / operand-ready boundary", lambda s:
            f'{cell(s["first_issue_from_metadata_us"])} '
            f'({"actual MMA" if s["first_issue_is_actual"] else "operand-ready proxy"})')
        row("Pair-start spread: first / median / last", lambda s: " / ".join(
            cell(s["mma_release_from_metadata_us"][key]) for key in ("first", "median", "last")))
        for field, label in (
            ("last_A_observation_minus_last_SF_observation_us",
             "All pairs: median latest A minus latest SF observation"),
            ("last_gather_marker_to_mma_us", "All pairs: median last activation-load marker to MMA"),
        ):
            row(label, lambda s, field=field: median([
                pair[field] for pair in s["compute_pairs"] if field in pair
            ]) if any(field in pair for pair in s["compute_pairs"]) else None)
        row("Remainder issuer wait, median duration", lambda s: median([
            v["wait_us"] for v in s["remainder_wait_intervals"]
        ]) if s["remainder_wait_intervals"] else None)
        row("Final accumulator-ready observation", lambda s: s["compute_end_from_metadata_us"])
        row("Compute span, duration", lambda s: s["issue_to_compute_end_us"])
        row("Kernel end", lambda s: s["kernel_end_from_metadata_us"])
        row("First MMA from selector entry, duration", lambda s: s["first_issue_from_selector_us"])
    lines += ["", "**Marker scope and sources**", "",
              "Producer events describe the profiled rank's outgoing work; they do not time incoming "
              "network latency. Publication issue is not completion. Readiness markers are consumer "
              "observations; differences between A and SF observations are not isolated operand stalls. "
              "Remainder wait measures the activation issuer, not an exact MMA stall. "
              "Compute span runs from the indicated first-MMA boundary to the final accumulator-ready "
              "observation. Consult each capture's variant notes for ownership and transport details.", ""]
    for name, arm in result.items():
        source = Path(arm["source"]).resolve()
        lines.append(f"- {cell(name)}: [{source.name}](<{source}>)")
        for key in ("genphase_index_scatter", "sf_ready_contract", "index_path", "activation_load", "remainder_wait"):
            if notes[name].get(key):
                lines.append("")
                lines.append(f"  {key}: {cell(notes[name][key])}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="NAME=decomposition.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path,
                        help="Optional side-by-side report preserving every capture")
    parser.add_argument("--ordinary-input", type=Path,
                        help="Ordered JSON of ordinary pass timings, including repeats; "
                             "requires --markdown-output")
    args = parser.parse_args()
    if args.ordinary_input and not args.markdown_output:
        parser.error("--ordinary-input requires --markdown-output")
    result, notes = {}, {}
    for item in args.inputs:
        name, path = item.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate arm: {name}")
        data = json.loads(Path(path).read_text())
        result[name] = summarize(data)
        notes[name] = data.get("interpretation", {})
        result[name]["source"] = path
    ordinary = read_ordinary_passes(args.ordinary_input, result) if args.ordinary_input else None
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.write_text(markdown_report(result, notes, ordinary))
    for name, arm in result.items():
        print(json.dumps(dict(arm=name, kernel_us=arm["kernel_us"], samples=[
            {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
            for s in arm["samples"]
        ])))


if __name__ == "__main__":
    main()
