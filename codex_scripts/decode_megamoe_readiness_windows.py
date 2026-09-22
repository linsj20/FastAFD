"""Decode reused IKET names without losing their raw names or window identity."""

from collections import defaultdict
import math


def decode_mma_retirement(markers, *, leader_ctas):
    """Decode the optional warp6 probe before decoding warp0 readiness windows."""
    leaders = list(leader_ctas)
    if (not leaders or len(set(leaders)) != len(leaders)
            or any(type(c) is not int or c < 0 or c % 2 for c in leaders)):
        raise ValueError("require distinct active even leader CTAs")
    names = {"MegaWindowWaitStart": "MegaMmaRetireWaitStart",
             "MegaWindowReady": "MegaMmaRetireWaitEnd"}
    if any(m["name"] in names.values() for m in markers):
        raise ValueError("mixed raw and decoded MMA retirement markers")
    expected = {(c, 0, 0, 6) for c in leaders}
    events = {name: {} for name in names.values()}
    out = []
    for marker in markers:
        m = dict(marker)
        loc = m["location"]
        if m["name"] in names and loc["warpId"] == 6:
            key = (*loc["ctaId"], 6)
            name, time = names[m["name"]], m["time_us"]
            if (key not in expected or key in events[name]
                    or not math.isfinite(time) or time < 0 or "wire_name" in m):
                raise ValueError("invalid or duplicate MMA retirement marker")
            events[name][key] = time
            m.update(wire_name=m["name"], name=name)
        out.append(m)
    if any(set(values) != expected for values in events.values()):
        raise ValueError("incomplete MMA retirement marker coverage")
    intervals = []
    for key in sorted(expected):
        start = events["MegaMmaRetireWaitStart"][key]
        end = events["MegaMmaRetireWaitEnd"][key]
        if end < start:
            raise ValueError("reversed MMA retirement interval")
        intervals.append(dict(cta=list(key[:3]), warp=6, start_us=start,
                              end_us=end, duration_us=end - start))
    return out, intervals


def decode_readiness_windows(markers, *, hidden, num_sms, validation_ctas, index_ctas,
                             middle_end=2048, active_index_ctas=None,
                             active_validation_ctas=None, active_issuer_ctas=None):
    if hidden <= 1024 or hidden % 512 or not 0 < index_ctas == validation_ctas <= num_sms // 2:
        raise ValueError("invalid readiness-window shape")
    if middle_end <= 1024 or middle_end % 512:
        raise ValueError("invalid middle-window endpoint")
    def participants(actual, allocated, role):
        expected = set(allocated)
        if actual is None:
            return expected
        actual = list(actual)
        if not actual or len(actual) != len(set(actual)) or not set(actual) <= expected:
            raise ValueError(f"invalid active {role} CTAs")
        return set(actual)
    active_sf = participants(active_index_ctas, range(index_ctas), "index")
    active_a = participants(active_validation_ctas,
                            range(index_ctas, index_ctas + validation_ctas), "validation")
    active_issue = participants(active_issuer_ctas, range(num_sms), "issuer")
    tail = hidden > middle_end
    repeated = {
        "GenphaseWindowValidateStart": ("GenphasePayloadValidateStart", "GenphaseTailValidateStart"),
        "GenphaseWindowPayloadIssue": ("GenphaseFirstPayloadIssue", "GenphaseTailPayloadIssue"),
        "MegaWindowWaitStart": ("MegaRemainderWaitStart", "MegaTailWaitStart"),
        "MegaWindowReady": ("MegaRemainderReady", "MegaTailReady"),
    }
    forbidden = {name for names in repeated.values() for name in names} | {"GenphaseFullPayloadIssue"}
    if any(m["name"] in forbidden for m in markers):
        raise ValueError("mixed raw and decoded readiness-window markers")
    out = [dict(m) for m in markers]
    indexed = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(out):
        loc = m["location"]
        key = (*loc["ctaId"], loc["warpId"])
        if not math.isfinite(m["time_us"]):
            raise ValueError("nonfinite readiness marker timestamp")
        indexed[m["name"]][key].append(i)
    for raw, names in repeated.items():
        expected = ({(c, 0, 0, 8) for c in active_a}
                    if raw.startswith("Genphase") else
                    {(c, 0, 0, 0) for c in active_issue})
        if indexed[raw].keys() != expected:
            raise ValueError(f"incomplete window locations: {raw}")
        for key, ids in indexed[raw].items():
            ids.sort(key=lambda i: out[i]["time_us"])
            if len(ids) != 1 + tail:
                raise ValueError(f"invalid window occurrence count: {raw}, {key}")
            for ordinal, i in enumerate(ids):
                out[i].update(name=names[ordinal], wire_name=raw, window_ordinal=ordinal)
    events = defaultdict(dict)
    for m in out:
        loc = m["location"]
        key = (*loc["ctaId"], loc["warpId"])
        if key in events[m["name"]]:
            raise ValueError(f"duplicate semantic marker: {m['name']}, {key}")
        events[m["name"]][key] = m["time_us"]

    def intervals(start, end, expected):
        if events[start].keys() != expected or events[end].keys() != expected:
            raise ValueError(f"incomplete window interval: {start}, {end}")
        rows = []
        for key in sorted(expected):
            a, b = events[start][key], events[end][key]
            if a > b:
                raise ValueError(f"reversed window interval: {start}, {key}")
            rows.append(dict(cta=list(key[:3]), warp=key[3], start_us=a,
                             end_us=b, duration_us=b-a))
        return rows

    a = {(c, 0, 0, 8) for c in active_a}
    sf = {(c, 0, 0, 12) for c in active_sf}
    issuers = {(c, 0, 0, 0) for c in active_issue}
    detail = {
        "prefix_validation": intervals("GenphasePayloadValidateStart", "GenphaseFirstPayloadIssue", a),
        "middle_validation": intervals("GenphaseRemainderValidateStart", "GenphaseMiddlePayloadIssue", sf),
        "middle_issuer_wait": intervals("MegaRemainderWaitStart", "MegaRemainderReady", issuers),
        "tail_validation": [], "tail_issuer_wait": [],
        "byte_windows": [[0, 1024], [1024, min(middle_end, hidden)]] + ([[middle_end, hidden]] if tail else []),
        "timestamp_origin": "rank-local kernel entry",
    }
    if tail:
        detail["tail_validation"] = intervals("GenphaseTailValidateStart", "GenphaseTailPayloadIssue", a)
        detail["tail_issuer_wait"] = intervals("MegaTailWaitStart", "MegaTailReady", issuers)
        for key in a:
            if events["GenphaseFirstPayloadIssue"][key] > events["GenphaseTailValidateStart"][key]:
                raise ValueError("tail validation precedes prefix publication")
        for key in issuers:
            if events["MegaRemainderReady"][key] > events["MegaTailWaitStart"][key]:
                raise ValueError("tail acquire precedes middle acquire")
    return out, detail
