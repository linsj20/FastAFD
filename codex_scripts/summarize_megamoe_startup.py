#!/usr/bin/env python3
"""Summarize rank-local startup waits from saved all-rank IKET captures.

Timestamps from different ranks are deliberately not subtracted: their clock
offsets have not been calibrated. All measured samples are retained.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def indexed(table, index):
    return table[int(index)] if isinstance(table, list) else table[str(index)]


def inter_sample_intervals(records):
    """Compare same-rank intervals; fixed clock offsets cancel before pooling.

    A narrowing interval range at route readiness is consistent with ranks
    converging on peer publication. It does not identify the cause of launch
    skew or measure absolute arrival skew, transport latency, or clock drift.
    """
    if not records:
        raise ValueError('No rank records')
    count = len(next(iter(records.values())))
    if any([row['ordinal'] for row in rows] != list(range(count))
           for rows in records.values()):
        raise ValueError('Inconsistent sample ordinals across ranks')
    intervals = []
    for ordinal in range(1, count):
        ranks = {}
        for rank, rows in records.items():
            previous, current = rows[ordinal - 1:ordinal + 1]
            gap = (current['main_start_timestamp_ns'] -
                   previous['main_start_timestamp_ns']) / 1000
            if gap <= 0:
                raise ValueError('Nonmonotonic main launches')
            ranks[rank] = dict(main_start_us=gap)
            for name, metric in [('routes_ready_us', 'all_routes_ready_us'),
                                 ('first_mma_us', 'first_mma_issue_us')]:
                ranks[rank][name] = gap + current[metric] - previous[metric]
                if ranks[rank][name] <= 0:
                    raise ValueError(f'Nonmonotonic {name} events')
        ranges = {metric: max(row[metric] for row in ranks.values()) -
                         min(row[metric] for row in ranks.values())
                  for metric in ('main_start_us', 'routes_ready_us', 'first_mma_us')}
        intervals.append(dict(from_ordinal=ordinal - 1, to_ordinal=ordinal,
                              ranks=ranks, interval_range_across_ranks_us=ranges))
    return intervals


def summarize_rank(path, samples):
    trace = json.loads(path.read_text())
    strings = trace['stringTable']
    launches = list(trace['launches'])
    graphs = trace.get('graphLaunches', {})
    if not isinstance(graphs, dict):
        raise ValueError('Expected graph-keyed trace')
    for key in sorted(graphs, key=lambda x: int(re.search(r':(\d+)$', x)[1])):
        group = graphs[key]
        launches.extend(group['launches'] if isinstance(group, dict) else group)
    main, producer = [], []
    for launch in launches:
        name = launch.get('kernelName', '')
        if isinstance(name, int):
            name = indexed(strings, name)
        if 'mega_moe' in name and 'fp4' in name:
            main.append(launch)
        if any(indexed(strings, m['markerNameIdx']) == 'RouteMetadataPublished'
               for m in launch.get('markers', [])):
            producer.append(launch)
    if len(main) != len(producer) or len(main) < samples:
        raise ValueError(f'Unpaired producer/main launches: {path}')
    records = []
    for ordinal, (p, m) in enumerate(zip(producer[-samples:], main[-samples:])):
        start = min(w['startTs'] for w in m['warpLifetimes'])
        end = max(w['endTs'] for w in m['warpLifetimes'])
        ps = min(w['startTs'] for w in p['warpLifetimes'])
        pe = max(w['endTs'] for w in p['warpLifetimes'])
        markers = defaultdict(list)
        for event in m['markers']:
            markers[indexed(strings, event['markerNameIdx'])].append(event['timestamp'])
        waits = {}
        for event in m['markers']:
            name = indexed(strings, event['markerNameIdx'])
            if name in ('GenphaseMetadataFromProducer', 'GenphaseTopkReceived'):
                waits.setdefault(event.get('locIdx', event.get('locationIdx')), {})[name] = event['timestamp']
        durations = [(v['GenphaseTopkReceived'] - v['GenphaseMetadataFromProducer']) / 1000
                     for v in waits.values()]
        if not durations or min(durations) < 0:
            raise ValueError(f'Invalid readiness interval: {path}')
        pm = defaultdict(list)
        for event in p['markers']:
            pm[indexed(strings, event['markerNameIdx'])].append(event['timestamp'])
        records.append(dict(
            ordinal=ordinal, kernel_us=(end-start)/1000,
            producer_us=(pe-ps)/1000, producer_to_main_us=(start-ps)/1000,
            producer_publish_us=(max(pm['RouteMetadataPublished'])-ps)/1000,
            producer_metadata_us=(max(pm['RouteMetadataPublished'])-min(pm['RouteMetadataTransferStart']))/1000
                if pm['RouteMetadataTransferStart'] else None,
            producer_ids_issued_us=(max(pm['RouteMetadataIssued'])-ps)/1000
                if pm['RouteMetadataIssued'] else None,
            producer_sf_issued_us=(max(pm['RouteSfTransferIssued'])-ps)/1000
                if pm['RouteSfTransferIssued'] else None,
            producer_ids_issue_to_publish_us=(max(pm['RouteMetadataPublished'])-max(pm['RouteMetadataIssued']))/1000
                if pm['RouteMetadataIssued'] else None,
            producer_sf_issue_to_publish_us=(max(pm['RouteMetadataPublished'])-max(pm['RouteSfTransferIssued']))/1000
                if pm['RouteSfTransferIssued'] else None,
            route_wait_min_us=min(durations), route_wait_max_us=max(durations),
            metadata_wait_entry_us=(min(markers['GenphaseMetadataFromProducer'])-start)/1000,
            all_routes_ready_us=(max(markers['GenphaseTopkReceived'])-start)/1000,
            payload_stores_issued_us=(max(markers['GenphasePayloadStoresIssued'])-start)/1000
                if markers['GenphasePayloadStoresIssued'] else None,
            first_mma_issue_us=(min(markers['MegaMmaIssue'])-start)/1000,
            main_start_timestamp_ns=start,
        ))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for run in args.runs:
        contract = json.loads((run/'decomposition.json').read_text())['contract']
        if any(contract.get(key) for key in ('receiver_route_sort', 'expert_parallel_sort',
                                             'source_direct_sort', 'source_parallel_sort')):
            raise ValueError(f'Retired sort capture requires its frozen workflow: {run}')
        records = {}
        for rank in sorted((run/'ranks').glob('rank-*'), key=lambda p: int(p.name[5:])):
            paths = list((rank/'iket').glob('*.trace.json'))
            if len(paths) != 1:
                raise ValueError(f'Expected one trace: {rank}')
            records[rank.name[5:]] = summarize_rank(paths[0], args.samples)
        expected = contract['ep_size']
        if set(records) != {str(i) for i in range(expected)}:
            raise ValueError(f'Incomplete ranks: {run}')
        key = '/'.join(run.parts[-3:])
        results[key] = dict(rank_count=expected, samples=args.samples, ranks=records,
                           inter_sample_intervals=inter_sample_intervals(records))
        print(f'{key}: {expected} ranks x {args.samples} samples', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        scope='Rank-local durations only; no cross-rank clock alignment', cases=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
