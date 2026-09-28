"""CPU checks for submission records and refusal to overwrite native caches."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codex_scripts'))
import seed_native_cache
import submit_plan
import ncu_tcp_launcher
from summarize_megamoe_router_timeline import audit_joined_preparation
from analyze_megamoe_throughput import aggregate, aggregate_summaries, audit_correctness, timeline_windows
from prepare_megamoe_throughput import normal_route_bucket, sweep_policy_difference
from decode_megamoe_readiness_windows import decode_readiness_windows, decode_mma_wait_probe
from review_megamoe_dispatch_sass import (compare_preparations, preparation_instructions,
                                         diagnostic_syscall_blocks, normalize_validation_helper_symbols)


class CodexWorkflowTests(unittest.TestCase):
    def test_sweep_rejects_reused_capture_outputs_before_launch(self):
        from run_megamoe_sweep_group import load_groups
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(contract=dict(model='qwen3', hidden=128, intermediate=128,
                experts=8, top_k=2, num_sms=150, bucket=16), cases=[
                dict(case_id=f'ep2_tokens{total}', ep=2, global_tokens=total,
                     rows_per_rank=total // 2) for total in (2, 4)])
            for arm in ('control', 'control_repeat'):
                (root / arm).mkdir()
                (root / arm / 'manifest.json').write_text(json.dumps(manifest))
            env = dict(MEGAMOE_SWEEP_ROOT=str(root / 'control'), MEGAMOE_IKET_EP='2',
                       MEGAMOE_SWEEP_CASES='ep2_tokens2', FASTAFD_IKET_CAPTURE_PHASE='normal')
            forward = dict(arm='control', env=env)
            repeat = dict(arm='control_repeat', env=dict(env))
            with self.assertRaisesRegex(ValueError, 'Duplicate capture output'):
                load_groups(dict(ep=2, groups=[forward, repeat]))
            repeat['env']['MEGAMOE_SWEEP_ROOT'] = str(root / 'control_repeat')
            self.assertEqual(len(load_groups(dict(ep=2, groups=[forward, repeat]))), 2)
            # Split stages can share a root when their cases do not overlap.
            large = dict(arm='control', env=dict(env, MEGAMOE_SWEEP_CASES='ep2_tokens4'))
            profile = dict(arm='control', env=dict(env, FASTAFD_IKET_CAPTURE_PHASE='profile'))
            self.assertEqual(len(load_groups(dict(ep=2, groups=[forward, large, profile]))), 3)
            both = dict(arm='control', env=dict(env, FASTAFD_IKET_CAPTURE_PHASE='both'))
            with self.assertRaisesRegex(ValueError, 'Duplicate capture output'):
                load_groups(dict(ep=2, groups=[both, profile]))
            (root / 'alias').symlink_to(root / 'control', target_is_directory=True)
            repeat['env']['MEGAMOE_SWEEP_ROOT'] = str(root / 'alias')
            with self.assertRaisesRegex(ValueError, 'Duplicate capture output'):
                load_groups(dict(ep=2, groups=[forward, repeat]))

    def test_normal_routing_capacity_and_complete_graph_audit(self):
        from collect_cuda_rank_timings import aggregate_rank_timings
        from analyze_megamoe_throughput import review_normal
        self.assertEqual(normal_route_bucket(1024, 8, 512, 10), 22656)
        with self.assertRaises(ValueError):
            normal_route_bucket(31, 8, 512, 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'analysis').mkdir()
            (root / 'logs').mkdir()
            runs, logs = [], ['', '']
            for arm in ['control', 'candidate', 'pull', 'pull_repeat', 'candidate_repeat', 'control_repeat']:
                path = root / arm
                path.mkdir()
                name = arm.removesuffix('_repeat')
                mode = 'record' if arm == 'control' else 'compare' if name != 'pull' else None
                env = dict(MEGAMOE_SMOKE_FIXED_PERFECT_ONLY='0', MEGAMOE_SMOKE_PERFECT_ROUTING='0',
                    MEGAMOE_SMOKE_SPECIALIZATION_MODES='1:1:1:1', MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT='joined',
                    MEGAMOE_SMOKE_GRAPH_BATCH='1', MEGAMOE_SMOKE_PREPARE_BARRIER='collective', DG_JIT_REQUIRE_CACHE='1',
                    MEGAMOE_SMOKE_ROWS='16', MEGAMOE_SMOKE_BUCKET='384', MEGAMOE_SMOKE_SMS='150')
                if mode:
                    env['MEGAMOE_RESULT_MODE'] = mode
                sample = dict(control=.004, candidate=.002, pull=.003)[name]
                audit = aggregate_rank_timings([dict(rank=rank, scope='joined preparation barrier/reset',
                    durations_ms=[sample] * 100) for rank in range(2)])
                (path / 'rank_timings.json').write_text(json.dumps(audit))
                info = dict(timed_scope='router_and_megamoe', calls_per_graph=1, prepared_graph_layout='joined',
                    dispatch_reset_outside_timing=False, milliseconds=audit['maximum_rank_mean_ms'], max_active_experts=8)
                (path / 'result.json').write_text(json.dumps(dict(status='ok', expert_weight_dtype='fp4',
                    numeric=None, exact_qwen=dict(case=info))))
                label = 'exact_bucket384_rows16_sms150_trial0'
                labels = ['exact_inputs', label, label + '_alternate',
                          *[label + f'_normal{i}' for i in range(3)], *[label + f'_padding{i}' for i in range(3)]]
                for rank in range(2):
                    if mode:
                        logs[rank] += ''.join(f'MEGAMOE_RESULT_{mode.upper()}_OK case={x} rank={rank}\n' for x in labels)
                    logs[rank] += f'MEGAMOE_REPLAY_EXACT_OK rank={rank} rows=16 scope=pipeline\n'
                runs.append(dict(model='qwen3', total=32, arm=arm, path=str(path), env=env))
            plan = dict(ep=2, arm_prefixes=['control', 'candidate', 'pull'], candidate='candidate',
                        specs=dict(qwen3=[32]), routing='normal', calls_per_arm=100, runs=runs)
            (root / 'scope.json').write_text(json.dumps(plan))
            for rank, log in enumerate(logs):
                (root / 'logs' / f'rank-{rank}.log').write_text(log)
            with contextlib.redirect_stdout(io.StringIO()):
                review_normal([str(root)])
            report = json.loads((root / 'analysis/review.json').read_text())
            self.assertEqual(report['correctness'], dict(record=18, compare=54, replay=12))
            self.assertEqual(report['rank_samples'], 1200)
            self.assertAlmostEqual(report['geometric_speedups']['control']['geometric_mean_speedup'], 2.)
            (root / 'logs/rank-1.log').write_text(logs[1].replace('_padding2', '_padding1', 1))
            with self.assertRaises(ValueError):
                review_normal([str(root)])

    def test_sweep_calibration_rejects_source_cache_and_layout_changes(self):
        a = dict(FASTAFD_SOURCE_REPO='/source', DG_JIT_CACHE_DIR='/cache',
                 DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS='1', MEGAMOE_SWEEP_ROOT='/a')
        b = dict(a, DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS='2', MEGAMOE_SWEEP_ROOT='/b')
        self.assertEqual(sweep_policy_difference(a, b), ['DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS'])
        for key, value in [('FASTAFD_SOURCE_REPO', '/other'), ('DG_JIT_CACHE_DIR', '/other'),
                           ('DG_MEGAMOE_ALL_PUSH_BLOCK_M', '96'), ('MEGAMOE_RESULT_MODE', 'record')]:
            with self.assertRaises(ValueError):
                sweep_policy_difference(a, dict(b, **{key: value}))
        with self.assertRaises(ValueError):
            sweep_policy_difference(a, a)

    def test_interleaved_calibration_keeps_aa_bb_out_of_policy_score(self):
        from collect_cuda_rank_timings import aggregate_rank_timings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            (root / 'analysis').mkdir()
            runs = []
            for pair in ['aa', 'ab', 'bb']:
                for order in ['abba', 'baab']:
                    labels = ['workers', 'rows'] if order == 'abba' else ['rows', 'workers']
                    flags = {label: {'DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS':
                             '1' if pair == 'aa' or (pair == 'ab' and label == 'workers') else '2'}
                             for label in labels}
                    arms = {label: aggregate_rank_timings([
                        dict(rank=rank, scope='joined preparation', metadata=dict(variant=flags[label]),
                             durations_ms=[(.04 if label == 'workers' else .02) if pair == 'ab'
                                           else (.01 if label == 'workers' else .05)] * 100)
                        for rank in range(2)]) for label in labels}
                    path = root / pair / order
                    path.mkdir(parents=True)
                    a, b = labels
                    capture = dict(status='PASS_INTERLEAVED_EXACT', variants=flags, entry_protocol='synchronized',
                                   warmups_per_arm=10, calls_per_arm=100, schedule=[a,b,b,a]*50,
                                   scalar_result_arm=labels[0], arms=arms)
                    (path / 'interleaved.json').write_text(json.dumps(capture))
                    info = dict(prepared_graph_layout='joined', timing_graph_batch=1,
                                dispatch_reset_outside_timing=False,
                                milliseconds=arms[labels[0]]['maximum_rank_mean_ms'])
                    (path / 'result.json').write_text(json.dumps(dict(status='ok', expert_weight_dtype='fp4',
                        numeric=None, exact_qwen=dict(case=info))))
                    runs.append(dict(model='qwen35', total=512, pair=pair, order=order, path=str(path),
                                     env=dict(MEGAMOE_INTERLEAVED_VARIANTS=json.dumps(flags))))
            counts = dict(RESULT_COMPARE=12, REPLAY_EXACT=12, INTERLEAVED_EXACT=12)
            for rank in range(2):
                (root / 'logs' / f'rank-{rank}.log').write_text(''.join(
                    f'MEGAMOE_{key}_OK\n' * 6 for key in counts))
            plan = dict(ep=2, calls_per_arm=100, warmups_per_arm=10, routing='balanced',
                        entry_protocol='synchronized', control='workers', candidate='rows',
                        pairs=['aa','ab','bb'], runs=runs, expected_correctness=counts)
            (root / 'scope.json').write_text(json.dumps(plan))
            command = [sys.executable, '-B', str(Path(__file__).resolve().parents[1] /
                       'codex_scripts/analyze_megamoe_throughput.py'), 'review-interleaved', str(root)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads((root / 'analysis/review.json').read_text())
            self.assertAlmostEqual(result['geometric_speedup']['geometric_mean_speedup'], 2.)
            self.assertAlmostEqual(result['geometric_speedups_by_pair']['aa']['geometric_mean_speedup'], .2)
            plan['runs'] = runs[:-1]
            (root / 'scope.json').write_text(json.dumps(plan))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_timeline_windows_keep_tail_and_reject_missing_samples(self):
        from copy import deepcopy
        metrics = ('producer_end_us', 'selector_start_us', 'selector_end_us',
                   'megamoe_start_us', 'megamoe_end_us', 'kernel_us')
        rows = [dict.fromkeys(metrics, value) for value in (1., 2., 30.)]
        trace = dict(rank_count=2, samples_per_rank=3, means=dict.fromkeys(metrics, 11.),
                     per_rank={str(rank): dict(records=deepcopy(rows), inter_call_gaps_us=[2., 4.])
                               for rank in range(2)})
        result = timeline_windows(trace, 2, 3, 2)
        self.assertEqual(result['overall']['kernel_us']['mean'], 11.)
        self.assertEqual(result['overall']['kernel_us']['p95'], 30.)
        self.assertEqual([w['calls_per_rank'] for w in result['windows']], [2, 1])
        self.assertEqual(result['windows'][1]['metrics']['kernel_us']['samples'], 2)
        self.assertEqual(result['windows'][1]['metrics']['kernel_us']['mean'], 30.)
        self.assertEqual(result['windows'][1]['metrics']['following_gap_us'], dict(samples=0))
        self.assertEqual(result['overall']['following_gap_us']['mean'], 3.)
        for mutate in (lambda v: v['per_rank'].pop('1'),
                       lambda v: v['per_rank']['0']['records'].pop(),
                       lambda v: v['per_rank']['0']['inter_call_gaps_us'].pop(),
                       lambda v: v['means'].update(kernel_us=10.),
                       lambda v: v['per_rank']['0']['records'][0].update(kernel_us=float('nan'))):
            broken = deepcopy(trace)
            mutate(broken)
            with self.assertRaises(ValueError):
                timeline_windows(broken, 2, 3, 2)

    def test_validation_helper_renames_preserve_bodies_and_call_targets(self):
        prefix = '$MEGAMOE_ENTRY$_ZN9deep_gemm20validate_window_rowsI'
        a, b, c, d = [prefix + suffix for suffix in ('OldA', 'OldB', 'NewA', 'NewB')]
        before = ['MEGAMOE_ENTRY', a, b]
        after = ['MEGAMOE_ENTRY', c, d]
        old_ops = [('0000', f'CALL.REL.NOINC `({a}) ;'),
                   ('0010', f'CALL.REL.NOINC `({b}) ;'), ('0020', 'MOV R1, R2 ;')]
        new_ops = [('0000', f'CALL.REL.NOINC `({c}) ;'),
                   ('0010', f'CALL.REL.NOINC `({d}) ;'), old_ops[2]]
        names, ops, renames = normalize_validation_helper_symbols(before, after, new_ops)
        self.assertEqual((names, ops, renames), (before, old_ops, {c: a, d: b}))
        changed_body = new_ops[:2] + [('0020', 'MOV R1, R3 ;')]
        changed_target = [('0000', f'CALL.REL.NOINC `({d}) ;')] + new_ops[1:]
        for changed in (changed_body, changed_target):
            self.assertNotEqual(normalize_validation_helper_symbols(before, after, changed)[1], old_ops)
        non_call = [('0000', f'MOV R1, `({c}) ;')]
        self.assertEqual(normalize_validation_helper_symbols(before, after, non_call)[1], non_call)
        for invalid in (['MEGAMOE_ENTRY', c], ['MEGAMOE_ENTRY', c, c], ['MEGAMOE_ENTRY', c, 'vprintf']):
            with self.assertRaises(ValueError):
                normalize_validation_helper_symbols(before, invalid, new_ops)
        # Swapped symbol spellings are mapped simultaneously, preserving targets.
        swapped = [op.replace(a, 'TEMP').replace(b, a).replace('TEMP', b) for _, op in old_ops]
        self.assertEqual(normalize_validation_helper_symbols(before, ['MEGAMOE_ENTRY', b, a],
            [(address, op) for (address, _), op in zip(old_ops, swapped)])[1], old_ops)

    def test_diagnostic_syscalls_require_metadata_and_terminal_unchanged_path(self):
        metadata = ('//----- nvinfo : EIATTR_EXTERNS\n.word index@(vprintf)\n'
                    '//----- nvinfo : EIATTR_SYSCALL_OFFSETS\n.word 0x30\n'
                    '//----- nvinfo : NEXT\n')
        ops = [('0000', '@P0 BRA `(.L_retry) ;'),
               ('0010', 'MOV R19, 0x0 ;'),
               ('0020', 'LDC.64 R2, c[0x4][R19] ;'),
               ('0030', 'CALL.ABS.NOINC R2 ;'),
               ('0040', 'BPT.TRAP 0x1 ;')]
        proven = diagnostic_syscall_blocks(metadata, ops)
        self.assertEqual(proven, {'0030': ops})
        for invalid in (metadata.replace('vprintf', 'unknown'),
                        metadata.replace('0x30', '0x40'), ''):
            self.assertEqual(diagnostic_syscall_blocks(invalid, ops), {})
        for ending in ('EXIT ;', '@P0 BPT.TRAP 0x1 ;', 'BRA `(.L_live) ;'):
            self.assertEqual(diagnostic_syscall_blocks(metadata, ops[:-1] + [('0040', ending)]), {})
        changed = ops[:1] + [('0010', 'MOV R19, 0x8 ;')] + ops[2:]
        self.assertNotEqual(diagnostic_syscall_blocks(metadata, changed), proven)

    def test_aggregate_fixed_scope_keeps_losers_and_excludes_extra_controls(self):
        def spec(ep, total):
            return dict(model='model', ep=ep, total=total, case_id=f'ep{ep}_tokens{total}')
        a, b, c, context, extra = [spec(*x) for x in ((2, 2), (2, 4), (4, 4), (4, 8), (2, 6))]
        scope = dict(case_count=3, cases=[a, b, c], context_cases=[context],
                     retained_focus_controls=[a, extra])
        summaries = {ep: dict(status='PASS', ep=ep, matched_collective_preparation=True,
                              cases={}) for ep in (2, 4)}
        for row, ratio in zip((a, b, c, context, extra), (4, 1, .25, .01, 100)):
            key = 'model/' + row['case_id']
            summaries[row['ep']]['cases'][key] = dict(rank_count=row['ep'], samples_per_rank_pass=1000,
                paired_means=dict(candidate=dict(kernel_us=1, megamoe_end_us=2),
                                  pull=dict(kernel_us=ratio, megamoe_end_us=2 * ratio)))
        result = aggregate_summaries(summaries, scope, ['candidate'], ['pull'])
        metric = result['scores']['comparison']['comparisons']['candidate_vs_pull']['kernel_us']
        self.assertEqual(result['unique_cases'], 5)
        self.assertAlmostEqual(metric['geometric_mean_speedup'], 1)
        self.assertEqual(metric['regressions'], {'model/ep4_tokens4': .25})
        del summaries[4]['cases']['model/ep4_tokens4']
        with self.assertRaisesRegex(ValueError, 'Missing scope cases'):
            aggregate_summaries(summaries, scope, ['candidate'], ['pull'])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'aggregate.json'
            output.write_text(json.dumps(result))
            with self.assertRaises(FileNotFoundError):
                aggregate([temp, '--scope', str(Path(temp) / 'missing.json'), '--output', str(output)])
            self.assertEqual(json.loads(output.read_text())['status'], 'INCOMPLETE')

    def test_cleanup_metadata_normalization_preserves_operands_and_callees(self):
        with tempfile.TemporaryDirectory() as temp:
            header = Path(temp) / 'kernel.cuh'
            header.write_text('\n\nDG_DEVICE_ASSERT(false and "timeout");\n')
            kernel = '_ZN9deep_gemm27sm100_fp8_fp4_mega_moe_implBefore'
            sass = (f'.type {kernel},@function\n'
                    '/*0000*/ IMAD.MOV.U32 R10, RZ, RZ, 0x3 ;\n'
                    '/*0010*/ STL [R1+0x8], R10 ;\n'
                    '/*0020*/ CALL.ABS.NOINC R2 ;\n'
                    '/*0030*/ BPT.TRAP 0x1 ;\n'
                    f'/*0040*/ CALL.REL.NOINC `(${kernel}$_calleeA) ;\n')
            before = preparation_instructions(sass, header)
            header.write_text('\n\n\n\nDG_DEVICE_ASSERT(false and "timeout");\n')
            changed = sass.replace('Before', 'After').replace('0x3 ;', '0x5 ;')
            after = preparation_instructions(changed, header)
            self.assertEqual(before[:2], after[:2])
            self.assertNotEqual(after[1], preparation_instructions(changed.replace('R2 ;', 'R3 ;'), header)[1])
            self.assertNotEqual(after[1], preparation_instructions(changed.replace('_calleeA', '_calleeB'), header)[1])
            self.assertNotEqual(after[1], preparation_instructions(changed.replace('[R1+0x8]', '[R1+0xc]'), header)[1])

    def test_cleanup_comparison_rejects_active_options_and_changed_instructions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before, after, output = [root / name for name in ('before', 'after', 'review')]
            for path in (before, after, output):
                path.mkdir()
            def write_manifest(path, config):
                (path / 'manifest.json').write_text(json.dumps(dict(
                    configurations=[dict(case='ep8_tokens32')], specializations=[dict(
                        cases=['ep8_tokens32'], config=config, target=str(path))])))
            write_manifest(before, 'Config(block_m=16, removed_option=0)')
            write_manifest(after, 'Config(block_m=16)')
            with patch('review_megamoe_dispatch_sass.subprocess.check_output',
                       return_value=' /*0000*/ MOV R1, R2;\n') as disassemble:
                compare_preparations([(before, after)], ['removed_option'], 'nvdisasm', output)
                review = json.loads((output / 'preservation_review.json').read_text())
                self.assertEqual(review['specializations'], 1)
                write_manifest(before, 'Config(block_m=16, removed_option=1)')
                with self.assertRaisesRegex(ValueError, 'Removed option was active'):
                    compare_preparations([(before, after)], ['removed_option'], 'nvdisasm', output)
                write_manifest(before, 'Config(block_m=16, removed_option=0)')
                disassemble.side_effect = [' /*0000*/ MOV R1, R2;\n', ' /*0000*/ MOV R1, R3;\n']
                with self.assertRaisesRegex(ValueError, 'Changed compiled instructions'):
                    compare_preparations([(before, after)], ['removed_option'], 'nvdisasm', output)
                self.assertEqual(json.loads((output / 'preservation_review.json').read_text())['status'],
                                 'INCOMPLETE')

    def test_mma_probe_requires_all_phases_and_k_waits(self):
        names = []
        for phase, count in (("FC1", 2), ("FC2", 1)):
            names += ["MegaMmaProbe" + phase, "MegaMmaReuseWait", "MegaMmaReuseReady"]
            names += ["MegaMmaOperandWait", "MegaMmaOperandReady"] * count
        markers = [dict(name=name, location=dict(ctaId=[0, 0, 0], warpId=6), time_us=i)
                   for i, name in enumerate(names)]
        args = dict(leader_ctas=[0], mma_warp=6, fc1_k_blocks=2,
                    fc2_k_blocks=1, kernel_end_us=20)
        result = decode_mma_wait_probe(markers, **args)
        self.assertEqual([x['operand_wait_sum_us'] for x in result], [2, 1])
        self.assertEqual([x['reuse_wait_us'] for x in result], [1, 1])
        for bad in (markers[:-1], markers[1:], markers + [markers[-1]],
                    [dict(m, time_us=100) if i == 3 else m for i, m in enumerate(markers)]):
            with self.assertRaises(ValueError):
                decode_mma_wait_probe(bad, **args)
        for changed in (dict(leader_ctas=[0, 2]), dict(mma_warp=5),
                        dict(fc1_k_blocks=1), dict(fc2_k_blocks=2)):
            with self.assertRaises(ValueError):
                decode_mma_wait_probe(markers, **dict(args, **changed))

    def test_ncu_listener_requires_owned_socket_despite_inaccessible_child_fd(self):
        proc = self.root / 'proc'
        for pid in (101, 102):
            (proc / str(pid) / 'fd').mkdir(parents=True)
            (proc / str(pid) / 'task' / str(pid)).mkdir(parents=True)
            (proc / str(pid) / 'task' / str(pid) / 'children').write_text(
                '102' if pid == 101 else '')
        for fd in ('0', '1'):
            (proc / '102/fd' / fd).touch()
        (proc / 'net').mkdir()
        tcp = proc / 'net/tcp'

        def readlink(fd):
            if fd.name == '0':
                raise PermissionError('child changed dumpability')
            return 'socket:[535]'

        with patch.object(ncu_tcp_launcher, 'Path', side_effect=lambda s: self.root / s.lstrip('/')), \
                patch.object(ncu_tcp_launcher.os, 'readlink', side_effect=readlink):
            for inode, expected in ((535, True), (999, False)):
                tcp.write_text(f'header\n0: 00000000:C041 00000000:0000 0A 0 0 0 0 0 {inode}\n')
                self.assertEqual(ncu_tcp_launcher.owns_listener(101, 49217), expected)

    def test_readiness_decoder_tracks_issuer_and_receiver_roles(self):
        for unified, dedicated, narrow in (
                (u, d, n) for u, d in ((False, False), (True, False), (True, True))
                for n in range(4)):
            issuer = 4 if unified else 0
            validator = 0 if dedicated else 8
            middle = validator + (narrow or (0 if dedicated else 4))
            def marker(name, cta, warp, time):
                return dict(name=name, location=dict(ctaId=[cta, 0, 0], warpId=warp), time_us=time)
            markers = [
                marker("GenphaseWindowValidateStart", 1, validator, 1),
                marker("GenphaseWindowPayloadIssue", 1, validator, 2),
                marker("GenphaseWindowValidateStart", 1, validator, 3),
                marker("GenphaseWindowPayloadIssue", 1, validator, 4),
                marker("GenphaseRemainderValidateStart", 0, middle, 1),
                marker("GenphaseMiddlePayloadIssue", 0, middle, 3),
                marker("MegaWindowWaitStart", 0, issuer, 2),
                marker("MegaWindowReady", 0, issuer, 4),
                marker("MegaWindowWaitStart", 0, issuer, 5),
                marker("MegaWindowReady", 0, issuer, 6),
            ]
            args = dict(hidden=4096, num_sms=2, validation_ctas=1, index_ctas=1,
                        active_issuer_ctas=[0], unified_tma=unified,
                        dedicated_dispatch=dedicated, narrow_sf=narrow)
            _, intervals = decode_readiness_windows(markers, **args)
            self.assertEqual(intervals["middle_issuer_wait"][0]["warp"], issuer)
            self.assertEqual(intervals["tail_validation"][0]["warp"], validator)
            self.assertEqual(intervals["middle_validation"][0]["warp"], middle)
            _, shorter = decode_readiness_windows(markers, **dict(args, first_end=512, middle_end=3072))
            self.assertEqual(shorter["byte_windows"], [[0, 512], [512, 3072], [3072, 4096]])
            for invalid_end in (0, 256, 768, 1536):
                with self.assertRaises(ValueError):
                    decode_readiness_windows(markers, **dict(args, first_end=invalid_end))
            with self.assertRaises(ValueError):
                decode_readiness_windows(markers, **dict(args, unified_tma=not unified))
            with self.assertRaises(ValueError):
                decode_readiness_windows(markers[:-1], **args)
            with self.assertRaises(ValueError):
                decode_readiness_windows(markers, **dict(args, narrow_sf=0 if narrow else 1))

    def test_readiness_decoder_preserves_unrelated_repeated_probes(self):
        def marker(name, cta, warp, time):
            return dict(name=name, location=dict(ctaId=[cta, 0, 0], warpId=warp), time_us=time)
        markers = [marker("GenphaseWindowValidateStart", 1, 8, 1),
                   marker("GenphaseWindowPayloadIssue", 1, 8, 2),
                   marker("GenphaseRemainderValidateStart", 0, 12, 1),
                   marker("GenphaseMiddlePayloadIssue", 0, 12, 3),
                   marker("MegaWindowWaitStart", 0, 0, 2),
                   marker("MegaWindowReady", 0, 0, 4),
                   marker("MegaLocalReduceEnd", 0, 12, 5),
                   marker("MegaLocalReduceEnd", 0, 12, 6)]
        args = dict(hidden=1536, num_sms=2, validation_ctas=1, index_ctas=1, active_issuer_ctas=[0])
        decoded, _ = decode_readiness_windows(markers, **args)
        self.assertEqual(decoded[-2:], markers[-2:])
        for invalid in (markers + [markers[0]], markers + [markers[3]], markers[1:]):
            with self.assertRaises(ValueError):
                decode_readiness_windows(invalid, **args)

    def test_correctness_audit_preserves_rank_order_across_delayed_stdout(self):
        def result(rank, label):
            return f"MEGAMOE_RESULT_COMPARE_OK case={label} rank={rank}\n"
        def replay(rank, rows):
            return f"MEGAMOE_REPLAY_EXACT_OK rank={rank} rows={rows} scope=pipeline\n"
        first = result(0, "first") + replay(0, 32)
        second = result(0, "second") + replay(0, 1024)
        delayed = result(1, "first") + replay(1, 32)
        end = result(1, "second") + replay(1, 1024)
        expected = [("compare", "first"), ("replay", "32/pipeline"),
                    ("compare", "second"), ("replay", "1024/pipeline")]
        log = first + "SWEEP_CASE_START case=second\n" + second + delayed + end
        self.assertEqual(audit_correctness(log, expected, 2), dict(record=0, compare=4, replay=4))
        for invalid in (log.replace(delayed, ""), log + delayed,
                        first + second + end + delayed, log + replay(2, 32)):
            with self.assertRaises(ValueError):
                audit_correctness(invalid, expected, 2)

    def test_joined_preparation_requires_exact_nodes_stream_and_metadata_order(self):
        def row(start, end, name, stream=7):
            return (start, end, name, 1, 1, stream)
        producer, selector = row(100, 120, "producer"), row(110, 140, "selector")
        preparation = [row(0, 20, "decomposition_collective_barrier("),
                       row(19, 40, "at::native::FillFunctor<int>"),
                       row(39, 60, "at::native::FillFunctor<int>"),
                       row(59, 105, "decomposition_collective_barrier(")]
        # Adjacent Nsight envelopes may overlap; metadata must follow preparation.
        self.assertEqual(audit_joined_preparation(preparation, producer, selector), preparation)
        invalid = [preparation[:-1], preparation + [row(70, 80, "unexpected")],
                   [*preparation[:3], row(59, 111, "decomposition_collective_barrier(")],
                   [*preparation[:3], row(59, 105, "decomposition_collective_barrier(", 8)],
                   [preparation[0], row(19, 40, "unexpected"), *preparation[2:]]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit_joined_preparation(rows, producer, selector)
        barriers = [preparation[0], preparation[-1]]
        self.assertEqual(audit_joined_preparation(barriers, producer, selector,
                                                 barrier_only=True), barriers)
        # A pull prefix cannot silently satisfy an all-push reset contract.
        with self.assertRaises(ValueError):
            audit_joined_preparation(barriers, producer, selector)
        with self.assertRaises(ValueError):
            audit_joined_preparation(preparation, producer, selector, barrier_only=True)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def invoke(self, module, *args):
        with patch.object(sys, 'argv', [module.__file__, *map(str, args)]), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_submit_records_both_formats_and_keeps_dry_run_job_free(self):
        for kind, dry, output in [('sbatch', False, '123;cluster\n'),
                                  ('afd', False, 'submitted job=456 run_dir=/work/case\n'),
                                  ('afd', True, 'dry run complete\n')]:
            with self.subTest(kind=kind, dry=dry):
                root = self.root / (kind + str(dry))
                root.mkdir()
                env = dict(OUTPUT_ROOT='/work/plan', FASTAFD_TASK_ROOT='/work/plan',
                           FASTAFD_CUDA_EXTRACT_TEMP_ROOT='/work/extract',
                           FASTAFD_SOURCE_REPO='/work/source')
                (root / 'env.json').write_text(json.dumps(env))
                command = ['sbatch', '--parsable', 'run.sbatch'] if kind == 'sbatch' else ['bash', 'run.sh']
                (root / ('submit.json' if kind == 'sbatch' else 'launch.json')).write_text(json.dumps(command))
                with patch.object(submit_plan.subprocess, 'run',
                                  return_value=subprocess.CompletedProcess(
                                      [], 0, output, 'scheduler warning\n')) as run:
                    self.invoke(submit_plan, kind, root, *(['--dry-run'] if dry else []))
                sent = json.loads(next(call.kwargs['input'] for call in run.call_args_list
                                       if 'input' in call.kwargs))
                self.assertEqual(sent['cmd'], command)
                self.assertEqual((root / 'job.json').exists(), not dry)
                if kind == 'afd':
                    self.assertEqual(sent['env']['FASTAFD_DRY_RUN'], '1' if dry else '0')
                if not dry:
                    self.assertEqual(run.call_args.args[0],
                                     ['rsync', '-a', str(root / 'job.json'),
                                      'oci-hsg:/work/plan/job.json'])
                    record = json.loads((root / 'job.json').read_text())
                    self.assertEqual(record['job'], '123' if kind == 'sbatch' else '456')
                    with patch.object(submit_plan.subprocess, 'run') as run:
                        with self.assertRaises(ValueError):
                            self.invoke(submit_plan, kind, root)
                        run.assert_not_called()

    def test_invalid_sbatch_plan_never_contacts_remote(self):
        (self.root / 'env.json').write_text(json.dumps({'OUTPUT_ROOT': '/work/plan'}))
        (self.root / 'submit.json').write_text(json.dumps(['bash', 'run.sh']))
        with patch.object(submit_plan.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                self.invoke(submit_plan, 'sbatch', self.root)
            run.assert_not_called()

    def test_deepgemm_links_idempotently_and_refuses_unrelated_library(self):
        donor = self.root / '_minisgl_deepgemm.test.so'
        donor.write_bytes(b'fixture')
        cache = self.root / 'cache'
        args = (donor, '--cache-root', cache, '--output', self.root / 'seed.json')
        self.invoke(seed_native_cache, *args)
        target = cache / 'lib' / donor.name
        self.assertTrue(target.samefile(donor))
        self.invoke(seed_native_cache, *args)
        target.unlink()
        target.write_bytes(b'unrelated')
        with self.assertRaises(RuntimeError):
            self.invoke(seed_native_cache, *args)
        self.assertEqual(target.read_bytes(), b'unrelated')

    def test_tvm_scopes_modules_and_refuses_existing_target(self):
        donor = self.root / 'donor'
        module = donor / 'module'
        module.mkdir(parents=True)
        (module / 'native.so').write_bytes(b'fixture')
        (module / 'build.ninja').write_text('fixture')
        cache = self.root / 'cache'
        args = (donor, '--kind', 'tvm', '--cache-root', cache, '--node-ip', '192.0.2.1',
                '--rank-count', '2', '--output', self.root / 'seed.json')
        self.invoke(seed_native_cache, *args)
        node = cache / 'ray-node-192_0_2_1'
        for root in [node, node / 'rank1', node / 'rank2']:
            self.assertTrue((root / 'module').samefile(module))
        with self.assertRaises(ValueError):
            self.invoke(seed_native_cache, *args)


if __name__ == '__main__':
    unittest.main()
