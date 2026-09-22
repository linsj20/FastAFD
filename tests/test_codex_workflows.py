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
from summarize_megamoe_router_timeline import audit_joined_preparation


class CodexWorkflowTests(unittest.TestCase):
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
                sent = json.loads(run.call_args.kwargs['input'])
                self.assertEqual(sent['cmd'], command)
                self.assertEqual((root / 'job.json').exists(), not dry)
                if kind == 'afd':
                    self.assertEqual(sent['env']['FASTAFD_DRY_RUN'], '1' if dry else '0')
                if not dry:
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
