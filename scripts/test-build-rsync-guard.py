#!/usr/bin/env python3
"""CPU regressions for the rsync --delete destination guard in the build script.

build-dspark-vllm-runtime.sh mirrors the local checkout to the worker with
`rsync -az --delete` into ${WORKER_CHECKOUT:-…}: a wrong or stale
WORKER_CHECKOUT (typo, a directory used for something else) deleted whatever
lived there, with no sanity check. The build now refuses to sync into a
remote directory that is non-empty AND not a DSpark recipe checkout (no
docker-compose.dspark.yml).

These tests extract the shipped guard command and run it against the four
destination states.
"""
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-dspark-vllm-runtime.sh"
SOURCE = BUILD.read_text()

_m = re.search(r'ssh "\$host" "(if \[ -d .*fi)"\s*$', SOURCE, re.M)
assert _m, "rsync guard command not found in build script"
# The command is written for an outer double-quoted context; unescape to the
# form the remote bash receives.
GUARD = _m.group(1).replace('\\"', '"').replace("\\$", "$")


def run_guard(checkout: Path) -> subprocess.CompletedProcess:
    # In production the guard runs inside an outer double-quoted ssh argument,
    # so $checkout expands before the remote bash sees it (the single quotes
    # then quote the concrete path). Reproduce that expansion here.
    cmd = GUARD.replace("$checkout", shlex.quote(str(checkout)))
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)


class GuardBehavior(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workdir)

    def test_missing_directory_allowed(self):
        r = run_guard(self.workdir / "does-not-exist")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_empty_directory_allowed(self):
        empty = self.workdir / "empty"
        empty.mkdir()
        r = run_guard(empty)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_recipe_checkout_allowed(self):
        checkout = self.workdir / "checkout"
        checkout.mkdir()
        (checkout / "docker-compose.dspark.yml").write_text("services: {}\n")
        (checkout / "README.md").write_text("existing checkout\n")
        r = run_guard(checkout)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_foreign_non_empty_directory_refused(self):
        foreign = self.workdir / "foreign"
        foreign.mkdir()
        (foreign / "important-data.txt").write_text("do not delete\n")
        r = run_guard(foreign)
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to rsync --delete into", r.stderr)
        self.assertIn(str(foreign), r.stderr)
        # The refusal must not modify the directory.
        self.assertTrue((foreign / "important-data.txt").exists())


class SourceShape(unittest.TestCase):
    def test_guard_precedes_rsync(self):
        self.assertLess(SOURCE.index("refusing to rsync --delete into"),
                        SOURCE.index("rsync -az --delete"))

    def test_rsync_delete_preserved(self):
        self.assertIn('rsync -az --delete "$SCRIPT_DIR/" "$host:$checkout/"', SOURCE)

    def test_guard_marks_recipe_checkout(self):
        self.assertIn("$checkout/docker-compose.dspark.yml", GUARD)


if __name__ == "__main__":
    unittest.main()
