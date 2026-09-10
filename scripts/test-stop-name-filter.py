#!/usr/bin/env python3
"""CPU regressions for the anchored container-name filters in the stop script.

docker's `--filter name=` takes an *unanchored* regular expression, so the old
`name=${project}-vllm-dspark` patterns matched (and `docker rm -f`'d) any
container whose name merely *contained* the string. The stop script now routes
every name filter through project_name_filters(), which escapes regex
metacharacters in the project name and anchors the pattern to the compose
container-name shape <project>[-_]<service>([-_]<index>)?.

These tests extract the shipped helper from the stop script, have bash produce
the two filter regexes per project, and evaluate them against a match/reject
matrix with Python re (equivalent to Go's RE2 for this construct subset).
"""
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT / "stop-deepseek-v4-flash-dspark.sh"
SOURCE = STOP.read_text()

_fn_start = SOURCE.index("project_name_filters() {")
HELPER_FN = SOURCE[_fn_start:SOURCE.index("\n}", _fn_start) + 2]


def filters_for(project: str) -> tuple[str, str]:
    script = f"""set -euo pipefail
{HELPER_FN}
project_name_filters {project!r}
printf '%s\\n' "$RANK_NAME_RE"
printf '%s\\n' "$SIDECAR_NAME_RE"
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rank_re, sidecar_re = result.stdout.splitlines()
    return rank_re, sidecar_re


class HelperShape(unittest.TestCase):
    def test_helper_is_defined_before_first_use(self):
        self.assertLess(SOURCE.index("project_name_filters() {"),
                        SOURCE.index('project_name_filters "$project"'))

    def test_all_filter_callers_anchored(self):
        # Every docker --filter name= site must consume the anchored variables;
        # the unanchored ${project}-<service> form must not come back. Comments
        # (including the helper's own explanation of the old form) are ignored.
        code = "\n".join(line for line in SOURCE.splitlines()
                         if not line.lstrip().startswith("#"))
        self.assertNotIn("name=${project}-", code)
        self.assertEqual(SOURCE.count("name=${RANK_NAME_RE}"), 5)
        self.assertEqual(SOURCE.count("name=${SIDECAR_NAME_RE}"), 3)
        self.assertEqual(SOURCE.count('project_name_filters "$project"'), 5)


class RankFilterMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rank_re, cls.sidecar_re = filters_for("deepseek-v4-flash")

    def assert_match(self, pattern, name, want):
        self.assertEqual(re.search(pattern, name) is not None, want, f"{name!r} vs {pattern!r}")

    def test_rank_matches_own_containers(self):
        for name in ("deepseek-v4-flash-vllm-dspark-1",
                     "deepseek-v4-flash-vllm-dspark-12",
                     "deepseek-v4-flash_vllm-dspark_1",   # compose v1 naming
                     "deepseek-v4-flash-vllm-dspark"):    # bare manual name
            with self.subTest(name=name):
                self.assert_match(self.rank_re, name, True)

    def test_rank_rejects_foreign_containers(self):
        for name in ("deepseek-v4-flash-vllm-dspark-old",       # suffix overmatch
                     "deepseek-v4-flash-vllm-dspark-1-backup",
                     "deepseek-v4-flash-vllm-dspark-1a",
                     "deepseek-v4-flash-vllm-dspark-1-2",
                     "my-deepseek-v4-flash-vllm-dspark-1",      # prefix overmatch
                     "deepseek-v4-flash-vllm-dspark2-1",
                     "deepseek-v4-flash-vl-sidecar-1",          # other service
                     "deepseek-v4-flash-xvllm-dspark-1",
                     "deepseek-v4-flash-vllm-dsparkx-1"):
            with self.subTest(name=name):
                self.assert_match(self.rank_re, name, False)

    def test_sidecar_matches_own_containers(self):
        for name in ("deepseek-v4-flash-vl-sidecar-1",
                     "deepseek-v4-flash_vl-sidecar_1",
                     "deepseek-v4-flash-vl-sidecar"):
            with self.subTest(name=name):
                self.assert_match(self.sidecar_re, name, True)

    def test_sidecar_rejects_foreign_containers(self):
        for name in ("deepseek-v4-flash-vl-sidecar-old",
                     "deepseek-v4-flash-vllm-dspark-1",
                     "my-deepseek-v4-flash-vl-sidecar-1"):
            with self.subTest(name=name):
                self.assert_match(self.sidecar_re, name, False)


class MetacharProjectNames(unittest.TestCase):
    def test_dotted_project_name_is_literal(self):
        rank_re, sidecar_re = filters_for("deepseek.v4.foo")
        self.assertIsNotNone(re.search(rank_re, "deepseek.v4.foo-vllm-dspark-1"))
        self.assertIsNotNone(re.search(sidecar_re, "deepseek.v4.foo-vl-sidecar-1"))
        # Without escaping, the dots would act as wildcards and match these.
        self.assertIsNone(re.search(rank_re, "deepseekXv4Xfoo-vllm-dspark-1"))
        self.assertIsNone(re.search(rank_re, "deepseek-v4.foo-vllm-dspark-1"))

    def test_plus_and_brackets_are_literal(self):
        rank_re, _ = filters_for("ab+c[d]")
        self.assertIsNotNone(re.search(rank_re, "ab+c[d]-vllm-dspark-1"))
        self.assertIsNone(re.search(rank_re, "abbbbc[d]-vllm-dspark-1"))
        self.assertIsNone(re.search(rank_re, "ab+cd-vllm-dspark-1"))

    def test_filter_is_valid_for_go_regexp_constructs(self):
        # The pattern must only use constructs valid in both Python re and
        # Go's RE2: anchored ^...$, [-_], ([-_][0-9]+)?, and \\-escaped
        # punctuation. Guards against accidentally emitting pythonisms later.
        rank_re, sidecar_re = filters_for("deepseek-v4-flash")
        for pattern in (rank_re, sidecar_re):
            with self.subTest(pattern=pattern):
                self.assertTrue(pattern.startswith("^"), pattern)
                self.assertTrue(pattern.endswith("$"), pattern)
                self.assertNotIn("\\-", pattern)  # RE2 rejects \-


class ForceRmAssembly(unittest.TestCase):
    """The heredoc-built remote command in force_rm_project_containers must
    deliver the anchored filters to docker unmangled (end-anchor `$` intact,
    no premature expansion) on both the local `bash -c` and ssh paths."""

    def run_force_rm(self, project: str) -> list[str]:
        fn_start = SOURCE.index("force_rm_project_containers() {")
        fn = SOURCE[fn_start:SOURCE.index("\n}\n", fn_start) + 3]
        workdir = Path(tempfile.mkdtemp())
        log = workdir / "docker.log"
        docker = workdir / "docker"
        docker.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$@" >> "$DOCKER_LOG"\n'
            "exit 0\n"
        )
        docker.chmod(0o755)
        script = f"""set -euo pipefail
export DOCKER_LOG={shlex.quote(str(log))}
export PATH={shlex.quote(str(workdir))}:$PATH
stop_warn() {{ echo "warn: $*" >&2; }}
WORKER_REACHABLE=0
WORKER2_HOST=
{HELPER_FN}
{fn}
force_rm_project_containers {shlex.quote(project)} local
"""
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        lines = log.read_text().splitlines() if log.exists() else []
        shutil.rmtree(workdir)
        assert result.returncode == 0, result.stderr
        return lines

    def test_filters_arrive_anchored_and_intact(self):
        args = self.run_force_rm("deepseek-v4-flash")
        self.assertIn("name=^deepseek-v4-flash[-_]vl-sidecar([-_][0-9]+)?$", args)
        self.assertIn("name=^deepseek-v4-flash[-_]vllm-dspark([-_][0-9]+)?$", args)
        # The label filter survives alongside, still exact.
        self.assertIn("label=com.docker.compose.project=deepseek-v4-flash", args)

    def test_metachar_project_arrives_escaped(self):
        args = self.run_force_rm("deepseek.v4.foo")
        self.assertIn(r"name=^deepseek\.v4\.foo[-_]vllm-dspark([-_][0-9]+)?$", args)


if __name__ == "__main__":
    unittest.main()
