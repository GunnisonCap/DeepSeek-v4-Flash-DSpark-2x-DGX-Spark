#!/usr/bin/env python3
"""CPU regression for bench-patches.sh prompt generation (float-repetition bug).

run_ttft built its prompt as:  python3 -c "print('hello ' * (N * 4 / 3 // 6))"
In Python 3 the `/` yields a float, `'hello ' * <float>` raises TypeError, and
the shell `|| echo "hello world"` fallback silently fired for EVERY labelled
prompt size (256…8192): the whole short-context TTFT section measured the same
2-token prompt. The shipped expression must stay integer arithmetic.
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "bench-patches.sh"
SOURCE = BENCH.read_text()

_m = re.search(r"""prompt=\$\(python3 -c "print\('hello ' \* \(([^)]+)\)\)\"""", SOURCE)
assert _m, "prompt-generation line not found in bench-patches.sh"
EXPR = _m.group(1).strip()


class PromptExpression(unittest.TestCase):
    def test_expression_is_integer_arithmetic(self):
        # No single `/` (float division); `//` floor division is required so
        # str * <int> never raises TypeError on any Python 3.
        self.assertNotRegex(EXPR, r"(?<!/)/(?!/)", EXPR)

    def test_expression_matches_shipped_line(self):
        self.assertIn(f"""python3 -c "print('hello ' * ({EXPR}))\"""", SOURCE)


class PromptGeneration(unittest.TestCase):
    def test_all_bench_sizes_generate_real_prompts(self):
        for n in (256, 512, 1024, 2048, 4096, 8192):
            with self.subTest(n=n):
                expr = EXPR.replace("$prompt_tokens", str(n))
                out = subprocess.run(
                    ["python3", "-c", f"print('hello ' * ({expr}))"],
                    capture_output=True, text=True,
                )
                self.assertEqual(out.returncode, 0, out.stderr)
                words = out.stdout.split()
                expected = eval(expr)  # same arithmetic, computed by the harness
                self.assertEqual(len(words), expected)
                self.assertTrue(all(w == "hello" for w in words))
                # The whole point: the fallback must never be what we send.
                self.assertNotEqual(out.stdout.strip(), "hello world")

    def test_sizes_are_distinguishable(self):
        # Every labelled size must produce a different prompt, or the TTFT
        # table silently collapses rows again.
        lengths = set()
        for n in (256, 512, 1024, 2048, 4096, 8192):
            expr = EXPR.replace("$prompt_tokens", str(n))
            lengths.add(eval(expr))
        self.assertEqual(len(lengths), 6, lengths)


if __name__ == "__main__":
    unittest.main()
