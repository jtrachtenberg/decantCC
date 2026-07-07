"""Token-estimate tests. conversion_cost is uncalled during a run (per-answer
cost comes from response usage) but supports a-priori sizing and the raw-PDF /
caching paths, so pin its two branches. Offline."""

import unittest

from decant_eval.models import AnswerResult
from decant_eval.tokens import CHARS_PER_TOKEN, conversion_cost, estimate_tokens


class _CountingClient:
    """Just enough of AnthropicModelClient for the count_tokens branch."""

    def count_input_tokens(self, *, model, system, prompt):
        return 42


class TestTokens(unittest.TestCase):
    def test_estimate_is_chars_over_four(self):
        self.assertEqual(estimate_tokens("a" * (CHARS_PER_TOKEN * 10)), 10)
        self.assertEqual(estimate_tokens(""), 1)  # floor at 1

    def test_conversion_cost_uses_client_when_available(self):
        self.assertEqual(conversion_cost("some text", client=_CountingClient()), 42)

    def test_conversion_cost_falls_back_to_estimate(self):
        text = "x" * 400
        self.assertEqual(conversion_cost(text, client=None), estimate_tokens(text))


class TestAnswerResult(unittest.TestCase):
    def test_shape(self):
        r = AnswerResult(text="hi", input_tokens=3, output_tokens=1)
        self.assertEqual(r.text, "hi")


if __name__ == "__main__":
    unittest.main()
