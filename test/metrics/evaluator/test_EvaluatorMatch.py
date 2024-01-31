import unittest

from easyjailbreak.datasets import Instance
from easyjailbreak.metrics.Evaluator import EvalatorMatch


class TestEvalatorMatch(unittest.TestCase):

    def test_evaluate_with_match(self):
        # Test scenario with matching responses
        instance = Instance(target_responses=["response1", "response2"],
                            reference_responses=["response1", "response3"])
        evaluator = EvalatorMatch()
        evaluator._evaluate(instance)
        self.assertEqual(instance.eval_results, [1, 0])  # Matching first response

    def test_evaluate_without_match(self):
        # Test scenario with no matching responses
        instance = Instance(target_responses=["response4", "response5"],
                            reference_responses=["response1", "response2"])
        evaluator = EvalatorMatch()
        evaluator._evaluate(instance)
        self.assertEqual(instance.eval_results, [0, 0])  # No matches

    def test_evaluate_empty_responses(self):
        # Test scenario with empty responses
        instance = Instance(target_responses=[], reference_responses=["response1", "response2"])
        evaluator = EvalatorMatch()
        evaluator._evaluate(instance)
        self.assertEqual(instance.eval_results, [])  # Empty result due to no target responses

if __name__ == '__main__':
    unittest.main()
