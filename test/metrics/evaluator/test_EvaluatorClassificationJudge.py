import unittest
from unittest.mock import MagicMock, patch

from easyjailbreak.datasets import Instance
from easyjailbreak.metrics.Evaluator import EvaluatorClassificatonJudge
from easyjailbreak.models import WhiteBoxModelBase

class TestEvaluatorClassificatonJudge(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.mock_model = WhiteBoxModelBase(self.model, self.tokenizer)
        self.evaluator = EvaluatorClassificatonJudge(self.mock_model)


    def test_init(self):
        # Test initialization
        self.assertIsInstance(self.evaluator.model, MagicMock)
        self.assertIsInstance(self.evaluator.tokenizer, MagicMock)
        self.assertEqual(self.evaluator.prompt_pattern, "{response}")


    def test_evaluate(self):
        # Test the evaluation process
        instance = Instance(target_responses=["response"])
        self.evaluator.judge = MagicMock(return_value=True)
        self.evaluator._evaluate(instance)
        self.assertEqual(len(instance.eval_results), 1)

    def test_format(self):
        # Test the format method
        instance = Instance(response="test response")
        formatted_text = self.evaluator._format(instance)
        self.assertEqual(formatted_text, "test response")

if __name__ == '__main__':
    unittest.main()
