"""
Integration tests for MiniMaxModel.

These tests verify end-to-end behaviour against the live MiniMax API.
They are skipped when the ``MINIMAX_API_KEY`` environment variable is not set.

Run with:
    MINIMAX_API_KEY=your_key python -m pytest test/models/test_minimaxmodel_integration.py -v
"""

import os
import unittest

from easyjailbreak.models.minimax_model import MiniMaxModel
from easyjailbreak.models.openai_model import OpenaiModel

_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
_SKIP_REASON = 'MINIMAX_API_KEY not set'


@unittest.skipUnless(_API_KEY, _SKIP_REASON)
class TestMiniMaxModelIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def test_generate_simple_prompt(self):
        """Model should return a non-empty response to a simple prompt."""
        model = MiniMaxModel(api_keys=_API_KEY)
        response = model.generate('What is 2 + 2? Reply with just the number.')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertIn('4', response)

    def test_generate_with_generation_config(self):
        """generation_config parameters are forwarded to the API."""
        model = MiniMaxModel(
            api_keys=_API_KEY,
            generation_config={'temperature': 0.5, 'max_tokens': 256},
        )
        response = model.generate('Say hello in one word.')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_batch_generate(self):
        """batch_generate returns one response per conversation."""
        model = MiniMaxModel(api_keys=_API_KEY, generation_config={'max_tokens': 256})
        responses = model.batch_generate([
            ['Say yes.'],
            ['Say no.'],
        ])
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(isinstance(r, str) and len(r) > 0 for r in responses))

    def test_isinstance_openai_model(self):
        """Live model instance passes OpenaiModel isinstance check."""
        model = MiniMaxModel(api_keys=_API_KEY)
        self.assertIsInstance(model, OpenaiModel)

    def test_highspeed_model(self):
        """MiniMax-M2.7-highspeed variant should work."""
        model = MiniMaxModel(
            api_keys=_API_KEY,
            model_name='MiniMax-M2.7-highspeed',
            generation_config={'max_tokens': 256},
        )
        response = model.generate('Say OK.')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == '__main__':
    unittest.main()
