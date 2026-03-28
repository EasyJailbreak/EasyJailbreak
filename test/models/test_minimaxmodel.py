import unittest
from unittest.mock import MagicMock, patch

from easyjailbreak.models.minimax_model import MiniMaxModel
from easyjailbreak.models.openai_model import OpenaiModel


class TestMiniMaxModel(unittest.TestCase):
    """Unit tests for MiniMaxModel."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_default_init(self, mock_openai_cls):
        """Default construction uses MiniMax-M2.7 and the MiniMax base URL."""
        model = MiniMaxModel(api_keys='test-key')
        self.assertEqual(model.model_name, 'MiniMax-M2.7')
        mock_openai_cls.assert_called_once_with(
            api_key='test-key',
            base_url='https://api.minimax.io/v1',
        )

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_custom_model_name(self, mock_openai_cls):
        """Users can choose a different MiniMax model variant."""
        model = MiniMaxModel(api_keys='k', model_name='MiniMax-M2.7-highspeed')
        self.assertEqual(model.model_name, 'MiniMax-M2.7-highspeed')

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_isinstance_openai_model(self, mock_openai_cls):
        """MiniMaxModel must pass isinstance checks for OpenaiModel."""
        model = MiniMaxModel(api_keys='k')
        self.assertIsInstance(model, OpenaiModel)

    # ------------------------------------------------------------------
    # Temperature clamping
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_temperature_clamped_to_min(self, mock_openai_cls):
        """Temperature=0 should be clamped to 0.01."""
        model = MiniMaxModel(api_keys='k', generation_config={'temperature': 0})
        self.assertAlmostEqual(model.generation_config['temperature'], 0.01)

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_temperature_clamped_to_max(self, mock_openai_cls):
        """Temperature above 1 should be clamped to 1.0."""
        model = MiniMaxModel(api_keys='k', generation_config={'temperature': 2.0})
        self.assertAlmostEqual(model.generation_config['temperature'], 1.0)

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_temperature_in_range_unchanged(self, mock_openai_cls):
        """Valid temperature should pass through unchanged."""
        model = MiniMaxModel(api_keys='k', generation_config={'temperature': 0.7})
        self.assertAlmostEqual(model.generation_config['temperature'], 0.7)

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_no_temperature_key(self, mock_openai_cls):
        """Config without temperature should be left as-is."""
        model = MiniMaxModel(api_keys='k', generation_config={'max_tokens': 512})
        self.assertNotIn('temperature', model.generation_config)
        self.assertEqual(model.generation_config['max_tokens'], 512)

    # ------------------------------------------------------------------
    # Think-tag stripping
    # ------------------------------------------------------------------

    def test_strip_think_tags_simple(self):
        """Single <think> block is removed."""
        text = '<think>internal reasoning</think>Hello world'
        self.assertEqual(MiniMaxModel._strip_think_tags(text), 'Hello world')

    def test_strip_think_tags_multiline(self):
        """Multiline <think> blocks are removed."""
        text = '<think>\nstep 1\nstep 2\n</think>\nAnswer: 42'
        self.assertEqual(MiniMaxModel._strip_think_tags(text), 'Answer: 42')

    def test_strip_think_tags_multiple(self):
        """Multiple <think> blocks are all removed."""
        text = '<think>a</think>Hello <think>b</think>world'
        self.assertEqual(MiniMaxModel._strip_think_tags(text), 'Hello world')

    def test_strip_think_tags_no_tags(self):
        """Text without <think> tags is returned unchanged."""
        text = 'Just normal text'
        self.assertEqual(MiniMaxModel._strip_think_tags(text), 'Just normal text')

    def test_strip_think_tags_none(self):
        """None input returns None."""
        self.assertIsNone(MiniMaxModel._strip_think_tags(None))

    # ------------------------------------------------------------------
    # Generate (with think-tag stripping + temp clamping)
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_generate_strips_think_tags(self, mock_openai_cls):
        """generate() should strip <think> tags from the response."""
        model = MiniMaxModel(api_keys='k')

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='<think>reasoning</think>The answer is 42'))
        ]
        model.client.chat.completions.create.return_value = mock_response

        result = model.generate('What is the answer?')
        self.assertEqual(result, 'The answer is 42')

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_generate_clamps_kwarg_temperature(self, mock_openai_cls):
        """generate() should clamp temperature passed as a keyword arg."""
        model = MiniMaxModel(api_keys='k')

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='Hello'))
        ]
        model.client.chat.completions.create.return_value = mock_response

        model.generate('Hi', temperature=0)

        call_kwargs = model.client.chat.completions.create.call_args
        self.assertGreaterEqual(call_kwargs.kwargs.get('temperature', call_kwargs[1].get('temperature', 0.01)), 0.01)

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_generate_plain_text(self, mock_openai_cls):
        """generate() returns plain text when no think tags are present."""
        model = MiniMaxModel(api_keys='k')

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='Plain response'))
        ]
        model.client.chat.completions.create.return_value = mock_response

        result = model.generate('Hello')
        self.assertEqual(result, 'Plain response')

    # ------------------------------------------------------------------
    # Batch generate
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_batch_generate(self, mock_openai_cls):
        """batch_generate() should return stripped results for each conversation."""
        model = MiniMaxModel(api_keys='k')

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='<think>x</think>Answer'))
        ]
        model.client.chat.completions.create.return_value = mock_response

        results = model.batch_generate([['q1'], ['q2']])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r == 'Answer' for r in results))

    # ------------------------------------------------------------------
    # System message
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_set_system_message(self, mock_openai_cls):
        """set_system_message() should work (inherited from OpenaiModel)."""
        model = MiniMaxModel(api_keys='k')
        model.set_system_message('You are a helpful assistant.')
        self.assertEqual(model.conversation.system_message, 'You are a helpful assistant.')

    # ------------------------------------------------------------------
    # Conversation attribute
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_has_conversation_attribute(self, mock_openai_cls):
        """MiniMaxModel must have 'conversation' attribute for attacker compat."""
        model = MiniMaxModel(api_keys='k')
        self.assertTrue(hasattr(model, 'conversation'))

    # ------------------------------------------------------------------
    # Generation config not mutated
    # ------------------------------------------------------------------

    @patch('easyjailbreak.models.openai_model.OpenAI')
    def test_original_config_not_mutated(self, mock_openai_cls):
        """The original dict passed as generation_config should not be mutated."""
        original = {'temperature': 0, 'max_tokens': 100}
        original_copy = dict(original)
        MiniMaxModel(api_keys='k', generation_config=original)
        self.assertEqual(original, original_copy)


if __name__ == '__main__':
    unittest.main()
