import unittest
from unittest.mock import MagicMock, patch

from easyjailbreak.models import OpenaiModel


class TestOpenaiModel(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_api_key"
        self.model_name = "test_model"
        self.mock_client = MagicMock()
        self.openai_model = OpenaiModel(self.model_name, self.api_key)

    @patch('your_module.OpenAI')
    def test_init(self, mock_openai):
        # Test initialization
        mock_openai.assert_called_with(api_key=self.api_key)
        self.assertEqual(self.openai_model.model_name, self.model_name)

    def test_set_system_message(self):
        # Test setting system message
        system_message = "System message"
        self.openai_model.set_system_message(system_message)
        self.assertEqual(self.openai_model.conversation.system_message, system_message)

    @patch('your_module.OpenAI')
    def test_generate(self, mock_openai):
        # Test generating a response
        message = "Test message"
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Response"))])
        response = self.openai_model.generate(message)
        self.assertEqual(response, "Response")

    @patch('your_module.OpenAI')
    def test_batch_generate(self, mock_openai):
        # Test batch generation
        conversations = [["Message 1"], ["Message 2"]]
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Response"))])
        responses = self.openai_model.batch_generate(conversations)
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(response == "Response" for response in responses))

if __name__ == '__main__':
    unittest.main()
