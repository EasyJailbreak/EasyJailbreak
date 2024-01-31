import unittest
from unittest.mock import MagicMock

from easyjailbreak.seed import SeedLLM


class TestSeedLLM(unittest.TestCase):

    def setUp(self):
        # 设置一个模拟的模型
        self.mock_model = MagicMock()
        self.mock_model.generate = MagicMock(return_value='mocked_output')
        self.mock_model.decode = MagicMock(return_value='decoded_output')

    def test_init(self):
        # 测试初始化
        seeds_example = ["seed1", "seed2"]
        seed_llm = SeedLLM(model=self.mock_model, seeds=seeds_example)
        self.assertEqual(seed_llm.seeds, seeds_example)

    def test_new_seeds_with_valid_insert(self):
        # 测试 new_seeds 方法与有效的 insert_values
        seed_llm = SeedLLM(model=self.mock_model)
        seeds = seed_llm.new_seeds(insert_values={'query': 'test'}, seeds_num=2)
        self.assertEqual(len(seeds), 2)
        self.assertEqual(seeds, ['decoded_output', 'decoded_output'])

    def test_new_seeds_with_invalid_insert(self):
        # 测试 new_seeds 方法与无效的 insert_values
        seed_llm = SeedLLM(model=self.mock_model)
        with self.assertRaises(AttributeError):
            seed_llm.new_seeds(insert_values={'wrong_key': 'test'})

    def test_mock_model_methods_called(self):
        # 确保调用了模型的 generate 和 decode 方法
        seed_llm = SeedLLM(model=self.mock_model)
        seed_llm.new_seeds(seeds_num=1)
        self.mock_model.generate.assert_called_once()
        self.mock_model.decode.assert_called_once_with('mocked_output')

if __name__ == "__main__":
    unittest.main()
