import json
import unittest
from unittest.mock import mock_open, patch

from easyjailbreak.seed import SeedTemplate


class TestSeedTemplate(unittest.TestCase):

    def test_SeedTemplate(self):
        # inital seedtemplate
        seedtemplate = SeedTemplate()

        with self.assertRaises(AssertionError):
            seedtemplate.new_seeds(seeds_num=10)  # 请求的种子数量超过模板池的大小

        # test new_seeds
        new_seeds = seedtemplate.new_seeds(seeds_num=1, prompt_usage='attack', method_list=['default'], template_file=None)
        self.assertEqual(len(new_seeds), 1)

        # test new_seeds
        #　这个测试存在bug 原因未知
        new_seeds = seedtemplate.new_seeds(seeds_num=3, prompt_usage='attack', method_list=['Gptfuzzer','DeepInception','ICA','ICA'], template_file=None)
        self.assertEqual(len(new_seeds), 3)

        # 不给出seeds_num和method_list
        new_seeds = seedtemplate.new_seeds(seeds_num=None, prompt_usage='attack', method_list=None, template_file=None)
        self.assertEqual(len(new_seeds), 1)

        # 故意设置seeds_num大于模板数量，测试抛出异常
        with self.assertRaises(AssertionError):
            new_seeds = seedtemplate.new_seeds(seeds_num=100, prompt_usage='attack', method_list=['ReNeLLM','DeepInception','ICA'], template_file=None)

        #seeds_num= -1 时，测试抛出异常
        with self.assertRaises(AssertionError):
            new_seeds = seedtemplate.new_seeds(seeds_num=-1, prompt_usage='attack', method_list=None, template_file=None)

        # 输入错误的method_list
        with self.assertRaises(AttributeError):
            new_seeds = seedtemplate.new_seeds(seeds_num=1, prompt_usage='attack', method_list=['AAB','BBA'], template_file=None)


if __name__ == "__main__":
    unittest.main()
