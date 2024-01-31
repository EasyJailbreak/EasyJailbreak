import unittest
import numpy as np
from unittest.mock import MagicMock

from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.selector.EXP3SelectPolicy import EXP3SelectPolicy


class TestEXP3SelectPolicy(unittest.TestCase):

    def setUp(self):
        # 创建一个模拟的 JailbreakDataset
        self.mock_dataset = JailbreakDataset([Instance(query="test query 1"), Instance(query="test query 2")])

    def test_init_and_initial(self):
        # 测试初始化和 initial 方法
        selector = EXP3SelectPolicy(Dataset=self.mock_dataset)
        self.assertEqual(selector.energy, 1.0)
        self.assertEqual(selector.gamma, 0.05)
        self.assertEqual(selector.alpha, 25)
        self.assertIsNotNone(selector.weights)
        self.assertIsNotNone(selector.probs)

    def test_select(self):
        # 测试选择方法
        selector = EXP3SelectPolicy(Dataset=self.mock_dataset)
        selected_instance = selector.select()
        self.assertIn(selected_instance, self.mock_dataset._dataset)
        self.assertGreater(selected_instance.visited_num, 0)

    def test_update(self):
        # 测试更新方法
        selector = EXP3SelectPolicy(Dataset=self.mock_dataset)
        selector.select()  # 先选择一个实例，以便有 last_choice_index
        prompt_nodes = JailbreakDataset([Instance(num_jailbreak=0), Instance(num_jailbreak=1)])
        selector.update(prompt_nodes)
        # 测试是否正确更新权重
        self.assertNotEqual(selector.weights[selector.last_choice_index], 1)

if __name__ == "__main__":
    unittest.main()
