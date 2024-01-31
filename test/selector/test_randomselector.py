import unittest
from unittest.mock import MagicMock

from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.selector.RandomSelector import RandomSelectPolicy


class TestRandomSelectPolicy(unittest.TestCase):

    def setUp(self):
        # 创建一个模拟的 JailbreakDataset
        self.mock_dataset = JailbreakDataset([Instance(query="test query 1"), Instance(query="test query 2")])

    def test_init(self):
        # 测试初始化
        selector = RandomSelectPolicy(Datasets=self.mock_dataset)
        self.assertEqual(selector.Datasets, self.mock_dataset)

    def test_select(self):
        # 测试选择方法
        selector = RandomSelectPolicy(Datasets=self.mock_dataset)
        selected_instance = selector.select()
        self.assertIn(selected_instance, self.mock_dataset._dataset)
        self.assertGreater(selected_instance.visited_num, 0)

if __name__ == "__main__":
    unittest.main()
