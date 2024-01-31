import unittest

from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.selector.SelectBasedOnScores import PrunePolicy


class TestPrunePolicy(unittest.TestCase):

    def test_init(self):
        # 测试初始化
        dataset = JailbreakDataset([])
        tree_width = 2
        selector = PrunePolicy(dataset, tree_width)
        self.assertEqual(selector.tree_width, tree_width)
        self.assertEqual(selector.Datasets, dataset)

    def test_select_empty_dataset(self):
        # 测试空数据集的情况
        dataset = JailbreakDataset([])
        selector = PrunePolicy(dataset, tree_width=2)
        with self.assertRaises(NotImplementedError):
            selector.select(None)

    def test_select_non_empty_dataset(self):
        # 测试非空数据集的选择逻辑
        instance_list = [Instance(query='query1', jailbreak_prompt='prompt1', eval_results=[2]),
                         Instance(query='query2', jailbreak_prompt='prompt2', eval_results=[8]),
                         Instance(query='query3', jailbreak_prompt='prompt3', eval_results=[1])]
        dataset = JailbreakDataset(instance_list)
        selector = PrunePolicy(dataset, tree_width=2)
        selected_instances = selector.select(dataset)
        self.assertEqual(len(selected_instances), 2)
        self.assertIn(selected_instances[0], instance_list)
        self.assertIn(selected_instances[1], instance_list)
        self.assertGreater(selected_instances[0].eval_results[-1], selected_instances[1].eval_results[-1])

if __name__ == "__main__":
    unittest.main()
