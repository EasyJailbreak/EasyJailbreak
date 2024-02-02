r"""
'SelectBasedOnScores', select those instances whose scores are high(scores are on the extent of jailbreaking),
detail information can be found in the following paper.

Paper title: Tree of Attacks: Jailbreaking Black-Box LLMs Automatically
arXiv link: https://arxiv.org/abs/2312.02119
Source repository: https://github.com/RICommunity/TAP
"""
import numpy as np
from typing import List

from easyjailbreak.selector.selector import SelectPolicy
from easyjailbreak.datasets import Instance,JailbreakDataset

class SelectBasedOnScores(SelectPolicy):
    """
    This class implements a selection policy based on the scores of instances in a JailbreakDataset.
    It selects a subset of instances with high scores, relevant for jailbreaking tasks.
    """
    def __init__(self, Dataset: JailbreakDataset, tree_width):
        r"""
        Initialize the selector with a dataset and a tree width.

        :param ~JailbreakDataset Dataset: The dataset from which instances are to be selected.
        :param int tree_width: The maximum number of instances to select.
        """
        super().__init__(Dataset)
        self.tree_width = tree_width
    def select(self, dataset:JailbreakDataset) -> JailbreakDataset:
        r"""
        Selects a subset of instances from the dataset based on their scores.

        :param ~JailbreakDataset dataset: The dataset from which instances are to be selected.
        :return List[Instance]: A list of selected instances with high evaluation scores.
        """
        if dataset != None:
            list_dataset = [instance for instance in dataset]
            # Ensures that elements with the same score are randomly permuted
            np.random.shuffle(list_dataset)
            list_dataset.sort(key=lambda x:x.eval_results[-1],reverse=True)

            # truncate/select based on judge_scores/instance.eval_results[-1]
            width = min(self.tree_width, len(list_dataset))
            truncated_list = [list_dataset[i] for i in range(width) if list_dataset[i].eval_results[-1] > 0]
            # Ensure that the truncated list has at least two elements
            if len(truncated_list) == 0:
                truncated_list = [list_dataset[0],list_dataset[1]]

            return JailbreakDataset(truncated_list)
        else:
            list_dataset = [instance for instance in self.Datasets]
            # Ensures that elements with the same score are randomly permuted
            np.random.shuffle(list_dataset)
            list_dataset.sort(key=lambda x: x.eval_results[-1], reverse=True)

            # truncate/select based on judge_scores/instance.eval_results[-1]
            width = min(self.tree_width, len(list_dataset))
            truncated_list = [list_dataset[i] for i in range(width) if list_dataset[i].eval_results[-1] > 0]
            # Ensure that the truncated list has at least two elements
            if len(truncated_list) == 0:
                truncated_list = [list_dataset[0], list_dataset[1]]

            return JailbreakDataset(truncated_list)