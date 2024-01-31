"""
UCBSelectPolicy class
==========================
"""
import numpy as np
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.selector.selector import SelectPolicy

__all__ = ["UCBSelectPolicy"]

class UCBSelectPolicy(SelectPolicy):
    """
    A selection policy based on the Upper Confidence Bound (UCB) algorithm. This policy is designed
    to balance exploration and exploitation when selecting instances from a JailbreakDataset.
    It uses the UCB formula to select instances that either have high rewards or have not been explored much.
    """
    def __init__(self,
                 explore_coeff: float = 1.0,
                 Dataset: JailbreakDataset = None):
        """
        Initializes the UCBSelectPolicy with a given JailbreakDataset and exploration coefficient.

        :param float explore_coeff: Coefficient to control the exploration-exploitation balance.
        :param ~JailbreakDataset Dataset: The dataset from which instances will be selected.
        """
        super().__init__(Dataset)

        self.step = 0
        self.last_choice_index:int = 0
        self.explore_coeff = explore_coeff
        self.rewards = [0 for _ in range(len(Dataset))]

    def select(self) -> JailbreakDataset:
        """
        Selects an instance from the dataset based on the UCB algorithm.

        :return ~JailbreakDataset: The selected JailbreakDataset from the dataset.
        """
        if len(self.Datasets) > len(self.rewards):
            self.rewards.extend([0 for _ in range(len(self.Datasets) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(self.Datasets))
        for i, prompt_node in enumerate(self.Datasets):
            smooth_visited_num = prompt_node.visited_num + 1
            scores[i] = self.rewards[i] / smooth_visited_num + \
                        self.explore_coeff * np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = int(np.argmax(scores))
        self.Datasets[self.last_choice_index].visited_num += 1
        return JailbreakDataset([self.Datasets[self.last_choice_index]])

    def update(self, Dataset: JailbreakDataset):
        """
        Updates the rewards for the last selected instance based on the success of the prompts.

        :param ~JailbreakDataset Dataset: The dataset containing prompts used for updating rewards.
        """
        succ_num = sum([prompt_node.num_jailbreak for prompt_node in Dataset])
        self.rewards[self.last_choice_index] += succ_num / len(Dataset)
