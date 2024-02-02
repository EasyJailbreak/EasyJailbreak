"""
MCTSExploreSelectPolicy class
================================
"""
import numpy as np
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.selector import SelectPolicy


class MCTSExploreSelectPolicy(SelectPolicy):
    """
    This class implements a selection policy based on the Monte Carlo Tree Search (MCTS) algorithm.
    It is designed to explore and exploit a dataset of instances for effective jailbreaking of LLMs.
    """
    def __init__(self, dataset, inital_prompt_pool, Questions,ratio=0.5, alpha=0.1, beta=0.2):
        """
        Initialize the MCTS policy with dataset and parameters for exploration and exploitation.

        :param ~JailbreakDataset dataset: The dataset from which instances are to be selected.
        :param ~JailbreakDataset initial_prompt_pool: A collection of initial prompts to start the selection process.
        :param ~JailbreakDataset Questions: A set of questions or tasks to be addressed by the selected instances.
        :param float ratio: The balance between exploration and exploitation (default 0.5).
        :param float alpha: Penalty parameter for level adjustment (default 0.1).
        :param float beta: Reward scaling factor (default 0.2).
        """
        super().__init__(dataset)
        self.inital_prompt_pool = inital_prompt_pool
        self.Questions = Questions
        self.step = 0
        self.mctc_select_path = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio  # balance between exploration and exploitation
        self.alpha = alpha  # penalty for level
        self.beta = beta


    def select(self) -> JailbreakDataset:
        """
        Selects an instance from the dataset using MCTS algorithm.

        :return ~JailbreakDataset: The selected instance from the dataset.
        """
        self.step += 1
        if len(self.Datasets) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.Datasets) - len(self.rewards))])
        self.mctc_select_path = []

        cur = max(
            self.inital_prompt_pool._dataset,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.children) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.children,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return JailbreakDataset([cur])

    def update(self, prompt_nodes: JailbreakDataset):
        """
        Updates the weights of nodes in the MCTS tree based on their performance.

        :param ~JailbreakDataset prompt_nodes: Dataset of prompt nodes to update.
        """
        # update weight
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        last_choice_node = self.Datasets[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.Questions)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * max(self.beta, (1 - 0.1 * last_choice_node.level))
