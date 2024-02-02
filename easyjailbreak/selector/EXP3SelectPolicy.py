"""
EXP3SelectPolicy class
==========================
"""
import numpy as np
from easyjailbreak.selector import SelectPolicy
from easyjailbreak.datasets import Instance, JailbreakDataset

__all__ = ["EXP3SelectPolicy"]

class EXP3SelectPolicy(SelectPolicy):
    """
    A selection policy based on the Exponential-weight algorithm for Exploration and Exploitation (EXP3).
    This policy is designed for environments with adversarial contexts, balancing between exploring new instances
    and exploiting known rewards in a JailbreakDataset.
    """

    def __init__(self,
                 Dataset: JailbreakDataset,
                 energy: float = 1.0,
                 gamma: float = 0.05,
                 alpha: float = 25):
        """
        Initializes the EXP3SelectPolicy with a given JailbreakDataset and parameters for the EXP3 algorithm.

        :param ~JailbreakDataset Dataset: The dataset from which instances will be selected.
        :param float energy: Initial energy level (not used in current implementation).
        :param float gamma: Parameter for controlling the exploration-exploitation trade-off.
        :param float alpha: Learning rate for the weight updates.
        """
        super().__init__(Dataset)

        self.energy = energy
        self.gamma = gamma
        self.alpha = alpha
        self.last_choice_index = None

        self.initial()

    def initial(self):
        """
        Initializes or resets the weights and probabilities for each instance in the dataset.
        """
        self.weights = [1. for _ in range(len(self.Datasets))]
        self.probs = [0. for _ in range(len(self.Datasets))]

    def select(self) -> JailbreakDataset:
        """
        Selects an instance from the dataset based on the EXP3 algorithm.

        :return ~JailbreakDataset: The selected instance from the dataset.
        """
        if len(self.Datasets) > len(self.weights):
            self.weights.extend([1. for _ in range(len(self.Datasets) - len(self.weights))])
        if len(self.Datasets) > len(self.probs):
            self.probs.extend([0. for _ in range(len(self.Datasets) - len(self.probs))])

        np_weights = np.array(self.weights)
        probs = (1 - self.gamma) * np_weights / np_weights.sum() + self.gamma / len(self.Datasets)

        self.last_choice_index = np.random.choice(len(self.Datasets), p=probs)

        self.Datasets[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]

        return JailbreakDataset([self.Datasets[self.last_choice_index]])

    def update(self, prompt_nodes: JailbreakDataset):
        """
        Updates the weights of the last chosen instance based on the success of the prompts.

        :param ~JailbreakDataset prompt_nodes: The dataset containing prompts used for updating weights.
        """
        succ_num = sum([prompt_node.num_jailbreak for prompt_node in prompt_nodes])

        r = 1 - succ_num / len(prompt_nodes)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(self.alpha * x / len(self.Datasets))
