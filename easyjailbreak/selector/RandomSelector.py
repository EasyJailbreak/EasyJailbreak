"""
RandomSelectPolicy class
============================
"""
import random
from easyjailbreak.selector import SelectPolicy
from easyjailbreak.datasets import JailbreakDataset

__all__ = ["RandomSelectPolicy"]

class RandomSelectPolicy(SelectPolicy):
    """
    A selection policy that randomly selects an instance from a JailbreakDataset.
    It extends the SelectPolicy abstract base class, providing a concrete implementation
    for the random selection strategy.
    """

    def __init__(self, Datasets: JailbreakDataset):
        """
        Initializes the RandomSelectPolicy with a given JailbreakDataset.

        :param ~JailbreakDataset Datasets: The dataset from which instances will be randomly selected.
        """
        super().__init__(Datasets)

    def select(self) -> JailbreakDataset:
        """
        Selects an instance randomly from the dataset and increments its visited count.

        :return ~JailbreakDataset: The randomly selected instance from the dataset.
        """
        seed = random.choice(self.Datasets._dataset)
        seed.visited_num += 1
        return JailbreakDataset([seed])
