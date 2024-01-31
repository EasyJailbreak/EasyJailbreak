"""
RoundRobinSelectPolicy class
"""
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.selector.selector import SelectPolicy

__all__ = ["RoundRobinSelectPolicy"]

class RoundRobinSelectPolicy(SelectPolicy):
    """
    A selection policy that selects instances from a JailbreakDataset in a round-robin manner.
    This policy iterates over the dataset, selecting each instance in turn, and then repeats the process.
    """

    def __init__(self, Dataset: JailbreakDataset):
        """
        Initializes the RoundRobinSelectPolicy with a given JailbreakDataset.

        :param ~JailbreakDataset Dataset: The dataset from which instances will be selected in a round-robin fashion.
        """
        super().__init__(Dataset)
        self.index: int = 0

    def select(self) -> JailbreakDataset:
        """
        Selects the next instance in the dataset based on a round-robin approach and increments its visited count.

        :return ~JailbreakDataset: The selected instance from the dataset.
        """
        seed = self.Datasets[self.index]
        seed.visited_num += 1
        self.index = (self.index + 1) % len(self.Datasets)
        return JailbreakDataset([seed])

    def update(self, prompt_nodes: JailbreakDataset = None):
        """
        Updates the selection index based on the length of the dataset.

        :param ~JailbreakDataset prompt_nodes: Not used in this implementation.
        """
        self.index = (self.index - 1 + len(self.Datasets)) % len(self.Datasets)
