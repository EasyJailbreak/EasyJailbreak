"""
SelectPolicy class
=======================
This file contains the implementation of policies for selecting instances from datasets,
specifically tailored for use in easy jailbreak scenarios. It defines abstract base classes
and concrete implementations for selecting instances based on various criteria.
"""
from abc import ABC, abstractmethod
from easyjailbreak.datasets import Instance, JailbreakDataset

__all__ = ["SelectPolicy"]

class SelectPolicy(ABC):
    """
    Abstract base class representing a policy for selecting instances from a JailbreakDataset.
    It provides a framework for implementing various selection strategies.
    """

    def __init__(self, Datasets: JailbreakDataset):
        """
        Initializes the SelectPolicy with a given JailbreakDataset.

        :param ~JailbreakDataset Datasets: The dataset from which instances will be selected.
        """
        self.Datasets = Datasets
        for k, instance in enumerate(self.Datasets):
            instance.visited_num = 0
            instance.index = k

    @abstractmethod
    def select(self) -> JailbreakDataset:
        """
        Abstract method that must be implemented by subclasses to define the selection strategy.

        :return ~Instance: The selected instance from the dataset.
        """
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, jailbreak_dataset: JailbreakDataset):
        """
        Updates the internal state of the selection policy, if necessary.

        :param ~JailbreakDataset jailbreak_dataset: The dataset to update the policy with.
        """
        pass

    def initial(self):
        """
        Initializes or resets any internal state of the selection policy, if necessary.
        """
        pass
