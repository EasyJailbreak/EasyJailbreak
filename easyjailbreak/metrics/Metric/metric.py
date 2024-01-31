"""
Metric Class
========================
This module defines the Metric class, an abstract base class used for creating various
metrics that evaluate the results and data quality in the context of adversarial examples
and attacks. It provides a standardized interface for defining and implementing custom
metrics in the TextAttack framework.
"""

from abc import ABC, abstractmethod
from easyjailbreak.datasets import JailbreakDataset
__all__ = ["Metric"]

class Metric(ABC):
    r"""
    Abstract base class for defining metrics to evaluate adversarial example results
    and data quality in the context of TextAttack.

    This class serves as a blueprint for implementing various types of metrics, ensuring
    consistent interfaces and functionalities across different metric implementations.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        r"""
        Initializes the Metric instance.

        This abstract method should be implemented in subclasses to set up any necessary
        configurations for the specific metric.

        :param **kwargs: Arbitrary keyword arguments specific to each metric implementation.
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate(self, dataset: JailbreakDataset):
        r"""
        Abstract method for computing metric values based on the provided results.

        This method should be implemented in subclasses to calculate the metric based on
        the attack results.

        :param ~JailbreakDataset dataset: A list of instances with the results
                        of attacks on different instances in the dataset.
        """
        raise NotImplementedError
