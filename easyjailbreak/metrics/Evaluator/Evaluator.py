"""
Evaluator class
=================
"""
from abc import ABC, abstractmethod
from easyjailbreak.datasets import JailbreakDataset

class Evaluator(ABC):
    """
    Abstract base class for implementing evaluation strategies for JailbreakDataset instances.
    This class serves as a template for creating custom evaluators, allowing for diverse and specialized evaluation strategies tailored to different datasets or models.
    """
    def __init__(self, eval_model = None):
        """
        Initialize the Evaluator instance.

        :param ~ModelBase eval_model: The evaluation model to be used for dataset evaluation. Can be any model compatible with the evaluation strategy.
        """
        self.eval_model = eval_model

    def __call__(self, Datasets: JailbreakDataset, **kwargs):
        """
        Execute the evaluation process on the provided JailbreakDataset instances.

        :param ~JailbreakDataset Datasets: The dataset(s) to be evaluated.
        :param **kwargs: Additional keyword arguments that can be passed to the _evaluate method.
        """
        for instance in Datasets:
            self._evaluate(instance, **kwargs)

    @abstractmethod
    def _evaluate(self, instance, **kwargs):
        """
        Abstract method to be implemented by subclasses for specific evaluation strategies.

        :param ~Instance instance: A single instance from the JailbreakDataset to be evaluated.
        :param **kwargs: Additional keyword arguments relevant to the specific evaluation strategy.
        :return: The result of the evaluation, the nature of which depends on the specific implementation in subclasses.
        """
        return NotImplementedError()
