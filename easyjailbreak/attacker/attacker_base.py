"""
Attack Recipe Class
========================

This module defines a base class for implementing NLP jailbreak attack recipes.
These recipes are strategies or methods derived from literature to execute
jailbreak attacks on language models, typically to test or improve their robustness.

"""
from easyjailbreak.models import ModelBase
from easyjailbreak.loggers.logger import Logger
from easyjailbreak.datasets import JailbreakDataset, Instance

from abc import ABC, abstractmethod
from typing import Optional
import logging

__all__ = ['AttackerBase']

class AttackerBase(ABC):
    def __init__(
        self,
        attack_model: Optional[ModelBase],
        target_model: ModelBase,
        eval_model: Optional[ModelBase],
        jailbreak_datasets: JailbreakDataset,
        **kwargs
    ):
        """
        Initialize the AttackerBase.

        Args:
            attack_model (Optional[ModelBase]): Model used for the attack. Can be None.
            target_model (ModelBase): Model to be attacked.
            eval_model (Optional[ModelBase]): Evaluation model. Can be None.
            jailbreak_datasets (JailbreakDataset): Dataset for the attack.
        """
        assert attack_model is None or isinstance(attack_model, ModelBase)
        self.attack_model = attack_model

        assert isinstance(target_model, ModelBase)
        self.target_model = target_model
        self.eval_model = eval_model
        
        assert isinstance(jailbreak_datasets, JailbreakDataset)
        self.jailbreak_datasets = jailbreak_datasets

        self.logger = Logger()

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        """
        Perform a single-instance attack, a common use case of the attack method. Returns a JailbreakDataset containing the attack results.

        Args:
            instance (Instance): The instance to be attacked.

        Returns:
            JailbreakDataset: The attacked dataset containing the modified instances.
        """
        return NotImplementedError

    @abstractmethod
    def attack(self):
        """
        Abstract method for performing the attack.
        """
        return NotImplementedError

    def log_results(self, cnt_attack_success):
        """
        Report attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {len(self.jailbreak_datasets)}")
        logging.info(f"Total jailbreak: {cnt_attack_success}")
        logging.info(f"Total reject: {len(self.jailbreak_datasets)-cnt_attack_success}")
        logging.info("========Report End===========")