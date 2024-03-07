"""
CodeChameleon Class
============================================
A novel framework for jailbreaking in LLMs based on
personalized encryption and decryption.

Paper title: CodeChameleon: Personalized Encryption Framework for Jailbreaking Large Language Models
arXiv Link: https://arxiv.org/abs/2402.16717
"""

import logging
logging.basicConfig(level=logging.INFO)
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import *

__all__ = ['CodeChameleon']
     
class CodeChameleon(AttackerBase):
    r"""
    Implementation of CodeChameleon Jailbreak Challenges in Large Language Models
    """
    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets: JailbreakDataset):
        r"""
        :param attack_model: The attack_model is used to generate the adversarial prompt. In this case, the attack_model should be set as None.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        :param template_file: The file path of the template.
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.mutations = [
            BinaryTree(attr_name='query'),
            Length(attr_name='query'),
            Reverse(attr_name='query'),
            OddEven(attr_name='query'),
        ]
        self.evaluator = EvaluatorGenerativeJudge(eval_model)
        self.current_jailbreak = 0
        self.current_query = 0
        self.current_reject = 0

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        single attack process using provided prompts and mutation methods.

        :param instance: The Instance that is attacked.
        """
        instance_ds = JailbreakDataset([instance])
        source_instance_list = []
        updated_instance_list = []

        for mutation in self.mutations:
            transformed_jailbreak_datasets = mutation(instance_ds)
            for item in transformed_jailbreak_datasets:
                source_instance_list.append(item)
                    
        for instance in source_instance_list:
            answer = self.target_model.generate(instance.jailbreak_prompt.format(decryption_function = instance.decryption_function, query = instance.query))
            instance.target_responses.append(answer)
            updated_instance_list.append(instance)
        return JailbreakDataset(updated_instance_list)
    
    def attack(self):
        r"""
        Execute the attack process using provided prompts and mutations.
        """
        logging.info("Jailbreak started!")
        self.attack_results = JailbreakDataset([])
        try:
            for Instance in self.jailbreak_datasets:
                results = self.single_attack(Instance)
                for new_instance in results:
                    self.attack_results.add(new_instance)
        
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        
        self.evaluator(self.attack_results)
        self.update(self.attack_results)
        logging.info("Jailbreak finished!")

    def update(self, Dataset: JailbreakDataset):
        r"""
        Update the state of the Jailbroken based on the evaluation results of Datasets.
        
        :param Dataset: The Dataset that is attacked.
        """
        for prompt_node in Dataset:
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")
        logging.info("========Report End===========")
