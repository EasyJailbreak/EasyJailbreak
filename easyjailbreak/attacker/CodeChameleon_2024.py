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
import os.path
__all__ = ['CodeChameleon']


class CodeChameleon(AttackerBase):
    r"""
    Implementation of CodeChameleon Jailbreak Challenges in Large Language Models
    """

    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets: JailbreakDataset):
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.mutations = [
            BinaryTree(attr_name='query'),
            # Length(attr_name='query'),
            # Reverse(attr_name='query'),
            # OddEven(attr_name='query'),
        ]
        self.evaluator = EvaluatorGenerativeJudge(eval_model)
        self.current_jailbreak = 0
        self.current_query = 0
        self.current_reject = 0

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        Single attack process with mutation-specific logging
        """
        instance_ds = JailbreakDataset([instance])
        source_instance_list = []
        updated_instance_list = []

        # Track mutations and log transformation results
        for mutation in self.mutations:
            mutation_name = mutation.__class__.__name__
            transformed_jailbreak_datasets = mutation(instance_ds)

            # Log mutation application and instance count
            logging.info(
                f"Applied {mutation_name} mutation. Generated {len(transformed_jailbreak_datasets)} instances.")

            # Add mutation metadata to instances
            for item in transformed_jailbreak_datasets:
                item.metadata = getattr(item, 'metadata', {})  # Initialize metadata if not exists
                item.metadata['applied_mutation'] = mutation_name
                source_instance_list.append(item)

        # Process generated instances and log details
        for instance in source_instance_list:
            try:
                # Generate response from target model
                formatted_prompt = instance.jailbreak_prompt.format(
                    decryption_function=instance.decryption_function,
                    query=instance.query
                )
                answer = self.target_model.generate(formatted_prompt)
                instance.target_responses.append(answer)

                # Log mutation-specific results
                mutation_used = instance.metadata.get('applied_mutation', 'unknown')
                logging.info(
                    f"Mutation '{mutation_used}' processed\n"
                    f"Original query: {instance.query}\n"
                    f"Transformed query: {formatted_prompt}\n"
                    f"Model response: {answer}\n"
                    "-------------------------"
                )

                updated_instance_list.append(instance)

            except Exception as e:
                logging.error(f"Error processing instance: {str(e)}")

        return JailbreakDataset(updated_instance_list)

    def get_mutated_prompts(self) -> list:
        r"""
        获取所有变异后的提示词
        
        :return: 包含所有变异后提示词的列表，包括攻击结果信息
        """
        mutated_prompts = []
        for instance in self.attack_results:
            formatted_prompt = instance.jailbreak_prompt.format(
                decryption_function=instance.decryption_function,
                query=instance.query
            )
            mutated_prompts.append({
                'original_query': instance.query,
                'mutated_prompt': formatted_prompt,
                'mutation_type': instance.metadata.get('applied_mutation', 'unknown'),
                'num_jailbreak': instance.num_jailbreak,
                'num_reject': instance.num_reject,
            })
        return mutated_prompts

    def attack(self, save_path = 'CodeChameleon_attack_result.jsonl'):
        r"""
        Execute attack with enhanced progress logging
        """
        logging.info("Jailbreak started! Initializing mutations...")
        self.attack_results = JailbreakDataset([])
        try:
            for idx, instance in enumerate(self.jailbreak_datasets):
                logging.info(f"Processing instance {idx + 1}/{len(self.jailbreak_datasets)}")
                results = self.single_attack(instance)
                for new_instance in results:
                    self.attack_results.add(new_instance)

            # Final evaluation and reporting
            logging.info("Final evaluation starting...")
            self.evaluator(self.attack_results)
            self.update(self.attack_results)
            logging.info("Final evaluation completed.")

        except KeyboardInterrupt:
            logging.warning("Jailbreak interrupted by user! Partial results preserved.")

        self.log()
        self.jailbreak_datasets.save_to_jsonl(save_path)
        logging.info(
            'Jailbreak result saved at {}!'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)))
        
        # 返回变异后的提示词
        return self.attack_results, self.get_mutated_prompts()

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
