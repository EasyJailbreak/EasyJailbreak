"""
Cipher Class
============================================
This Class enables humans to chat with LLMs through cipher prompts topped with 
system role descriptions and few-shot enciphered demonstrations.

Paper title：GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher
arXiv Link: https://arxiv.org/pdf/2308.06463.pdf
Source repository: https://github.com/RobustNLP/CipherChat
"""
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import MorseExpert, CaesarExpert, AsciiExpert, SelfDefineCipher

__all__ = ['Cipher']

class Cipher(AttackerBase):
    r"""
    Cipher is a class for conducting jailbreak attacks on language models. It integrates attack
    strategies and policies to evaluate and exploit weaknesses in target language models.
    """
    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets: JailbreakDataset):
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        r"""
        Initialize the Cipher Attacker.
        :param attack_model: In this case, the attack_model should be set as None.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        """
        self.mutations = [
            MorseExpert(),
            CaesarExpert(),
            AsciiExpert(),
            SelfDefineCipher()
            ]
        self.evaluator = EvaluatorGenerativeJudge(eval_model)
        self.info_dict = {'query': []}
        self.info_dict.update({expert.__class__.__name__: [] for expert in self.mutations})
        self.df = None

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        Conduct four cipher attack_mehtods on a single source instance.
        """
        source_jailbreakdataset = JailbreakDataset([instance])
        source_instance_list = []
        updated_instance_list = []

        for mutation in self.mutations:
            transformed_JailbreakDatasets = mutation(source_jailbreakdataset)
            for item in transformed_JailbreakDatasets:
                source_instance_list.append(item)

        for instance in source_instance_list:
            answer = self.target_model.generate(instance.jailbreak_prompt.format(encoded_query = instance.encoded_query))
            instance.encoded_target_responses = answer
            updated_instance_list.append(instance)

        for i,instance in enumerate(updated_instance_list):
            mutation = self.mutations[i]
            instance.target_responses.append(mutation.decode(instance.encoded_target_responses))
            updated_instance_list[i] = instance

        return JailbreakDataset(updated_instance_list)

    def attack(self):
        r"""
        Execute the attack process using four cipher methods on the entire jailbreak_datasets.
        """
        logging.info("Jailbreak started!")
        assert len(self.jailbreak_datasets) > 0, "The jailbreak_datasets must be a non-empty JailbreakDataset object."
        self.attack_results = JailbreakDataset([])
        try:
            for instance in self.jailbreak_datasets:
                self.info_dict['query'].append(instance.query)
                results = self.single_attack(instance)
                for new_instance in results:
                    self.attack_results.add(new_instance)
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.evaluator(self.attack_results)
        self.update(self.info_dict)

    def update(self, dictionary: dict):
        r"""
        Update the state of the Cipher based on the evaluation results of attack_results.
        """
        keys_iterator = iter(list(dictionary.keys())[1:])
        for evaluated_instance in self.attack_results:
            try:
                key = next(keys_iterator)
                dictionary[key].append(evaluated_instance.eval_results[-1])
            except StopIteration:
                keys_iterator = iter(list(dictionary.keys())[1:])
                key = next(keys_iterator)
                dictionary[key].append(evaluated_instance.eval_results[-1])
        self.df = pd.DataFrame(dictionary)
        self.df['q_s_r'] = self.df.apply(lambda row: row[1:].sum() / len(row[1:]), axis=1)
        column_probabilities = self.df.iloc[:, 1:].apply(lambda col: col.sum() / len(col))
        column_probabilities = pd.Series(['m_s_r'] + list(column_probabilities), index=self.df.columns)
        self.df.loc[self.df.index.max() + 1] = column_probabilities

    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("====================Jailbreak report:======================")
        for column in self.df.columns[1:-1]:
            logging.info(f"The success rate of {column}:{self.df[column].iloc[-1]* 100:.2f}%")
        logging.info("================Success Rate for Each Item:===============")
        for idx in self.df.index[:-1]:
            query_string = self.df.loc[idx, self.df.columns[0]]
            logging.info(f"{idx+1}.The jailbreak success rate of this query is {self.df.loc[idx].iloc[-1]* 100:.2f}%, {query_string}")
        logging.info("==================Overall success rate:====================")
        logging.info(f"{self.df.iloc[-1, -1]* 100:.2f}%")
        logging.info("======================Report End============================")
          
