"""
Multilingual Class
============================================
This Class translates harmful queries from English into nine non-English
languages with varying levels of resources, and in intentional scenarios,
malicious users deliberately combine malicious instructions with multilingual 
prompts to attack LLMs.

Paper title: MULTILINGUAL JAILBREAK CHALLENGES IN LARGE LANGUAGE MODELS
arXiv Link: https://arxiv.org/pdf/2310.06474.pdf
Source repository: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
"""
from collections import defaultdict
import requests
import logging
logging.basicConfig(level=logging.INFO)
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import *
from collections import defaultdict

__all__ = ['Multilingual']
class Multilingual(AttackerBase):
    r"""
    Multilingual is a class for conducting jailbreak attacks on language models.
    It can translate harmful queries from English into nine non-English languages.
    """
    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets: JailbreakDataset):
        r"""
        Initialize the Multilingual attack instance.
        :param attack_model: The attack_model should be set to None.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.mutations = [
            # Chinese
            Translate(language='zh-CN'),
            # Italian
            Translate(language='it'),
            # Vietnamese
            Translate(language='vi'),
            # Arabic
            Translate(language='ar'),
            # Korean
            Translate(language='ko'),
            # Thai
            Translate(language='th'),
            # Bengali
            Translate(language='bn'),
            # Swahili
            Translate(language='sw'),
            # Javanese
            Translate(language='jv'),
            ]
        self.evaluator = EvaluatorGenerativeJudge(eval_model)

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        Execute the single attack process using provided prompts.
        """
        instance_dataset = JailbreakDataset([instance])
        mutated_instance_list = []
        updated_instance_list = []

        for mutation in self.mutations:
            transformed_dataset = mutation(instance_dataset)
            for item in transformed_dataset:
                mutated_instance_list.append(item)
        
        for instance in mutated_instance_list:
            if instance.jailbreak_prompt is not None:
                answer = self.target_model.generate(instance.jailbreak_prompt.format(translated_query = instance.translated_query))
            else:
                answer = self.target_model.generate(instance.query)
            en_answer = self.translate_to_en(answer)
            instance.target_responses.append(en_answer)
            updated_instance_list.append(instance)

        return JailbreakDataset(updated_instance_list)

    def attack(self,):
        r"""
        Execute the attack process using Multilingual Jailbreak in Large Language Models.
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

    def update(self, dataset):
        r"""
        update the state of the Jailbroken based on the evaluation results of Datasets.
        """
        lang_groups = defaultdict(list)
        untranslate_query_groups = defaultdict(list)
        for prompt_node in dataset:
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject
            lang_groups[prompt_node.lang].append(prompt_node)
            untranslate_query_groups[prompt_node.query].append(prompt_node)

        self.lang_success_rates = {
            lang: sum(node.eval_results[-1] for node in group) / len(group) * 100
            for lang, group in lang_groups.items()
        }

        self.query_success_rates = {
            query: sum(node.eval_results[-1] for node in group) / len(group) * 100
            for query, group in untranslate_query_groups.items()
        }

        self.total_success_rate = sum(node.eval_results[-1] for node in dataset) / len(dataset) * 100

    def log(self):
        r"""
        report the attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")

        for lang, rate in self.lang_success_rates.items():
            logging.info(f"Success rate of {lang}: {rate:.2f}%")

        for query, rate in self.query_success_rates.items():
            logging.info(f"Jailbreak success rate for query '{query}': {rate:.2f}%")

        logging.info(f"Total success rate: {self.total_success_rate:.2f}%")
        logging.info("========Report End===========")
    
    def translate_to_en(self, text, src_lang='auto'):
        r"""
        Translate target response to English.
        """
        googleapis_url = 'https://translate.googleapis.com/translate_a/single'
        url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url,src_lang,'en',text)
        data = requests.get(url).json()
        res = ''.join([s[0] for s in data[0]])
        return res
