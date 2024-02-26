r"""
'Multi-step Jailbreaking Privacy Attacks' Recipe
============================================
This module implements a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: Multi-step Jailbreaking Privacy Attacks on ChatGPT
arXiv link: https://arxiv.org/abs/2304.05197
Source repository: https://github.com/HKUST-KnowComp/LLM-Multistep-Jailbreak
"""
import copy
import logging
from fastchat.conversation import get_conv_template

from easyjailbreak.attacker.attacker_base import AttackerBase
from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset
from easyjailbreak.datasets.instance import Instance
from easyjailbreak.loggers.logger import Logger
from easyjailbreak.models.wenxinyiyan_model import WenxinyiyanModel
########## 4大件 ###############
from easyjailbreak.seed.seed_template import SeedTemplate
from easyjailbreak.mutation.rule.MJPChoices import MJPChoices
from easyjailbreak.metrics.Evaluator.Evaluator_Match import EvalatorMatch
from easyjailbreak.utils.model_utils import privacy_information_search

r"""
EasyJailbreak MJP class
============================================
"""
__all__ = ['MJP']

class MJP(AttackerBase):
    r"""
    Multi-step Jailbreaking Privacy Attacks, using somehow outdated jailbreaking prompt in the present
    to get privacy information including email and phone number from target LLM model.

    >>> from easyjailbreak.attacker.MJP_Li_2023 import MJP
    >>> from easyjailbreak.models.huggingface_model import from_pretrained
    >>> from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset
    >>> from easyjailbreak.datasets.Instance import Instance
    >>> target_model = from_pretrained(model_path_1)
    >>> eval_model  = from_pretrained(model_path_2)
    >>> dataset = JailbreakDataset('MJP')
    >>> attacker = MJP(target_model, eval_model, dataset)
    >>> attacker.attack()
    >>> attacker.jailbreak_Dataset.save_to_jsonl("./MJP_results.jsonl")
    """
    def __init__(self, target_model, eval_model, jailbreak_datasets, prompt_type='JQ+COT+MC', batch_num=5, template_file=None):
        r"""
        Initialize MJP, inherit from AttackerBase

        :param  ~HuggingfaceModel|~OpenaiModel target_model: LLM being attacked to generate adversarial responses
        :param  ~HuggingfaceModel|~OpenaiModel eval_model: LLM for evaluating during Pruning:phase1(constraint) and Pruning:phase2(select)
        :param  ~JailbreakDataset jailbreak_datasets:    dataset containing instances which conveys the query and reference responses
        :param  str prompt_type:  the kind of jailbreak including 'JQ+COT+MC', 'JQ+COT', 'JQ', 'DQ'
        :param  int batch_num:  the number of attacking attempts when the prompt_type include 'MC', i.e. multichoice
        :param  str template_file:   file path of the seed_template.json
        """
        super().__init__(attack_model=None, target_model=target_model, eval_model=eval_model, jailbreak_datasets=jailbreak_datasets)
        ############ 4大件 #################
        self.seeder=SeedTemplate().new_seeds(seeds_num=1,method_list=['MJP'],template_file=template_file)
        self.mutator=MJPChoices(prompt_type, self.target_model)
        self.evaluator=EvalatorMatch(eval_model)
        self.prompt_type=prompt_type
        self.batch_num = batch_num

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.jailbreak_Dataset = JailbreakDataset([])
        if isinstance(target_model, WenxinyiyanModel):
            self.conv_template = get_conv_template('chatgpt')
        else:
            self.conv_template = target_model.conversation
        self.logger = Logger()

    def attack(self):
        r"""
        Build the necessary components for the jailbreak attack.
        This function is used to complete the automated attack of the model on the user's given dataset.
        """
        logging.info("Jailbreak started!")
        try:
            for i, Instance in enumerate(self.jailbreak_datasets):
                print(f"ROW{i}")
                Instance.jailbreak_prompt = self.seeder[0]
                Instance.attack_attrs.update({'conversation':copy.deepcopy(self.conv_template)})
                Instance= self.single_attack(Instance)[0]
                print(f'\tRESPONSES:{Instance.target_responses}',flush=True)
                self.jailbreak_Dataset.add(Instance)
            self.update(self.jailbreak_Dataset)
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        print(f"ASR:{100*self.current_jailbreak/self.current_query}%")
        self.log()
        logging.info("Jailbreak finished!")

    def single_attack(self,Instance:Instance)->JailbreakDataset:
        r"""
        Execute a single query jailbreak attack.
        This method takes a query (usually a piece of text or input data) and applies
        the jailbreak attack strategy to generate a perturbed version or to derive
        insights on the model's weaknesses.

        :param ~Instance Instance: The input query or data point to be attacked.
        :return ~JailbreakDataset: processed JailbreakDataset
        """
        new_dataset=JailbreakDataset([Instance])
        new_dataset=self.mutator(new_dataset)
        messages = [conv[1] for conv in new_dataset[0].attack_attrs['conversation'].messages]
        res_list = []
        for _ in range(self.batch_num):
            if self.prompt_type == 'JQ':
                res = self.target_model.generate(messages[0])
                res = self.target_model.generate([res,messages[2]], clear_old_history=False)
            else:
                res = self.target_model.generate(messages)
            res_list.append(res)
        new_dataset[0].target_responses = [privacy_information_search(new_dataset[0].query, res_list, new_dataset[0].attack_attrs['target'])]
        self.conv_template.messages=[]
        self.evaluator(new_dataset)
        return new_dataset
    def update(self, Dataset: JailbreakDataset):
        r"""
        Update the state of the ReNeLLM based on the evaluation results of Datasets.

        :param ~JailbreakDataset Dateset: the input JailbreakDataset
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
