"""
ICA Class
============================================
This Class executes the In-Context Attack algorithm described in the paper below.
This part of code is based on the paper.

Paper title: Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations
arXiv link: https://arxiv.org/pdf/2310.06387.pdf
"""
import logging
import tqdm
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.datasets.instance import Instance
from easyjailbreak.seed import SeedTemplate
from easyjailbreak.metrics.Evaluator import EvaluatorPatternJudge


class ICA(AttackerBase):
    r"""
    In-Context Attack(ICA) crafts malicious contexts to guide models in generating harmful outputs.
    """

    def __init__(
            self, 
            target_model, 
            jailbreak_datasets: JailbreakDataset,
            attack_model = None,
            eval_model = None,
            max_query: int = 100,
            max_jailbreak: int = 100,
            max_reject: int = 100,
            max_iteration: int = 100,
            prompt_num: int = 5,
            user_input: bool = False,
            pattern_dict = None,
            ):
        r"""
        Initialize the ICA attack instance.
        :param ~model_wrapper target_model: The target model to be attacked.
        :param ~JailbreakDataset jailbreak_datasets: The dataset containing harmful queries.
        :param ~int prompt_num: The number of in-context demonstration.
        :param ~bool user_input: whether to use in-context demonstration input by user.
        :param ~dict pattern_dict: the pattern dictionary used in EvaluatorPatternJudge.
        """

        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)

        self.attack_results = JailbreakDataset([])
        self.evaluator = EvaluatorPatternJudge(pattern_dict=pattern_dict)

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration

        # ICA特有参数
        self.prompt_num: int = prompt_num
        self.user_input: bool = user_input

        # 初始化jailbreak prompt
        if not user_input:
            init_prompt = SeedTemplate().new_seeds(seeds_num=1, prompt_usage='attack', method_list=['ICA'])
            prompt = init_prompt[0]
        else:
            harmful_prompts = []
            harmful_responses = []
            print("Please input " + str(prompt_num) + " pairs of harmful prompts and corresponding responses\n")
            for i in range(prompt_num):
                harmful_prompts.append(input("harmful prompt:"))  
                harmful_responses.append(input("harmful response:")) 
            prompt = ""
            for i in range(prompt_num):
                prompt += "User:" + harmful_prompts[i] + '\nAssistant:' + harmful_responses[i] + '\n'
            prompt += "User:{query}"

        for instance in self.jailbreak_datasets:
            instance.jailbreak_prompt = prompt


    def single_attack(self, sample: Instance): 
        r"""
        Conduct a single attack on sample with n-shot attack demonstrations.
        Split the original jailbreak_prompt by roles and merge them into the current conversation_template as in-context demonstration.
        """
        prompt = sample.jailbreak_prompt.format(query=sample.query)
        prompt_splits = prompt.split("\n")
        messages = []
        for i in range(0, 2*self.prompt_num, 2):
            messages.append(prompt_splits[i].replace("User:", ""))
            messages.append(prompt_splits[i+1].replace("Assistant:", ""))
        messages.append(prompt_splits[-1].replace("User:", ""))
        response = self.target_model.generate(messages=messages)
        sample.target_responses.append(response)
        return JailbreakDataset([sample])


    def update(self, Dataset):
        """
        Update the state of the attack.
        """
        self.current_iteration += 1
        for Instance in Dataset:
            self.current_jailbreak += Instance.num_jailbreak
            self.current_query += Instance.num_query
            self.current_reject += Instance.num_reject


    def attack(self):
        """
        Main loop for the attack process, iterate through jailbreak_datasets.
        """
        logging.info("Jailbreak started!")
        try:
            for Instance in tqdm.tqdm(self.jailbreak_datasets, desc="processing instance"):
                mutated_instance = self.single_attack(Instance)[0]
                self.attack_results.add(mutated_instance)
            self.evaluator(self.attack_results)
            self.update(self.attack_results)
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        self.log()
        logging.info("Jailbreak finished!")
        return self.attack_results


    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("Jailbreak report:")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")
