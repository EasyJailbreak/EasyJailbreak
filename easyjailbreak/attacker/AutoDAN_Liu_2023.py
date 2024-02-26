'''
AutoDAN Class
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: AUTODAN: GENERATING STEALTHY JAILBREAK PROMPTS ON ALIGNED LARGE LANGUAGE MODELS
arXiv link: https://arxiv.org/abs/2310.04451
Source repository: https://github.com/SheltonLiu-N/AutoDAN.git
'''

import logging
import gc
import numpy as np
import torch
import torch.nn as nn
import time
import random
from fastchat import model
import nltk
from nltk.corpus import stopwords, wordnet
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from itertools import islice
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.datasets.instance import Instance
from easyjailbreak.mutation.generation import Rephrase
from easyjailbreak.mutation.rule import CrossOver, ReplaceWordsWithSynonyms
from easyjailbreak.metrics.Evaluator import EvaluatorPatternJudge
from easyjailbreak.seed import SeedTemplate

__all__ = ["AutoDAN", "autodan_PrefixManager"]

def load_conversation_template(template_name):
    r"""
    load conversation template
    """
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    return conv_template


def get_developer(model_name):
    r"""
    get model developer
    """
    developer_dict = {"llama2": "Meta"}
    return developer_dict[model_name]


def generate(model: AutoModelForCausalLM, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]
    return output_ids[assistant_role_slice.stop:]


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None
        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)
        gc.collect()
    del batch_input_ids, batch_attention_mask
    return torch.cat(logits, dim=0)


class AutoDAN(AttackerBase):
    r"""
    AutoDAN is a class for conducting jailbreak attacks on language models.
    AutoDAN can automatically generate stealthy jailbreak prompts by hierarchical genetic algorithm.
    """

    def __init__(
            self,
            attack_model,
            target_model,
            jailbreak_datasets: JailbreakDataset,
            eval_model = None,
            max_query: int = 100,
            max_jailbreak: int = 100,
            max_reject: int = 100,
            max_iteration: int = 100,
            device = 'cuda:0',
            num_steps: int = 100,
            sentence_level_steps: int = 5,
            word_dict_size: int = 30,
            batch_size: int = 64,
            num_elites: float = 0.1,
            crossover_rate: float = 0.5,
            mutation_rate: float = 0.01,
            num_points: int =5,
            model_name: str = "llama2",
            low_memory: int = 0,
            pattern_dict: dict = None,
            ):
        r"""
        Initialize the AutoDAN attack instance.
        :param ~model_wrapper attack_model: The model used to generate attack prompts.
        :param ~model_wrapper target_model: The target model to be attacked.
        :param ~JailbreakDataset jailbreak_datasets: The dataset containing harmful queries.
        :param ~model_wrapper eval_model: The model used for evaluating attck effectiveness during attacks.
        :param ~int num_steps: the number of paragraph-level iteration of AutoDAN-HGA algorithm.
        :param ~int sentence_level_steps: the number of sentence-level iteration of AutoDAN-HGA algorithm.
        :param ~int word_dict_size: the word_dict size of AutoDAN-HGA algorithm.
        :param ~int batch_size: the number of candidate prompts of each query.
        :param ~float num_elites: the proportion of elites used in Genetic Algorithm.
        :param ~float crossover_rate: the probability to execute crossover mutation.
        :param ~float mutation_rate: the probability to execute rephrase mutation.
        :param ~int num_points: the number of break points used in crossover mutation.
        :param ~str model_name: the target model name.
        :param ~int low_memory: 1 if low memory else 0
        :param ~dict pattern_dict: the pattern dictionary used in EvaluatorPatternJudge.
        """

        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)

        self.attack_results = JailbreakDataset([])

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration

        # AutoDAN特有参数
        self.device = device
        self.num_steps: int = num_steps
        self.sentence_level_steps: int = sentence_level_steps
        self.word_dict_size: int = word_dict_size 
        self.batch_size: int = batch_size  
        self.num_elites: float = num_elites  
        self.crossover_rate: float = crossover_rate  
        self.num_points: int = num_points  
        self.mutation_rate: float = mutation_rate  
        self.model_name: str = model_name  
        self.low_memory: int = low_memory  

        self.evaluator = EvaluatorPatternJudge(pattern_dict=pattern_dict)

        self.crossover_mutation = CrossOver(attr_name='jailbreak_prompt', num_points=self.num_points, seed_pool=None)
        self.replace_words_with_synonyms_mutation = ReplaceWordsWithSynonyms(attr_name="jailbreak_prompt", word_dict={})
        self.rephrase_mutation = Rephrase(model=self.attack_model)

        self.conv_template = load_conversation_template(model_name)

        # original candidate prompts
        self.reference = SeedTemplate().new_seeds(seeds_num=self.batch_size, prompt_usage='attack', method_list=["AutoDAN-a"])
        for o in range(len(self.reference)):
            self.reference[o] = self.reference[o].replace('ChatGPT', self.model_name)
            self.reference[o] = self.reference[o].replace('chatGPT', self.model_name)
            self.reference[o] = self.reference[o].replace('ModelKeeper', get_developer(self.model_name))
            self.reference[o] = self.reference[o] + ' [PROMPT]:'


    def roulette_wheel_selection(self, data_list, score_list, num_selected):
        r"""
        apply roulette_wheel_selection on data_list
        """
        selection_probs = np.exp(score_list - np.max(score_list))
        selection_probs = selection_probs / selection_probs.sum()
        selected_indices = np.random.choice(len(data_list), size=num_selected, p=selection_probs, replace=True)
        selected_data = [data_list[i] for i in selected_indices]
        return selected_data


    def replace_with_synonyms(self, sentence, num=10):
        r"""
        replace words in sentence with synonyms
        """
        T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
            "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
        stop_words = set(stopwords.words('english'))
        words = nltk.word_tokenize(sentence)
        uncommon_words = [word for word in words if word.lower() not in stop_words and word.lower() not in T]
        selected_words = random.sample(uncommon_words, min(num, len(uncommon_words)))
        for word in selected_words:
            synonyms = wordnet.synsets(word)
            if synonyms and synonyms[0].lemmas():
                synonym = synonyms[0].lemmas()[0].name()
                sentence = sentence.replace(word, synonym, 1)
        return sentence

    
    def construct_momentum_word_dictionary(self, word_dict, individuals, score_list):
        r"""
        calculate momentum with score_list to maintain a momentum word_dict
        """ 
        word_scores = {}
        for individual, score in zip(individuals, score_list):
            T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
                "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
            stop_words = set(stopwords.words('english'))
            words = nltk.word_tokenize(individual)
            uncommon_words = [word for word in words if word.lower() not in stop_words and word.lower() not in T]
            for word in uncommon_words:
                if word in word_scores.keys():
                    word_scores[word].append(score)
                else:
                    word_scores[word] = []
                    word_scores[word].append(score)
            for word, scores in word_scores.items():
                avg_score = sum(scores) * 1.0 / len(scores)
                if word in word_dict.keys():
                    word_dict[word] = (word_dict[word] + avg_score)/2
                else:
                    word_dict[word] = avg_score
            sorted_word_dict = dict(sorted(word_dict.items(), key = lambda x:x[1], reverse=True))
            return dict(islice(sorted_word_dict.items(), self.word_dict_size))


    def get_score_autodan(self, conv_template, instruction, target, model, device, test_controls=None, crit=None):
        r"""
        Convert all test_controls to token ids and find the max length
        """
        input_ids_list = []
        target_slices = []
        for item in test_controls:
            prefix_manager = autodan_PrefixManager(tokenizer=self.target_model.tokenizer,
                                                   conv_template=conv_template,
                                                   instruction=instruction,
                                                   target=target,
                                                   adv_string=item)
            input_ids = prefix_manager.get_input_ids(adv_string=item).to(device)
            input_ids_list.append(input_ids)
            target_slices.append(prefix_manager._target_slice)

        # Pad all token ids to the max length
        pad_tok = 0
        for ids in input_ids_list:
            while pad_tok in ids:
                pad_tok += 1

        # Find the maximum length of input_ids in the list
        max_input_length = max([ids.size(0) for ids in input_ids_list])

        # Pad each input_ids tensor to the maximum length
        padded_input_ids_list = []
        for ids in input_ids_list:
            pad_length = max_input_length - ids.size(0)
            padded_ids = torch.cat([ids, torch.full((pad_length,), pad_tok, device=device)], dim=0)
            padded_input_ids_list.append(padded_ids)

        # Stack the padded input_ids tensors
        input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)

        attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype).to(device)

        # Forward pass and compute loss
        logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=attn_mask, batch_size=len(test_controls))
        losses = []
        for idx, target_slice in enumerate(target_slices):
            loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
            logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
            targets = input_ids_tensor[idx, target_slice].unsqueeze(0)
            loss = crit(logits_slice, targets)
            losses.append(loss)

        del input_ids_list, target_slices, input_ids_tensor, attn_mask
        gc.collect()
        return torch.stack(losses)


    def get_score_autodan_low_memory(self, conv_template, instruction, target, model, device, test_controls=None,
                                    crit=None):
        r"""
        Convert all test_controls to token ids and find the max length when memory is low
        """
        losses = []
        for item in test_controls:
            prefix_manager = autodan_PrefixManager(tokenizer=self.target_model.tokenizer,
                                                conv_template=conv_template,
                                                instruction=instruction,
                                                target=target,
                                                adv_string=item)
            input_ids = prefix_manager.get_input_ids(adv_string=item).to(device)
            input_ids_tensor = torch.stack([input_ids], dim=0)

            # Forward pass and compute loss
            logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=None, batch_size=len(test_controls))

            target_slice = prefix_manager._target_slice
            loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
            logits_slice = logits[0, loss_slice, :].unsqueeze(0).transpose(1, 2)
            targets = input_ids_tensor[0, target_slice].unsqueeze(0)
            loss = crit(logits_slice, targets)
            losses.append(loss)

        del input_ids_tensor
        gc.collect()
        return torch.stack(losses)


    def evaluate_candidate_prompts(self, sample: Instance, prefix_manager):
        r"""
        Calculate current candidate prompts scores of sample, get the currently best prompt and the corresponding response.
        """
        if self.low_memory == 1:
            losses = self.get_score_autodan_low_memory(
                conv_template=self.conv_template, instruction=sample.query, target=sample.reference_responses[0],
                model=self.target_model,
                device=self.device,
                test_controls=sample.candidate_prompts,
                crit=nn.CrossEntropyLoss(reduction='mean')
                )
        else:
            losses = self.get_score_autodan(
                conv_template=self.conv_template, instruction=sample.query, target=sample.reference_responses[0],
                model=self.target_model,
                device=self.device,
                test_controls=sample.candidate_prompts,
                crit=nn.CrossEntropyLoss(reduction='mean')
                )
        score_list = losses.cpu().numpy().tolist()

        best_new_adv_prefix_id = losses.argmin()
        best_new_adv_prefix = sample.candidate_prompts[best_new_adv_prefix_id]

        current_loss = losses[best_new_adv_prefix_id]

        adv_prefix = best_new_adv_prefix

        output_ids = generate(
            model=self.target_model.model,
            tokenizer=self.target_model.tokenizer,
            input_ids=prefix_manager.get_input_ids(adv_string=adv_prefix).to(self.device),
            assistant_role_slice=prefix_manager._assistant_role_slice,
            gen_config=None
        )
        response = self.target_model.tokenizer.decode(output_ids).strip()

        return score_list, current_loss, adv_prefix, response


    def update(self, Dataset: JailbreakDataset):
        r"""
        update jailbreak state
        """
        self.current_iteration += 1
        for instance in Dataset:
            self.current_jailbreak += instance.num_jailbreak
            self.current_query += instance.num_query
            self.current_reject += instance.num_reject


    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("Jailbreak report:")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")
        logging.info(f"Total iteration: {self.current_iteration}")


    def attack(self):
        r"""
        Main loop for the attack process, iterate through jailbreak_datasets.
        """
        logging.info("Jailbreak started!")
        try:
            for instance in tqdm(self.jailbreak_datasets, desc="processing instance"):
                new_instance = self.single_attack(instance)[0]
                self.attack_results.add(new_instance)
            self.update(self.attack_results)
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        self.log()
        logging.info("Jailbreak finished!")
        return self.attack_results


    def single_attack(self, instance: Instance):
        r"""
        Perform the AutoDAN-HGA algorithm on a single query.
        """
        best_prompt = ""
        user_prompt = instance.query
        target = instance.reference_responses[0]
        prefix_manager = autodan_PrefixManager(tokenizer=self.target_model.tokenizer,
                                    conv_template=self.conv_template,
                                    instruction=user_prompt,
                                    target=target,
                                    adv_string=self.reference[0])

        new_adv_prefixes = self.reference
        instance.candidate_prompts = new_adv_prefixes

        # 1. Initialize population with LLM-based Diversification
        for i in range(len(instance.candidate_prompts)):
            if random.random() < self.mutation_rate:
                instance.candidate_prompts[i] = self.rephrase_mutation.rephrase(instance.candidate_prompts[i])

        word_dict = {}
        # GENETIC ALGORITHM
        # Paragraph-level Iterations
        for j in range(self.num_steps):
            with torch.no_grad():
                epoch_start_time = time.time()

                # 2. Evaluate the fitness score of each individual in population
                score_list, current_loss, adv_prefix, response = self.evaluate_candidate_prompts(instance, prefix_manager)

                # 3. Evaluate jailbreak success or not
                instance.target_responses.append(response)
                self.evaluator(JailbreakDataset([instance]))
                is_success = instance.eval_results[-1]

                if is_success == 1:
                    epoch_end_time = time.time()
                    epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)
                    print(
                        "################################\n"
                        f"Current Epoch: {j}/{self.num_steps}\n"
                        f"Passed:{is_success}\n"
                        f"Loss:{current_loss.item()}\n"
                        f"Epoch Cost:{epoch_cost_time}\n"
                        f"Current prefix:\n{adv_prefix}\n"
                        f"Current Response:\n{response}\n"
                        "################################\n")
                    gc.collect()
                    torch.cuda.empty_cache()
                    best_prompt = adv_prefix
                    break

                # 4. Sort the score_list and get corresponding control_prefixes
                score_list = [-x for x in score_list]
                sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
                sorted_control_prefixes = [new_adv_prefixes[i] for i in sorted_indices]
                sorted_socre_list = [score_list[i] for i in sorted_indices]

                # 5. Select the elites
                num_elites = int(self.batch_size * self.num_elites)
                elites = sorted_control_prefixes[:num_elites]

                # 6. Use roulette wheel selection for the remaining positions
                parents_list = self.roulette_wheel_selection(sorted_control_prefixes[num_elites:], sorted_socre_list[num_elites:], self.batch_size - num_elites)
                instance.candidate_prompts = parents_list

                # 7. Apply crossover and mutation to the selected parents
                mutated_prompts = []
                mutation_dataset = JailbreakDataset([])
                for p in parents_list:
                    mutation_dataset.add(Instance(jailbreak_prompt=p))
                for i in range(0, len(parents_list), 2):
                    parent1 = mutation_dataset[i]
                    parent2 = mutation_dataset[i + 1] if (i + 1) < len(parents_list) else mutation_dataset[0]
                    if random.random() < self.crossover_rate:
                        dataset = self.crossover_mutation(JailbreakDataset([parent1]), other_instance=parent2)
                        child1 = dataset[0]
                        child2 = dataset[1]
                        mutated_prompts.append(child1.jailbreak_prompt)
                        mutated_prompts.append(child2.jailbreak_prompt)
                    else:
                        mutated_prompts.append(parent1.jailbreak_prompt)
                        mutated_prompts.append(parent2.jailbreak_prompt)
                for i in range(len(mutated_prompts)):
                    if random.random() < self.mutation_rate:
                        mutated_prompts[i] = self.rephrase_mutation.rephrase(mutated_prompts[i])

                # 8. Combine elites with the mutated offspring
                next_generation = elites + mutated_prompts
                assert len(next_generation) == self.batch_size
                instance.candidate_prompts = next_generation
                
                # HIERARCHICAL GENETIC ALGORITHM
                # Sentence-level Iterations
                for s in range(self.sentence_level_steps):
                    # 9. Evaluate the fitness score of each individual in population
                    score_list, current_loss, adv_prefix, response = self.evaluate_candidate_prompts(instance, prefix_manager)

                    # 10. Evaluate jailbreak success or not
                    instance.target_responses.append(response)
                    self.evaluator(JailbreakDataset([instance]))
                    is_success = instance.eval_results[-1]

                    if is_success == 1:
                        break

                    # 11. Calculate momentum word score and Update sentences in each prompt
                    word_dict = self.construct_momentum_word_dictionary(word_dict, instance.candidate_prompts, score_list)
                    self.replace_words_with_synonyms_mutation.update(word_dict)
                    mutation_dataset = JailbreakDataset([])
                    for p in instance.candidate_prompts:
                        mutation_dataset.add(Instance(jailbreak_prompt=p))
                    dataset = self.replace_words_with_synonyms_mutation(mutation_dataset)

                    mutated_prompts = []
                    for d in dataset:
                        mutated_prompts.append(d.jailbreak_prompt)
                    instance.candidate_prompts = mutated_prompts

                # 12. Evaluate the fitness score of each individual in population
                score_list, current_loss, adv_prefix, response = self.evaluate_candidate_prompts(instance, prefix_manager)

                # 13. Evaluate jailbreak success or not
                instance.target_responses.append(response)
                self.evaluator(JailbreakDataset([instance]))
                is_success = instance.eval_results[-1]

                if is_success == 1:
                    epoch_end_time = time.time()
                    epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)
                    print(
                        "################################\n"
                        f"Current Epoch: {j}/{self.num_steps}\n"
                        f"Passed:{is_success}\n"
                        f"Loss:{current_loss.item()}\n"
                        f"Epoch Cost:{epoch_cost_time}\n"
                        f"Current prefix:\n{adv_prefix}\n"
                        f"Current Response:\n{response}\n"
                        "################################\n")
                    gc.collect()
                    torch.cuda.empty_cache()
                    best_prompt = adv_prefix
                    break

                epoch_end_time = time.time()
                epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)
                print(
                    "################################\n"
                    f"Current Epoch: {j}/{self.num_steps}\n"
                    f"Passed:{is_success}\n"
                    f"Loss:{current_loss.item()}\n"
                    f"Epoch Cost:{epoch_cost_time}\n"
                    f"Current prefix:\n{adv_prefix}\n"
                    f"Current Response:\n{response}\n"
                    "################################\n")
                gc.collect()
                torch.cuda.empty_cache()
                best_prompt = adv_prefix

        new_instance = instance.copy()
        new_instance.parents.append(instance)
        instance.children.append(new_instance)
        new_instance.jailbreak_prompt = best_prompt + '{query}'
        return JailbreakDataset([new_instance])


class autodan_PrefixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
        r"""
        :param ~str instruction: the harmful query.
        :param ~str target: the target response for the query.
        :param ~str adv_string: the jailbreak prompt.
        """
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string} {self.instruction} ")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.adv_string}{separator}{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))

                separator = ' ' if self.instruction else ''
                self.conv_template.update_last_message(f"{self.adv_string}{separator}{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks) - 1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)
            else:
                self._system_slice = slice(
                    None,
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        return input_ids
