import json
import os
import random
from typing import List
from .seed_base import SeedBase


class SeedTemplate(SeedBase):
    r"""
    A class that can use template to generate and store attack seeds.
    """
    def __init__(self, seeds: List[str] = None):
        r"""
        Initialize a SeedTemplate instance with a seed list that can be empty.
        :param ~List[str] seeds: Strings that contain the context for jailbreak queries.
        """
        super().__init__(seeds)

    def new_seeds(self, seeds_num= None, prompt_usage='attack', method_list: List[str] = None,
                  template_file=None):
        r"""
        Use template to generate new seeds, replacing the old batch.
        :param ~int seeds_num: Indicates how many seeds users want.
        :param ~str prompt_usage: Indicates whether these seeds are used for attacking or judging.
        :param ~List[str] method_list: Indicates the paper from which the templates originate.
        :param ~str template_file: Indicates the file that stores the templates.
        :return: new_seeds
        """
        self.seeds = []
        if method_list is None:
            method_list = ['default']

        if template_file is None:
            template_file = 'seed_template.json'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_file_path = os.path.join(current_dir, template_file)
            template_dict = json.load(open(template_file_path, 'r', encoding='utf-8'))
        else:
            template_dict = json.load(open(template_file, 'r', encoding='utf-8'))

        template_pool = []
        for method in method_list:
            try:
                template_pool.extend(template_dict[prompt_usage][method])
            except KeyError:
                raise AttributeError("{} contains no {} prompt template from the method {}".
                                     format(template_file, prompt_usage, method))

        if seeds_num is None:
            return template_pool
        else:
            assert seeds_num > 0, "The seeds_num must be a positive integer."
            assert seeds_num <= len(template_pool), \
            'The number of seeds in the template pool is less than the number being asked for.'
            index_list = random.sample(range(len(template_pool)), seeds_num)
            for index in index_list:
                self.seeds.append(template_pool[index])

            return self.seeds

if __name__ == '__main__':
    seedtemplate = SeedTemplate()
    new_seeds = seedtemplate.new_seeds(seeds_num=3, prompt_usage='attack',
                                       method_list=['Gptfuzzer', 'DeepInception', 'ICA', 'ICA'], template_file=None)
    print(new_seeds)
    print(len(new_seeds))
