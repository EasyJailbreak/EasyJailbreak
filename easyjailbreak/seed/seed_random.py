import random
from typing import List
from .seed_base import SeedBase


class SeedRandom(SeedBase):
    r"""
    A class that can randomly generate and store attack seeds.
    """
    def __init__(self, seeds: List[str] = None, posible_tokens: List[str] = None, seeds_num=1, seeds_max_length=100, early_stop_possibility=0.):
        r"""
        Initialize a SeedRandom instance with a seed list that can be empty.
        :param ~List[str] seeds: Strings that contain the context for jailbreak queries.
        :param ~int seeds_num: Indicates how many seeds users want.
        :param ~int seeds_max_length: Indicates the maximum length a seed can have.
        :param ~List[str] posible_tokens: Strings that will be randomly added to seed jailbreakprompt.
        :param ~float early_stop_possibility: Indicates the possibility of aborting generation,
                                              used to generate seeds with different lengths.
        """
        super().__init__(seeds)
        self.seeds_num = seeds_num
        self.seeds_max_length = seeds_max_length
        self.posible_tokens = posible_tokens
        self.early_stop_possibility = early_stop_possibility

    def new_seeds(self):
        r"""
        Use template to generate new seeds, replacing the old batch.
        :return: new_seeds
        """
        seeds = []
        for _ in range(self.seeds_num):
            seed = ''
            for _ in range(self.seeds_max_length):
                seed += random.choice(self.posible_tokens)
                if random.uniform(0, 1) < self.early_stop_possibility:
                    break
            seeds.append(seed)
        self.seeds = seeds
        return self.seeds