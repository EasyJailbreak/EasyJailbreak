from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.seed import SeedTemplate
import random
import re

class CrossOver(MutationBase):
    r"""
    The 'CrossOver' class serves to interlace sentences from two distinct texts.
    """
    def __init__(self, attr_name='jailbreak_prompt', num_points=5, 
                 seed_pool:JailbreakDataset=None):
        r"""
        Initializes the CrossOver instance.
        :param ~str attr_name: The attribute name in the instance that contains the text to be mutated.
        :param ~int num_points: The number of break points when crossover.
        :param ~JailbreakDataset seed_pool: The default cross-over seed pool.
        """
        self.attr_name = attr_name
        self.num_points = num_points
        self.seed_pool = seed_pool
        if seed_pool is None:
            self.seed_pool = SeedTemplate().new_seeds(seeds_num=10, prompt_usage='attack', method_list=["AutoDAN-a"], template_file='./easyjailbreak/seed/seed_template.json')
            self.seed_pool = JailbreakDataset([Instance(jailbreak_prompt=prompt) for prompt in self.seed_pool])


    def _get_mutated_instance(self, instance, **kwargs):
        r"""
        Generates a mutated instance of the given object.
        :param instance: The instance to be mutated.
        :param kwargs: Additional keyword arguments.
        :return: A list containing the mutated instance.
        """
        mutated_instances = []
        seed = getattr(instance, self.attr_name)
        if 'other_instance'  in kwargs:
            other_instance = kwargs['other_instance']
            other_seed = getattr(other_instance, self.attr_name)
        else:
            other_instance = random.choice(self.seed_pool._dataset)
            other_seed = getattr(other_instance, self.attr_name)
        child1, child2 = self.crossover(seed, other_seed, self.num_points)
        
        new_instance1 = instance.copy()
        setattr(new_instance1, self.attr_name, child1)
        new_instance2 = other_instance.copy()
        setattr(new_instance2, self.attr_name, child2)
        mutated_instances.append(new_instance1)
        mutated_instances.append(new_instance2)
        return mutated_instances

    def crossover(self, str1, str2, num_points):
        r"""
        The function determines the feasible points for intertwining or crossing over. 
        :return: two sentences after crossover
        """
        sentences1 = [s for s in re.split('(?<=[.!?])\s+', str1) if s]
        sentences2 = [s for s in re.split('(?<=[.!?])\s+', str2) if s]

        max_swaps = min(len(sentences1), len(sentences2)) - 1
        num_swaps = min(num_points, max_swaps)

        if num_swaps >= max_swaps:
            return str1, str2

        swap_indices = sorted(random.sample(range(1, max_swaps), num_swaps))

        new_str1, new_str2 = [], []
        last_swap = 0
        for swap in swap_indices:
            if random.choice([True, False]):
                new_str1.extend(sentences1[last_swap:swap])
                new_str2.extend(sentences2[last_swap:swap])
            else:
                new_str1.extend(sentences2[last_swap:swap])
                new_str2.extend(sentences1[last_swap:swap])
            last_swap = swap

        if random.choice([True, False]):
            new_str1.extend(sentences1[last_swap:])
            new_str2.extend(sentences2[last_swap:])
        else:
            new_str1.extend(sentences2[last_swap:])
            new_str2.extend(sentences1[last_swap:])

        return ' '.join(new_str1), ' '.join(new_str2)
