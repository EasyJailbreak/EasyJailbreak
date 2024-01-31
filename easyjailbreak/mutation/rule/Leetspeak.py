from typing import List
from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import Instance

class Leetspeak(MutationBase):
    """
    Mutation rule for Leetspeak
    """
    def __init__(self, attr_name='query'):
        self.attr_name = attr_name
    
    def _get_mutated_instance(self, instance) -> List[Instance]:
        if not hasattr(instance, self.attr_name):
            raise AttributeError(f"Attribute '{self.attr_name}' not found in instance")
        
        mutated_results = []
        seed = getattr(instance,self.attr_name)
        new_seed = self.leetspeak(seed)
        new_instance = instance.copy()
        setattr(new_instance, self.attr_name, new_seed)
        if new_instance.jailbreak_prompt is None:
             new_instance.jailbreak_prompt = """{query}"""
        new_instance.children.append(instance)
        instance.children.append(new_instance)
        mutated_results.append(new_instance)
        
        return mutated_results
    
    def leetspeak(self, seed):
        """
        Leeetspeak the seed.
        """
        leet_dict = {
            'a': '@',
            'e': '3',
            'i': '!',
            'o': '0',
            'are': 'r',
            'be': 'b'
        }
        def to_leetspeak(s):
            for key, val in leet_dict.items():
                s = s.replace(key, val)
            return s
        seed = to_leetspeak(seed)
        return seed
    

