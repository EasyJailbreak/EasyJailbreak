from typing import List
from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import Instance

class Disemvowel(MutationBase):
    """
    Mutation rule for Disemvowel
    """
    def __init__(self, attr_name='query'):
    
        self.attr_name = attr_name
    
    def _get_mutated_instance(self, instance) -> List[Instance]:
        if not hasattr(instance, self.attr_name):
            raise AttributeError(f"Attribute '{self.attr_name}' not found in instance")
        
        mutated_results = []
        seed = getattr(instance, self.attr_name)
        new_seed = self.disemvowel(seed)
        new_instance = instance.copy()
        setattr(new_instance, self.attr_name, new_seed)
        if new_instance.jailbreak_prompt is None:
             new_instance.jailbreak_prompt = """{query}"""
        new_instance.parents.append(instance)
        instance.children.append(new_instance)
        mutated_results.append(new_instance)
        
        return mutated_results
    
    def disemvowel(self, seed):
        """
        Disemvowel the seed.
        """
        seed = ''.join([char for char in seed if char not in 'aeiouAEIOU'])
        return seed

    

