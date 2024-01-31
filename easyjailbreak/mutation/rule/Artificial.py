from typing import List
from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import Instance
from easyjailbreak.seed import SeedTemplate


class Artificial(MutationBase):
    """
    Artificial mutation rule for Jailbroken
    """
    def __init__(self, attr_name='query'):
        self.attr_name = attr_name

    def _get_mutated_instance(self, instance) -> List[Instance]:
        
        if not hasattr(instance, self.attr_name):
            raise AttributeError(f"Attribute '{self.attr_name}' not found in instance")
        
        mutated_results = []
        prompt_seeds = SeedTemplate().new_seeds(method_list=['Jailbroken'],template_file="easyjailbreak\seed\seed_template.json")
        
        for prompt_seed in prompt_seeds:
            new_instance = instance.copy()
            for value in prompt_seed.values():
                new_instance.jailbreak_prompt = value
            new_instance.parents.append(instance)
            mutated_results.append(new_instance)

        return mutated_results
    

