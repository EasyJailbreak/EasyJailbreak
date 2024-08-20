import importlib.resources
from typing import List
from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import Instance
from easyjailbreak.seed import SeedTemplate


class Inception(MutationBase):
    """
    Incetion mutation rule for query.
    """
    def __init__(self, attr_name='query'):
        self.attr_name = attr_name

    def _get_mutated_instance(self, instance) -> List[Instance]:

        if not hasattr(instance, self.attr_name):
            raise AttributeError(f"Attribute '{self.attr_name}' not found in"
                                 " instance")

        mutated_results = []
        seed_path = importlib.resources.files("easyjailbreak.seed") / "seed_template.json"
        prompt_seeds = SeedTemplate().new_seeds(
            method_list=['DeepInception'],
            template_file=seed_path
        )
        new_instance = instance.copy()
        new_instance.jailbreak_prompt = prompt_seeds[-1]
        new_instance.parents.append(instance)
        mutated_results.append(new_instance)

        return mutated_results


