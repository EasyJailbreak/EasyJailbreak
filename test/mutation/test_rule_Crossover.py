import unittest
import sys
import os
sys.path.append(os.getcwd())
from easyjailbreak.mutation.rule import CrossOver
from easyjailbreak.datasets import Instance, JailbreakDataset

class TestCrossOver(unittest.TestCase):
    def test_instance(self):
        # inital instance
        instance1 = Instance(query='query1', jailbreak_prompt='Test1. Test2. Test3')
        instance2 = Instance(query="query2", jailbreak_prompt="Prompt1. Prompt2. Prompt3")
      
        # inital mutation
        mutation = CrossOver(attr_name='jailbreak_prompt', num_points=1, seed_pool=None)

        # 变异jailbreak_prompt
        mutated_instances = mutation(JailbreakDataset([instance1]), other_instance=instance2)
        assert len(mutated_instances) == 2
        assert mutated_instances[0].query == 'query1'
        print(mutated_instances[0].jailbreak_prompt)
        assert mutated_instances[0].parents == [instance1]
        assert mutated_instances[1].query == 'query2'
        assert mutated_instances[1].parents == [instance1]
        print(mutated_instances[1].jailbreak_prompt)

        # inital instance
        instance1 = Instance(query="Test1. Test2. Test3", jailbreak_prompt="prompt1")
        instance2 = Instance(query="Query1. Query2. Query3", jailbreak_prompt="prompt2")
        
        # inital mutation
        mutation = CrossOver(attr_name='query', num_points=1, seed_pool=None)

        # 变异query
        mutated_instances = mutation(JailbreakDataset([instance1]), other_instance=instance2)
        assert len(mutated_instances) == 2
        assert mutated_instances[0].jailbreak_prompt == 'prompt1'
        print(mutated_instances[0].query)
        assert mutated_instances[0].parents == [instance1]
        assert mutated_instances[1].jailbreak_prompt == 'prompt2'
        assert mutated_instances[1].parents == [instance1]
        print(mutated_instances[1].query)

if __name__ == '__main__':
    unittest.main()
