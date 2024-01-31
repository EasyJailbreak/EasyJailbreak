import unittest
from easyjailbreak.datasets import *

class TestInstance(unittest.TestCase):
    def test_instance(self):
        # inital instance
        instance1 = Instance(query='test', jailbreak_prompt='test_prompt')
        assert instance1.target_responses == []
        assert instance1.eval_results == []
        instance2 = Instance(query='test', jailbreak_prompt='test_prompt',reference_responses=['test1','test2'],target_responses=['test1','test2'],eval_results=[1,1])
        instance2.parents.append(instance1)
        instance1.children.append(instance2)

        assert instance2.query == 'test'
        assert instance2.jailbreak_prompt == 'test_prompt'
        assert instance2.reference_responses == ['test1','test2']
        assert instance2.target_responses == ['test1','test2']
        assert instance2.eval_results == [1,1]
        assert instance2.num_query == 2
        assert instance2.num_jailbreak == 2
        assert instance2.num_reject == 0


        print(instance1.to_dict())
        print(instance2.to_dict())
        # test to_dict
        assert instance1.to_dict() == {'query': 'test', 'jailbreak_prompt': 'test_prompt', 'reference_responses': [], 'target_responses': [], 'eval_results': [], 'index': None, 'attack_attrs': {'Mutation': None, 'query_class': None}}
        assert instance2.to_dict() == {'query': 'test', 'jailbreak_prompt': 'test_prompt', 'reference_responses': ['test1', 'test2'], 'target_responses': ['test1', 'test2'], 'eval_results': [1, 1], 'index': None, 'attack_attrs': {'Mutation': None, 'query_class': None}}



        # test copy
        instance3 = instance2.copy()
        assert instance3.query == 'test'
        assert instance3.jailbreak_prompt == 'test_prompt'
        assert instance3.reference_responses == ['test1','test2']
        assert instance3.target_responses == ['test1','test2']
        assert instance3.eval_results == [1,1]
        assert instance3.num_query == 2
        assert instance3.num_jailbreak == 2
        assert instance3.num_reject == 0
        assert instance3.parents == instance2.parents

        # test delete
        instance3.delete('query')
        with self.assertRaises(AttributeError):
            assert instance3.query == None
        instance3.delete('jailbreak_prompt')
        with self.assertRaises(AttributeError):
            assert instance3.jailbreak_prompt == None



if __name__ == '__main__':
    unittest.main()