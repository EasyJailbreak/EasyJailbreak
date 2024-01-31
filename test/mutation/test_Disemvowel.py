import unittest
from easyjailbreak.mutation.rule import Disemvowel
from easyjailbreak.datasets import Instance, JailbreakDataset

class Test_Disemvowel(unittest.TestCase):
    """
    This class is a unittest for Disemvowel.
    """
    def test_instance(self):
        # inital instance
        instance1 = Instance(query='test')
        # inital mutation
        mutation = Disemvowel(attr_name='query')
        
        # execute mutation
        jailbreakdataset_single = JailbreakDataset([instance1])
        mutated_instances = mutation(jailbreakdataset_single)
        assert len(mutated_instances) == 1
        assert mutated_instances[0].query == 'tst'
        assert mutated_instances[0].jailbreak_prompt == """{query}"""
        assert mutated_instances[0].parents == [instance1]

        # test attr_name not found
        instance2 = Instance(query='test')
        mutation = Disemvowel(attr_name='test')
        jailbreakdataset_single = JailbreakDataset([instance2])
        with self.assertRaises(AttributeError):
            mutated_instances = mutation(jailbreakdataset_single)
        
        # test jailbreak_prompt can be replaced
        instance3 = Instance(query='test', jailbreak_prompt='replaced')
        mutation = Disemvowel()
        jailbreakdataset_single = JailbreakDataset([instance3])
        mutated_instances = mutation(jailbreakdataset_single)
        assert mutated_instances[0].jailbreak_prompt == "replaced"


if __name__ == '__main__':
    unittest.main()