import unittest
from unittest.mock import MagicMock
from easyjailbreak.mutation.rule import Auto_payload_splitting
from easyjailbreak.datasets import Instance, JailbreakDataset

class Test_Auto_payload_splitting(unittest.TestCase):
    """
    This is a unittest for Auto_payload_splitting.
    """
    def test_instance(self):
        # inital instance
        instance1 = Instance(query='test')
        # inital mutation
        self.mock_model = MagicMock()
        self.mock_model.generate = MagicMock(return_value='Yes!')
        mutation = Auto_payload_splitting(model=self.mock_model)
        # execute mutation
        jailbreakdataset_single = JailbreakDataset([instance1])
        mutated_instances = mutation(jailbreakdataset_single)
        assert len(mutated_instances) == 1
        assert mutated_instances[0].query == 'Yes!'
        assert mutated_instances[0].jailbreak_prompt == "{query}"
        assert mutated_instances[0].parents == [instance1]

        # test attr_name not found
        instance2 = Instance(query='test')
        mutation = Auto_payload_splitting(model=self.mock_model,attr_name='test')
        jailbreakdataset_single = JailbreakDataset([instance2])
        with self.assertRaises(AttributeError):
            mutated_instances = mutation(jailbreakdataset_single)
        
        # test jailbreak_prompt can be replaced
        instance3 = Instance(query='test', jailbreak_prompt='replaced')
        mutation = Auto_payload_splitting(model=self.mock_model)
        jailbreakdataset_single = JailbreakDataset([instance3])
        mutated_instances = mutation(jailbreakdataset_single)
        assert mutated_instances[0].jailbreak_prompt == "replaced"
        
if __name__ == '__main__':
    unittest.main()