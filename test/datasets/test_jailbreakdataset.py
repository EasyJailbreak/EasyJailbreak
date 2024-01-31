import os.path
import unittest
from easyjailbreak.datasets import *
from test.settings import *

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        # test load csv
        dataset1 = JailbreakDataset.load_csv(path = CSV_DATA_PATH)
        self.assertEqual(7, len(dataset1))

        # test load jsonl
        dataset2 = JailbreakDataset.load_jsonl(path = JSON_DATA_PATH)
        self.assertEqual(7, len(dataset2))

        # merge
        dataset3 = JailbreakDataset.merge([dataset1, dataset2])
        self.assertEqual(14, len(dataset3))

        # test save csv
        dataset3.save_to_csv(os.path.join(current_path, 'mini_save.csv'))
        new_dataset = JailbreakDataset.load_csv(path = os.path.join(current_path, 'mini_save.csv'))
        self.assertEqual(14, len(new_dataset))

        # test save jsonl
        dataset3.save_to_jsonl(os.path.join(current_path, 'mini_save.jsonl'))
        new_dataset = JailbreakDataset.load_jsonl(path = os.path.join(current_path, 'mini_save.jsonl'))
        self.assertEqual(14, len(new_dataset))



if __name__ == '__main__':
    unittest.main()