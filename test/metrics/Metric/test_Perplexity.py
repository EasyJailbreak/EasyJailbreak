import unittest
import sys
import os
sys.path.append(os.getcwd())
from easyjailbreak.metrics.Metric import Perplexity
from easyjailbreak.datasets import Instance, JailbreakDataset

class TestPerplexity(unittest.TestCase):
    def test_instance(self):
        # inital dataset
        dataset = JailbreakDataset([])
        dataset.add(Instance(jailbreak_prompt='Hello.'))
        dataset.add(Instance(jailbreak_prompt='Please answer the following query.'))
        dataset.add(Instance(jailbreak_prompt='How are you?'))

        # inital metric
        metric = Perplexity()

        # calculate ASR metric
        all_metrics = metric.calculate(dataset)

        assert 'avg_prompt_perplexity' in all_metrics.keys()
        
        print(all_metrics['avg_prompt_perplexity'])


if __name__ == '__main__':
    unittest.main()