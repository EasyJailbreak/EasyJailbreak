import unittest
import sys
import os
import random
sys.path.append(os.getcwd())
from easyjailbreak.metrics.Metric import AttackSuccessRate
from easyjailbreak.datasets import Instance, JailbreakDataset

class TestASR(unittest.TestCase):
    def test_instance(self):
        # inital dataset
        dataset = JailbreakDataset([])
        for i in range(10):
            dataset.add(Instance(eval_results=[random.choice([0, 1])]))

        # inital metric
        metric = AttackSuccessRate()

        # calculate ASR metric
        all_metrics = metric.calculate(dataset)

        assert 'successful_attacks' in all_metrics.keys()
        assert 'failed_attacks' in all_metrics.keys()
        assert 'attack_success_rate' in all_metrics.keys()
        assert all_metrics['attack_success_rate'] == round(all_metrics['successful_attacks'] * 100.0 / len(dataset), 2)
        print(all_metrics['successful_attacks'])
        print(all_metrics['attack_success_rate'])


if __name__ == '__main__':
    unittest.main()