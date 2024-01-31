import unittest

from easyjailbreak.seed import SeedRandom


class TestSeedRandom(unittest.TestCase):

    def test_init(self):
        # 测试初始化方法
        seeds_example = ["seed1", "seed2"]
        seed_random = SeedRandom(seeds=seeds_example, seeds_num=2, seeds_max_length=50, early_stop_possibility=0.1)
        self.assertEqual(seed_random.seeds, seeds_example)
        self.assertEqual(seed_random.seeds_num, 2)
        self.assertEqual(seed_random.seeds_max_length, 50)
        self.assertAlmostEqual(seed_random.early_stop_possibility, 0.1)

    def test_new_seeds_count(self):
        # 测试 new_seeds 方法生成的种子数量是否正确
        seed_random = SeedRandom(seeds_num=5)
        seeds = seed_random.new_seeds()
        self.assertEqual(len(seeds), 5)

    def test_new_seeds_length(self):
        # 测试生成的种子长度是否符合要求
        max_length = 10
        seed_random = SeedRandom(seeds_num=5, seeds_max_length=max_length)
        seeds = seed_random.new_seeds()
        for seed in seeds:
            self.assertTrue(len(seed) <= max_length)

    def test_early_stop_effect(self):
        # 测试提前停止的效果
        seed_random = SeedRandom(seeds_num=10, seeds_max_length=100, early_stop_possibility=1.0)
        seeds = seed_random.new_seeds()
        for seed in seeds:
            self.assertEqual(len(seed), 1)  # 由于提前停止概率为 1，所以每个种子应该只有一个字符

if __name__ == "__main__":
    unittest.main()
