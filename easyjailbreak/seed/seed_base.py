from typing import List


class SeedBase:
    r"""
    A base class that can store and generate attack seeds.
    """
    def __init__(self, seeds: List[str] = None):
        r"""
        Initialize a SeedBase instance with a seed list that can be empty.
        :param ~List[str] seeds: Strings that contain the context for jailbreak queries.
        """
        if seeds is None:
            seeds = []
        self.seeds = seeds

    def __iter__(self):
        return self.seeds.__iter__()

    def new_seeds(self, **kwargs):
        r"""
        Generate new seeds, replacing the old batch.
        :param kwargs: Possible keywords for the generation process.
        :return: new_seeds
        """
        raise NotImplementedError
