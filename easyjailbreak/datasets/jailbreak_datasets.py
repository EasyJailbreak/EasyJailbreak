"""
Jailbreak_Dataset Module
========================
This module provides the JailbreakDataset class, which is designed to manage and manipulate datasets for the Easy Jailbreak application. It is capable of handling datasets structured with PromptNode instances, offering functionalities such as shuffling, accessing, and processing data points in an organized way for machine learning tasks related to Easy Jailbreak.
"""

import csv
import logging
import json
import os.path
import random
import sys
from typing import List, Union

import torch
from easyjailbreak.datasets import Instance
from datasets import load_dataset

__all__ = ["JailbreakDataset"]

class JailbreakDataset(torch.utils.data.Dataset):
    """
    JailbreakDataset class is designed for handling datasets specifically structured for the Easy Jailbreak application.
    It allows for the representation, manipulation, and access of data points in the form of Instance instances.
    This class provides essential functionalities such as shuffling, accessing, and formatting data for use in machine learning models.
    """

    def __init__(
        self,
        dataset: Union[List[Instance], str],
        shuffle: bool = False,
        local_file_type: str = 'json'
    ):
        """
        Initializes the JailbreakDataset with either a list of Instance instances or a dataset name.

        :param Union[List[Instance], str] dataset: A list of Instance instances or the name of a dataset to be loaded.
        :param bool shuffle: Specifies whether to shuffle the dataset upon initialization. Defaults to False.
        :param str local_file_type: Type of the local dataset file if `dataset` is a file path. Defaults to 'json'.
            Specify 'json' or 'jsonl'.
        """

        if isinstance(dataset, list):
            self._dataset = dataset
        elif isinstance(dataset, str):
            if os.path.exists(dataset):
                logging.warning('You chose to load a dataset from a local file. If it is not a json/jsonl file, please make sure you correctly set \'local_file_type\' parameter when you instantiate the JailbreakDataset.')
                raw_dataset = load_dataset(local_file_type, data_files=dataset)
            else:
                logging.warning(f'{dataset} is not a local dataset file. Try to load from huggingface repository `Lemhf14/EasyJailbreak_Datasets`.')
                raw_dataset = load_dataset("Lemhf14/EasyJailbreak_Datasets", dataset)
            instance_list = [Instance(**sample) for sample in raw_dataset['train']]
            self._dataset = instance_list
        else:
            logging.error('The dataset parameter for Jailbreak_Dataset should be a list of `Instance` or a string indicates the subset you want from Lemhf14/EasyJailbreak_Datasets')
            raise ValueError(f'Invalid dataset parameter `{dataset}`')

        self.shuffled = shuffle

        if shuffle:
            random.shuffle(self._dataset)

    @staticmethod
    def load_csv(path = 'data.csv', headers:List[int] = None):
        r"""
        Loads a CSV file into the dataset.

        :param str path: The path of the CSV file to be loaded.
        :param list[str] headers: A list of column names to be used as headers. Defaults to None.
        """
        import pandas as pd
        dataset = JailbreakDataset([])
        with open(path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset.add(Instance(**dict(row)))
        return dataset

    @staticmethod
    def load_jsonl(path = 'data.jsonl'):
        r"""
        Loads a JSONL file into the dataset.

        :param str path: The path of the JSONL file to be loaded.
        """
        import jsonlines

        dataset = JailbreakDataset([])
        with jsonlines.open(path) as reader:
            for obj in reader:
                dataset.add(Instance(**obj))
        return dataset


    def _format_as_dict(self, Instance: Instance):
        r"""
        Converts an Instance into a dictionary format, suitable for serialization or further processing.

        :param Instance instance: The Instance to be converted into dictionary format.
        :return dict: A dictionary representation of the Instance.
        """
        Instance_dict = Instance.__dict__
        return Instance_dict

    def shuffle(self):
        r"""
        Shuffles the dataset in place.

        This method randomizes the order of the dataset's elements and updates the `shuffled` attribute to True.
        """
        random.shuffle(self._dataset)
        self.shuffled = True

    def __getitem__(self, key):
        r"""
        Retrieves an item or a subset from the dataset based on the provided index or slice.

        :param int key: The index of the item to retrieve, or a slice indicate subset.
        :return Instance | JailbreakDataset: The Instance at the specified index, or a subset dataset.
        """
        if isinstance(key, slice):
            return JailbreakDataset(self._dataset[key])
        else:
            return self._dataset[key]

    def __len__(self):
        r"""
        Returns the total number of items in the dataset.

        :return int: The size of the dataset.
        """
        return len(self._dataset)

    def __iter__(self):
        r"""
        Provides an iterator over the dataset.

        :return iterator: An iterator for the dataset.
        """
        return iter(self._dataset)

    def add(self, Instance: Instance):
        r"""
        Adds a new Instance to the dataset.

        :param Instance instance: The Instance to be added to the dataset.
        """
        self._dataset.append(Instance)

    @classmethod
    def merge(cls, dataset_list):
        r"""
        Merges multiple JailbreakDataset instances into a single dataset.

        :param list[JailbreakDataset] dataset_list: A list of JailbreakDataset instances to be merged.
        :return JailbreakDataset: A new JailbreakDataset instance containing merged data from the provided datasets.
        """
        ans = []
        for dataset in dataset_list:
            ans.extend(dataset._dataset)
        return cls(ans)

    
    def save_to_jsonl(self, path='data.jsonl'):
        """
        Saves the dataset to a JSONL file using jsonlines library.

        :param str path: The path of the file where the dataset will be saved. Defaults to 'data.jsonl'.
        """
        import jsonlines

        with jsonlines.open(path, mode='w') as writer:
            for instance in self._dataset:
                data = {
                    'jailbreak_prompt': getattr(instance, 'jailbreak_prompt', None),
                    'query': getattr(instance, 'query', None),
                    'target_responses': getattr(instance, 'target_responses', None),
                    'eval_results': getattr(instance, 'eval_results', None)
                }
                writer.write(data)

    def save_to_csv(self, path = 'data.csv'):
        r"""
        Saves the dataset to a CSV file.

        :param str path: The path of the file where the dataset will be saved. Defaults to 'data.csv'.
        """
        data_list = [{'jailbreak_prompt': instance.jailbreak_prompt,
                    'query': instance.query,
                    'target_responses': str(instance.target_responses),
                    'eval_results': str(instance.eval_results)} for instance in self._dataset]

        if data_list:
            fieldnames = data_list[0].keys()

            with open(path , 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for data in data_list:
                    writer.writerow(data)

    def group_by(self, key):
        r"""
        Groups instances in the dataset based on a specified key function.

        :param function key: A function that takes an Instance and returns a hashable object for grouping.
        :return list[list[Instance]]: A list of lists, where each sublist contains Instances grouped by the specified key.
        """
        ans = dict()

        for instance in self:
            instance_key = key(instance)
            if instance_key not in ans:
                ans[instance_key] = []
            ans[instance_key].append(instance)
        return list(ans.values())

    def group_by_parents(self):
        r"""
        Groups instances in the dataset based on their parent nodes.

        :return list[list[Instance]]: A list of lists, where each sublist contains Instances grouped by their parent nodes.
        """
        return self.group_by(lambda x: tuple(sorted(list(set(id(parent) for parent in x.parents)))))



if __name__ == '__main__':
    dataset = JailbreakDataset.load_csv(path = r'D:\pycharmProject\EasyJailbreak\test\mini.csv', headers=[0,1])
    dataset.save_to_json(path = r'D:\pycharmProject\EasyJailbreak\test\mini.jsonl')