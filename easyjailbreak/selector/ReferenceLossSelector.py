from .selector import SelectPolicy
from ..datasets import JailbreakDataset
from ..datasets import Instance
from ..utils import model_utils
from ..models import WhiteBoxModelBase

import warnings
import torch
import logging

class ReferenceLossSelector(SelectPolicy):
    """
    This class implements a selection policy based on the reference loss. It selects instances from a set of parents
    based on the minimum loss calculated on their reference target, discarding others.
    """
    def __init__(self, model:WhiteBoxModelBase, batch_size=None, is_universal=False):
        """
        Initialize the selector with a model and optional configuration settings.

        :param ~WhiteBoxModelBase model: The model used for calculating loss.
        :param int|None batch_size: The size of each batch for loss calculation. If None, batch_size will be the same as the size of dataset. (default None).
        :param bool is_universal: If True, considers the loss of all instances with the same jailbreak_prompt together (default False).
        """
        assert isinstance(model, WhiteBoxModelBase)
        self.model = model
        self.batch_size = batch_size
        self.is_universal = is_universal

    def select(self, dataset)->JailbreakDataset:
        """
        Selects instances from the dataset based on the calculated reference loss.

        :param ~JailbreakDataset dataset: The dataset from which instances are to be selected.
        :return ~JailbreakDataset: A new dataset containing selected instances with minimum reference loss.
        """
        if not self.is_universal and len(dataset.group_by_parents()) > 1:
            # 将is_universal=False的情况当作True情况的特例来实现
            return JailbreakDataset.merge([self.select(JailbreakDataset(group)) for group in dataset.group_by_parents()])

        if self.batch_size is None:
            batches = [dataset]
        else:
            batches = [dataset[i: i+self.batch_size] for i in range(0, len(dataset), self.batch_size)]

        # calculate loss on reference response
        with torch.no_grad():
            for batch in batches:
                B = len(batch)
                # logging.debug(f'Loss selection: mini-batchsize = {B}')

                # encode
                batch_input_ids = []
                batch_labels = []
                for instance in batch:
                    assert len(instance.reference_responses) >= 1
                    if len(instance.reference_responses) > 1:
                        warnings.warn(f'传入`ReferenceLossSelector`的每个instance的reference_responses大小都为1，而不是{len(instance.reference_responses)}。将默认使用第一个。')

                    input_ids, _, _, target_slice = model_utils.encode_trace(self.model, instance.query, instance.jailbreak_prompt, instance.reference_responses[0])
                    labels = torch.full_like(input_ids, -100)
                    labels[:, target_slice] = input_ids[:, target_slice]
                    batch_input_ids.append(input_ids)
                    batch_labels.append(labels)
                batch_input_ids = model_utils.pad_and_stack(batch_input_ids, self.model.pad_token_id)
                batch_labels = model_utils.pad_and_stack(batch_labels, -100)

                # compute loss values for each instance in batch
                batch_loss = model_utils.batch_loss(self.model, batch_input_ids, batch_labels)    # B
                for idx, instance in enumerate(batch):
                    instance._loss = batch_loss[idx].item()
            
        # select
        best_group = None
        best_loss = None
        for group in dataset.group_by(lambda x: x.jailbreak_prompt):
            total_loss = sum([instance._loss for instance in group])
            if best_loss is None or total_loss < best_loss:
                best_loss = total_loss
                best_group = group
        logging.info(f'Loss selection: best loss = {best_loss}')
        logging.info(f'Loss Selection: best jailbreak prompt = `{best_group[0].jailbreak_prompt}`')

        return JailbreakDataset(best_group)