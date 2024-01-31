from ..mutation_base import MutationBase
from ...datasets import JailbreakDataset
from ...utils import model_utils

from ...models import WhiteBoxModelBase
from ...datasets import JailbreakDataset, Instance

import random
import torch
import logging
from typing import Optional


class MutationTokenGradient(MutationBase):
    def __init__(
            self, 
            attack_model: WhiteBoxModelBase,
            num_turb_sample: Optional[int] = 512,
            top_k: Optional[int] = 256,
            avoid_unreadable_chars: Optional[bool] = True,
            is_universal: Optional[bool] = False):
        """
        Initializes the MutationTokenGradient.

        :param WhiteBoxModelBase attack_model: Model used for the attack.
        :param int num_turb_sample: Number of mutant samples generated per instance. Defaults to 512.
        :param int top_k: Randomly select the target mutant token from the top_k with the smallest gradient values at each position.
            Defaults to 256.
        :param bool avoid_unreadable_chars: Whether to avoid generating unreadable characters. Defaults to True.
        :param bool is_universal: Whether a shared jailbreak prompt is optimized for all instances. Defaults to False.
        """
        self.attack_model = attack_model
        self.num_turb_sample = num_turb_sample
        self.top_k = top_k
        self.avoid_unreadable_chars = avoid_unreadable_chars
        self.is_universal = is_universal
        
    def __call__(
            self,
            jailbreak_dataset: JailbreakDataset
            )->JailbreakDataset:
        """
        Mutates the jailbreak_prompt in the sample based on the token gradient.
        
        :param JailbreakDataset jailbreak_dataset: Dataset for the attack.
        :return: A mutated dataset with the jailbreak prompt based on the token gradient.
        :rtype: JailbreakDataset
        
        .. note::
            - num_turb_sample: Number of mutant samples generated per instance.
            - top_k: Each mutation target is selected from the top_k tokens with the smallest gradient values at each position.
            - is_universal: Whether the jailbreak prompt is shared across all samples. If true, it uses the first sample in the jailbreak_dataset.
            This mutation method is probably the most complex to implement.
            - In case `is_universal=False` and more than one sample is present in the dataset, the method treats each instance separately and merges the results.
        """
        
        # Handle is_universal=False as a special case (multiple datasets of size 1)
        if not self.is_universal and len(jailbreak_dataset) > 1:
            ans = []
            for instance in jailbreak_dataset:
                new_samples = self(JailbreakDataset([instance]))
                ans.append(new_samples)
            return JailbreakDataset.merge(ans)
        # The rest of the implementation assumes is_universal=True

        # tokenizes
        universal_prompt_ids = None
        for instance in jailbreak_dataset:
            if isinstance(instance.reference_responses, str):
                ref_resp = instance.reference_responses
            else:
                ref_resp = instance.reference_responses[0]
            # 拼接成完整的字符串，并给出指示prompt和target在其中所在位置的slice
            instance.jailbreak_prompt = jailbreak_dataset[0].jailbreak_prompt   # 以jailbreak_dataset中的第一个样本为准
            input_ids, query_slice, jbp_slices, response_slice = model_utils.encode_trace(self.attack_model, instance.query, instance.jailbreak_prompt, ref_resp)
            instance._input_ids = input_ids
            instance._query_slice = query_slice
            instance._jbp_slices = jbp_slices
            instance._response_slice = response_slice
            if universal_prompt_ids is not None:
                instance._input_ids[:, jbp_slices[0]] = universal_prompt_ids[0] # 这里发生的是拷贝赋值
                instance._input_ids[:, jbp_slices[1]] = universal_prompt_ids[1]
            else:
                universal_prompt_ids = [input_ids[:, jbp_slices[0]], input_ids[:, jbp_slices[1]]]

        # 计算token gradient
        # 对每个样本分别计算jailbreak_prompt部分的token gradient，之后归一化，加起来，得到最终的jbp_token_grad
        jbp_token_grad = 0   # 1 * L1 * V
        for instance in jailbreak_dataset:
            token_grad = model_utils.gradient_on_tokens(self.attack_model, instance._input_ids, instance._response_slice)   # 1 * L * V
            token_grad = token_grad / (token_grad.norm(dim = 2, keepdim=True) + 1e-6)   # 1 * L * V
            jbp_token_grad += torch.cat([token_grad[:, instance._jbp_slices[0]], token_grad[:, instance._jbp_slices[1]]], dim=1)    # 1 * L1 * V
        L1 = jbp_token_grad.size(1)
        V = jbp_token_grad.size(2)
          
        # 生成变异体
        score_tensor = -jbp_token_grad  # 1 * L1 * V
        if self.avoid_unreadable_chars:
            ignored_ids = model_utils.get_nonsense_token_ids(self.attack_model)
            # logging.debug(f'Token Gradient: id ignored={len(ignored_ids)}/{V}')
            for token_ids in ignored_ids:
                score_tensor[:,:,token_ids] = float('-inf')
        top_k_indices = torch.topk(score_tensor, dim=2, k=self.top_k).indices
        with torch.no_grad():
            # 生成扰动后的
            turbed_prompt_ids_list = []
            for _ in range(self.num_turb_sample):
                new_prompt_ids = [universal_prompt_ids[0].clone(), universal_prompt_ids[1].clone()]   # [1 * L11, 1 * L12]; L11+L12==L1
                # 处理jailbreak_prompt第一个token是bos的特殊情况
                if universal_prompt_ids[0].size(1) >= 1 and universal_prompt_ids[0][0, 0] == self.attack_model.bos_token_id:
                    rel_idx = random.randint(1, L1-1)
                else:
                    rel_idx = random.randint(0, L1-1)   # 要替换jailbreak_prompt中的第几个token
                # 随机选择一个新的token_id
                new_token_id = top_k_indices[0][rel_idx][random.randint(0, self.top_k-1)]
                # 替换
                if rel_idx < new_prompt_ids[0].size(1):
                    new_prompt_ids[0][0][rel_idx] = new_token_id
                else:
                    new_prompt_ids[1][0][rel_idx-new_prompt_ids[0].size(1)] = new_token_id
                turbed_prompt_ids_list.append(new_prompt_ids)
                
            # 生成新的扰动后的dataset
            new_dataset = []
            for instance in jailbreak_dataset:
                new_instance_list = []
                for new_prompt_ids in turbed_prompt_ids_list:
                    new_input_ids = instance._input_ids.clone()
                    new_input_ids[:, instance._jbp_slices[0]] = new_prompt_ids[0]
                    new_input_ids[:, instance._jbp_slices[1]] = new_prompt_ids[1]
                    _, _, jailbreak_prompt, _ = model_utils.decode_trace(self.attack_model, new_input_ids, instance._query_slice, instance._jbp_slices, instance._response_slice)
                    if '\r' in jailbreak_prompt:
                        breakpoint()
                    new_instance = Instance(
                        query=instance.query,
                        jailbreak_prompt=jailbreak_prompt,
                        reference_responses=instance.reference_responses,
                        parents = [instance]
                    )
                    new_instance_list.append(new_instance)
                instance.children = new_instance_list
                new_dataset.extend(new_instance_list)
            
            # 删除所有的张量，释放显存
            for instance in jailbreak_dataset:
                instance.delete('_input_ids')
            return JailbreakDataset(new_dataset)

                    
