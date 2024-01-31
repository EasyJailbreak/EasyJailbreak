"""
This file contains a wrapper for Huggingface models, implementing various methods used in downstream tasks.
It includes the HuggingfaceModel class that extends the functionality of the WhiteBoxModelBase class.
"""

import sys
from .model_base import WhiteBoxModelBase
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
import functools
import torch
from fastchat.conversation import get_conv_template
from typing import Optional, Dict, List, Any
import logging

class HuggingfaceModel(WhiteBoxModelBase):
    """
    HuggingfaceModel is a wrapper for Huggingface's transformers models.
    It extends the WhiteBoxModelBase class and provides additional functionality specifically
    for handling conversation generation tasks with various models.
    This class supports custom conversation templates and formatting,
    and offers configurable options for generation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the HuggingfaceModel with a specified model, tokenizer, and generation configuration.

        :param Any model: A huggingface model.
        :param Any tokenizer: A huggingface tokenizer.
        :param str model_name: The name of the model being used. Refer to
            `FastChat conversation.py <https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py>`_
            for possible options and templates.
        :param Optional[Dict[str, Any]] generation_config: A dictionary containing configuration settings for text generation.
            If None, a default configuration is used.
        """
        super().__init__(model, tokenizer)
        self.model_name = model_name
        try:
            self.conversation = get_conv_template(model_name)
        except KeyError:
            logging.error(f'Invalid model_name: {model_name}. Refer to '
                          'https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py '
                          'for possible options and templates.')
            raise  # Continue raising the KeyError

        if model_name == 'llama-2':
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == 'zero_shot':
            self.conversation.roles = tuple(['### ' + r for r in self.conversation.template.roles])
            self.conversation.sep = '\n'

        self.format_str = self.create_format_str()

        if generation_config is None:
            generation_config = {}
        self.generation_config = generation_config

    def set_system_message(self, system_message: str):
        r"""
        Sets a system message to be used in the conversation.

        :param str system_message: The system message to be set for the conversation.
        """
        # TODO check llama2 add system prompt
        self.conversation.system_message = system_message

    def create_format_str(self):
        self.conversation.messages = []
        self.conversation.append_message(self.conversation.roles[0], "{prompt}")
        self.conversation.append_message(self.conversation.roles[1], "{response}")
        format_str = self.conversation.get_prompt()
        self.conversation.messages = []  # clear history
        return format_str

    def create_conversation_prompt(self, messages, clear_old_history=True):
        r"""
        Constructs a conversation prompt that includes the conversation history.

        :param list[str] messages: A list of messages that form the conversation history.
                                   Messages from the user and the assistant should alternate.
        :param bool clear_old_history: If True, clears the previous conversation history before adding new messages.
        :return: A string representing the conversation prompt including the history.
        """
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
        self.conversation.append_message(self.conversation.roles[-1], None)
        return self.conversation.get_prompt()

    def clear_conversation(self):
        r"""
        Clears the current conversation history.
        """
        self.conversation.messages = []

    def generate(self, messages, input_field_name='input_ids', clear_old_history=True, **kwargs):
        r"""
        Generates a response for the given messages within a single conversation.

        :param list[str]|str messages: The text input by the user. Can be a list of messages or a single message.
        :param str input_field_name: The parameter name for the input message in the model's generation function.
        :param bool clear_old_history: If True, clears the conversation history before generating a response.
        :param dict kwargs: Optional parameters for the model's generation function, such as 'temperature' and 'top_p'.
        :return: A string representing the pure response from the model, containing only the text of the response.
        """
        if isinstance(messages, str):
            messages = [messages]
        prompt = self.create_conversation_prompt(messages, clear_old_history=clear_old_history)

        input_ids = self.tokenizer(prompt,
                                   return_tensors='pt',
                                   add_special_tokens=False).input_ids.to(self.model.device.index)
        input_length = len(input_ids[0])

        kwargs.update({input_field_name: input_ids})
        output_ids = self.model.generate(**kwargs, **self.generation_config)
        output = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        return output

    def batch_generate(self, conversations, **kwargs)-> List[str]:
        r"""
        Generates responses for a batch of conversations.

        :param list[list[str]]|list[str] conversations: A list of conversations. Each conversation can be a list of messages
                                                         or a single message string. If a single string is provided, a warning
                                                         is issued, and the string is treated as a single-message conversation.
        :param dict kwargs: Optional parameters for the model's generation function.
        :return: A list of responses, each corresponding to one of the input conversations.
        """
        prompt_list = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn('If you want the model to generate batches based on several conversations, '
                              'please construct a list[str] for each conversation, or they will be divided into individual sentences. '
                              'Switch input type of batch_generate() to list[list[str]] to avoid this warning.')
                conversation = [conversation]
            prompt_list.append(self.create_conversation_prompt(conversation))
        input_ids = self.tokenizer(prompt_list,
                                   return_tensors='pt',
                                   padding=True,
                                   add_special_tokens=False)
        input_ids = {k: v.to(self.model.device.index) for k, v in input_ids.items()}
        kwargs.update(**input_ids)
        output_ids = self.model.generate(**kwargs, **self.generation_config)
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_ids["input_ids"].shape[1]:]
        output_list = self.batch_decode(output_ids)

        return output_list

    def __call__(self, *args, **kwargs):
        r"""
        Allows the HuggingfaceModel instance to be called like a function, which internally calls the model's
        __call__ method.

        :return: The output from the model's __call__ method.
        """
        return self.model(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        r"""
        Tokenizes the input using the model's tokenizer.

        :return: The tokenized output.
        """
        return self.tokenizer.tokenize(*args, **kwargs)

    def batch_encode(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs)-> List[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def format(self, **kwargs):
        return self.format_str.format(**kwargs)

    def format_instance(self, query, jailbreak_prompt, response):
        prompt = jailbreak_prompt.replace('{query}', query) # 之所以不用str.format方法，是因为jailbreak_prompt可能是任意字符串，其中可能包含{...}，而str.format不支持忽略未提供的key
        return self.format(prompt=prompt, response=response)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    #@functools.cache  # 记忆结果，避免重复计算。不影响子类的重载。
    def embed_layer(self) -> Optional[torch.nn.Embedding]:
        """
        Retrieve the embedding layer object of the model.

        This method provides two commonly used approaches to search for the embedding layer in Hugging Face models.
        If these methods are not effective, users should consider manually overriding this method.

        Returns:
            torch.nn.Embedding or None: The embedding layer object if found, otherwise None.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                return module

        for module in self.model.modules():
            if all(hasattr(module, attr) for attr in ['bos_token_id', 'eos_token_id', 'encode', 'decode', 'tokenize']):
                return module
        return None

    @property
    #@functools.cache
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        This method provides two commonly used approaches for obtaining the vocabulary size in Hugging Face models.
        If these methods are not effective, users should consider manually overriding this method.

        Returns:
            int: The size of the vocabulary.
        """
        if hasattr(self, 'config') and hasattr(self.config, 'vocab_size'):
            return self.config.vocab_size

        return self.embed_layer.weight.size(0)


def from_pretrained(model_name_or_path: str, model_name: str, tokenizer_name_or_path: Optional[str] = None,
                    dtype: Optional[torch.dtype] = None, **generation_config: Dict[str, Any]) -> HuggingfaceModel:
    """
    Imports a Hugging Face model and tokenizer with a single function call.

    :param str model_name_or_path: The identifier or path for the pre-trained model.
    :param str model_name: The name of the model, used for generating conversation template.
    :param Optional[str] tokenizer_name_or_path: The identifier or path for the pre-trained tokenizer.
        Defaults to `model_name_or_path` if not specified separately.
    :param Optional[torch.dtype] dtype: The data type to which the model should be cast.
        Defaults to None.
    :param generation_config: Additional configuration options for model generation.
    :type generation_config: dict

    :return HuggingfaceModel: An instance of the HuggingfaceModel class containing the imported model and tokenizer.

    .. note::
        The model is loaded for evaluation by default. If `dtype` is specified, the model is cast to the specified data type.
        The `tokenizer.padding_side` is set to 'right' if not already specified.
        If the tokenizer has no specified pad token, it is set to the EOS token, and the model's config is updated accordingly.

    **Example**

    .. code-block:: python

        model = from_pretrained('bert-base-uncased', 'bert-model', dtype=torch.float32, max_length=512)
    """
    if dtype is None:
        dtype = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    if tokenizer.padding_side is None:
        tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return HuggingfaceModel(model, tokenizer, model_name=model_name, generation_config=generation_config)