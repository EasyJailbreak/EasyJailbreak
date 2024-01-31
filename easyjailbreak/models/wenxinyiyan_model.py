"""
Wenxinyiyan Class
============================================
This class provides methods to interact with Baidu's Wenxin Workshop API for generating responses using an attack model.
It includes methods for obtaining an access token and for sending requests to the API.
https://cloud.baidu.com/?from=console
"""
from typing import List
import requests
import json
import warnings
from .model_base import BlackBoxModelBase

class WenxinyiyanModel(BlackBoxModelBase):
    r"""
    A class for interacting with Baidu's Wenxin Workshop API.

    This class allows users to generate text responses from Baidu's AI system
    by providing a simple interface to the Wenxin Workshop API. It manages authentication
    and request sending.
    """
    def __init__(self, API_KEY, SECRET_KEY):
        """
        Initializes the Wenxinyiyan instance with necessary credentials.
        :param str API_KEY: The API key for Baidu's service.
        :param str SECRET_KEY: The secret key for Baidu's service.
        """
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token="
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY

    @staticmethod
    def get_access_token(API_KEY, SECRET_KEY):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))

    def __call__(self, text_input):
        url = self.url + self.get_access_token(self.API_KEY, self.SECRET_KEY)
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": text_input
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        return response.json()['result']

#
    def generate(self, messages: 'List[str],str')->str:

        r"""
        Generate a response based on messages that include conversation history.

        :param ~List[str] messages: A list containing several messages.
                                    The user and assistant messages should appear in turns.
        :return: the response from the wenxinyiyan model based on a conversation history

        Example:
        messages = [
         "你好",
         "你好！有什么我可以帮助你的吗？请随时提出你的问题或需要帮助的内容，我会尽力提供准确和有用的答案。",
         "我想知道明天天气",]
        response = generate(messages)
        """
        # 判断message是str
        if isinstance(messages, str):
            messages = [messages]
        url = self.url + self.get_access_token(self.API_KEY, self.SECRET_KEY)
        processed_messages = []
        roles = ('user', 'assistant')
        for index, message in enumerate(messages):
            processed_messages.append({
                "role": roles[index % 2],
                "content": message
            })
        payload = json.dumps({
            "messages": processed_messages
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()['result']

    def batch_generate(self, conversations, **kwargs):
        responses = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn('If you want the model to generate batches based on several conversations, '
                              'please construct a list[str] for each conversation, or they will be divided into individual sentences. '
                              'Switch input type of batch_generate() to list[list[str]] to avoid this warning.')
            responses.append(self.generate(conversation))
        return responses