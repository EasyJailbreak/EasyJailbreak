from openai import OpenAI
from httpx import URL
from typing import Union, List
from .model_base import BlackBoxModelBase
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
调用Kimi Api moonshot-v1-8k
"""

class KimiModel(BlackBoxModelBase): 
    def __init__(self, api_key: str = "",):
            """
            Initialize Kimi model handler
            :param model_name: Model identifier, defaults to "moonshot-v1-8k"
            :param api_key: Moonshot API authentication key
            :param generation_config: Generation parameters configuration
            :param base_url: API endpoint URL, defaults to official endpoint
            """
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1"
            )
            
    def generate(self, prompt:str, temperature=0.01, top_p=0.9,max_tokens=512, repetition_penalty=1.0,n=1):
        """
        Generate response for a conversation
        :param messages: Message list or single message string
        :param clear_old_history: Clear previous conversation context
        :return: Generated response content
        """
        completion = self.client.chat.completions.create(
                model = "moonshot-v1-8k",
                messages = [
                    {"role": "user", "content": prompt}
                ],
                temperature = temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=repetition_penalty,    
            )
        return [completion.choices[0].message.content]

    def batch_generate(self, prompts: List[Union[str, dict]], temperature=0.9, top_p=0.9, max_tokens=8192, max_workers=5):
        results = []

        def _chat_wrapper(prompt):
            return self.chat(prompt, temperature, top_p,  max_tokens)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(_chat_wrapper, prompt): prompt for prompt in prompts}

            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    print(f'Prompt {prompt} generated an exception: {exc}')
                    results.append(None)  

        return results