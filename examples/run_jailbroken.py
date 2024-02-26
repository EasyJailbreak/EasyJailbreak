import os, sys
from easyjailbreak.datasets import JailbreakDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.attacker.Jailbroken_wei_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
sys.path.append(os.getcwd())

generation_config = {'max_new_tokens':100}
llama_model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'meta-llama/Llama-2-7b-chat-hf'
model =  AutoModelForCausalLM.from_pretrained(llama_model_path)
tokenizers = AutoTokenizer.from_pretrained(llama_model_path)
llama2_7b_chat = HuggingfaceModel(model = model,tokenizer=tokenizers, model_name= model_name,generation_config=generation_config)

chat_name = 'GPT-4'
api_key = 'your key'
GPT4 = OpenaiModel(model_name= chat_name,api_keys=api_key)


dataset_name = 'AdvBench'
num_attack = 1
dataset = JailbreakDataset(dataset_name)
dataset._dataset = dataset._dataset[:num_attack]

attacker = Jailbroken(attack_model=GPT4,
                   target_model=GPT4,
                   eval_model=GPT4,
                   jailbreak_datasets=dataset)
attacker.attack()
attacker.log()
attacker.attack_results.save_to_jsonl('AdvBench_jailbroken.jsonl')
