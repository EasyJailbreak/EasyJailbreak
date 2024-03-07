import os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.attacker.CodeChameleon_2024 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel


sys.path.append(os.getcwd())

generation_config = {'max_new_tokens':1000, 'do_sample':True, 'temperature':1,'top_p':0.9, 'use_cache':True, 'repetition_penalty':1}
model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'llama-2'
# "vicuna_v1.1" 'llama-2' 'mistral' 'internlm-chat' 'qwen-7b-chat' 'chatglm3'
model_size = '7B'
 
model =  AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map="auto")
tokenizers = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
target_model_now = HuggingfaceModel(model = model,tokenizer=tokenizers, model_name= model_name,generation_config=generation_config)

chat_name_eval = 'gpt-4-1106-preview'
api_key = 'your key'
GPT4 = OpenaiModel(model_name= chat_name_eval,api_keys=api_key)


dataset_name = 'AdvBench'
num_attack = 1
dataset = JailbreakDataset(dataset_name)
dataset._dataset = dataset._dataset[:num_attack]

attacker = CodeChameleon(attack_model=None,
                  target_model=target_model_now,
                  eval_model=GPT4,
                  jailbreak_datasets=dataset)

attacker.attack()
attacker.log()
file_name = "{}_{}_codechameleon.jsonl".format(model_name,model_size)
file_path = os.path.join("results", file_name)
attacker.attack_results.save_to_jsonl(file_path)
