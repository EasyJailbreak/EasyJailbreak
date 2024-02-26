import os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.ReNeLLM_ding_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
sys.path.append(os.getcwd())

generation_config = {'max_new_tokens':100}
llama_model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'Llama-2'
model =  AutoModelForCausalLM.from_pretrained(llama_model_path)
tokenizers = AutoTokenizer.from_pretrained(llama_model_path)
llama2_7b_chat = HuggingfaceModel(model = model,tokenizer=tokenizers, model_name= model_name,generation_config=generation_config)

chat_name = 'gpt-4-1106-preview'
openai_key = 'your key'
GPT4 = OpenaiModel(model_name= chat_name,api_keys=openai_key)

dataset_name = 'AdvBench'
dataset = JailbreakDataset(dataset_name)

attacker = ReNeLLM(attack_model=GPT4,
                   target_model=llama2_7b_chat,
                   eval_model=GPT4,
                   jailbreak_datasets=dataset)
attacker.attack()
attacker.jailbreakDatasets.save_to_jsonl('AdvBench_renellm.jsonl')