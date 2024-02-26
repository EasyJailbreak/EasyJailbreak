import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
sys.path.append(os.getcwd())
from easyjailbreak.attacker.AutoDAN_Liu_2023 import *
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak import models

# 加载model和tokenizer，作为target模型
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizers = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
llama2 = models.HuggingfaceModel(model,tokenizers,'llama-2')

your_key = ""
# 加载attack model
openai_model = models.OpenaiModel(model_name='gpt-3.5-turbo', api_keys=your_key, template_name='chatgpt')

# 加载数据
dataset = JailbreakDataset('AdvBench')

# attacker初始化
attacker = AutoDAN(
    attack_model=openai_model,
    target_model=llama2,
    jailbreak_datasets=dataset,
    eval_model = None,
    max_query=100,
    max_jailbreak=100,
    max_reject=100,
    max_iteration=100,
    device='cuda:0' if torch.cuda.is_available() else 'cpu' ,
    num_steps=100,
    sentence_level_steps=5,
    word_dict_size=30,
    batch_size=64,
    num_elites=0.1,
    crossover_rate=0.5,
    mutation_rate=0.01,
    num_points=5,
    model_name="llama2",
    low_memory=1,
    pattern_dict=None
)

attacker.attack()

attacker.attack_results.save_to_jsonl("AdvBench_AutoDAN.jsonl")
