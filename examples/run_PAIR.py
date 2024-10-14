import os
import sys
sys.path.append(os.getcwd())
from easyjailbreak.attacker.AutoDAN_Liu_2023 import AutoDAN
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
BASE_URL = "http://localhost:2213/v1"

# example target model 
target_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct", api_keys="EMPTY", base_url=BASE_URL)

# 加载attack model
attack_model = OpenaiModel(model_name="deepseek-chat", api_keys="", base_url="https://api.deepseek.com/")

eval_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct", api_keys="EMPTY", base_url=BASE_URL)

dataset = JailbreakDataset('HarmBench')

# Then instantiate the recipe.
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)

# Finally, start jailbreaking.
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
