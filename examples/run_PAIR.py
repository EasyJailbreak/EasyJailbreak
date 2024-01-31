import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
import sys
sys.path.append(os.getcwd())
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

dataset_name_or_path = 'AdvBench'
attack_model_name = 'vicuna_v1.1'
attack_model_path = 'lmsys/vicuna-13b-v1.5'
target_model_name = 'llama-2'
target_model_path = 'meta-llama/Llama-2-7b-chat-hf'
eval_model_name = 'gpt-4'
openai_key = "your key"

target_model = from_pretrained(target_model_path, model_name=target_model_name)
attack_model = from_pretrained(attack_model_path, model_name=attack_model_name)
eval_model = OpenaiModel(model_name=eval_model_name, api_keys=openai_key)

jailbreak_dataset = JailbreakDataset(dataset_name_or_path)

attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=jailbreak_dataset,
                n_streams=20,
                n_iterations=5)

dataset_name = dataset_name_or_path.split('/')[-1]
eval_result = attacker.attack(save_path="{}_result.jsonl".format(dataset_name))
