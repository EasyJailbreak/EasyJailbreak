import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.attacker.TAP_Mehrotra_2023 import TAP
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

sys.path.append(os.getcwd())
dataset = JailbreakDataset('AdvBench')[:100]
dataset_name_or_path = '/home/zhangshuo/xlm_workspace/easyjailbreak/dataset/AdvBench4.jsonl'
attack_model_name = 'vicuna'
attack_model_path = '/home/zhangshuo/models/vicuna-13b-v1.5'
target_model_name = 'llama'
target_model_path = '/home/zhangshuo/models/Llama-2-7b-chat-hf'
eval_model_name = 'gpt-4'
openai_key = "your key"

target_model = from_pretrained(target_model_path, model_name=target_model_name)
attack_model = from_pretrained(attack_model_path, model_name=attack_model_name)
eval_model = OpenaiModel(model_name=eval_model_name, api_keys=openai_key)

attacker = TAP(attack_model=attack_model,
               target_model=target_model,
               eval_model=eval_model,
               jailbreak_datasets=dataset,
               tree_width=10,
               tree_depth=10,
               root_num=1,
               branching_factor=4,
               keep_last_n=3,
               max_n_attack_attempts=5,
               template_file='/home/zhangshuo/zfk_workspace/EasyJailbreak/easyjailbreak/seed/seed_template.json')
dataset_name = dataset_name_or_path.split('/')[-1]
eval_result = attacker.attack(save_path="{}_result.jsonl".format(dataset_name))
