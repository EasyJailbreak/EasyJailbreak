import os
import sys
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker import GPTFuzzer
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
sys.path.append(os.getcwd())

# 评估模型的加载
model_path = 'hubert233/GPTFuzz'
judge_model = RobertaForSequenceClassification.from_pretrained(model_path)
judge_tokenizer = RobertaTokenizer.from_pretrained(model_path)
judge_eval_model = HuggingfaceModel(model=judge_model, tokenizer=judge_tokenizer, model_name='zero-shot')

# 目标模型的加载
gpt4_model_name = 'gpt-4-1106-preview'
api_key = 'your key'
GPT4 = OpenaiModel(model_name=gpt4_model_name, api_keys=api_key)

# 数据集的加载和配置
dataset_name = 'QuestionList'
dataset = JailbreakDataset(dataset_name)

# 攻击器的初始化和配置
attacker = GPTFuzzer(attack_model=GPT4,
                     target_model=GPT4,
                     eval_model=judge_eval_model,
                     jailbreak_datasets=dataset,
                     max_iteration=100,
                     max_query=  10000,
                     max_jailbreak= 1000,
                     max_reject= 10000)

# 执行攻击并记录日志
attacker.attack()
attacker.log()
attacker.jailbreakDatasets.save_to_jsonl('QuestionList_gptfuzz.jsonl')
