import os
import sys
from easyjailbreak.attacker.MJP_Li_2023 import MJP
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.datasets.instance import Instance
from easyjailbreak.models.openai_model import OpenaiModel
import pandas as pd

sys.path.append(os.getcwd())
# Initialization
openai_key = "your key"
target_model = OpenaiModel('gpt-3.5-turbo-0301',openai_key)
eval_model = OpenaiModel('gpt-3.5-turbo-0301',openai_key)

prompt_type='JQ+COT+MC'
# Data load
dataset = JailbreakDataset('MJP')

attacker = MJP(target_model, eval_model, jailbreak_datasets=dataset, prompt_type=prompt_type, batch_num=5)

attacker.attack()
attacker.jailbreak_Dataset.save_to_jsonl('MJP_results.jsonl')
