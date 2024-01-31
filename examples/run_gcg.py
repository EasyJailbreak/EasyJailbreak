"""
# GCG use case example.

# Using the GCG algorithm to optimize the prompt for the vicuna model to output the desired content.
"""
import easyjailbreak
import torch
import logging
from transformers import set_seed
from easyjailbreak.datasets import JailbreakDataset

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
set_seed(42)

# Load model and tokenizer
model = easyjailbreak.models.from_pretrained('lmsys/vicuna-7b-v1.5', 'vicuna_v1.1', dtype=torch.bfloat16, max_new_tokens=40)

# Load dataset
dataset_name = 'AdvBench'
dataset = JailbreakDataset(dataset_name)

# API class initialization and settings
attacker = easyjailbreak.attacker.GCG(
    attack_model=model,
    target_model=model,
    jailbreak_datasets=dataset,
    jailbreak_prompt_length=20,
    num_turb_sample=512,
    batchsize=256,      # decrease if OOM
    top_k=256,
    max_num_iter=500
)

# Launch the attack
attacker.attack()

attacker.jailbreak_datasets.save_to_jsonl(f'gcg_result.jsonl')