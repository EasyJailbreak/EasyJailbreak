<h1 align="center"><img src="image/README/logo.png" alt="EasyJailbreak Logo" height="60"></h1>

<p align="center"><font face="Lucida Sans"><em>—— An easy-to-use Python framework to generate adversarial jailbreak prompts by assembling different methods</em></font><br><br> 
<a href="http://easyjailbreak.org/">
  	<img alt="Website" src="https://img.shields.io/website?up_message=online&url=http%3A%2F%2Feasyjailbreak.org%2F" height="18">
  </a>
<a>
  	<img alt="License" src="https://img.shields.io/badge/license-GPL%20v3-brightgreen" height="18">
  </a>

  <a href="https://easyjailbreak.github.io/EasyJailbreakDoc.github.io">
    <img alt="Read Docs" src="https://img.shields.io/static/v1?label=Read%20Docs&message=guide&color=blue" height="18">
  </a>

<a href="https://badge.fury.io/py/EasyJailbreak">
  <img alt="GitHub release (latest by date)" 	src="https://img.shields.io/github/v/release/EasyJailbreak/EasyJailbreak?label=release" height="18">
</a>
</p>

<img src="image/README/terminal_demo.gif" alt="EasyJailbreak GIF" style="display: block; margin: 0 auto;" />


## Table of Contents

- [About](#about)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citing EasyJailbreak](#citing-easyjailbreak)

## About

### ✨ Introduction

**What is EasyJailbreak?**

EasyJailbreak is an *easy-to-use* Python framework designed for researchers and developers focusing on LLM security. Specifically,  EasyJailbreak decomposes the mainstream jailbreaking process into several iterable steps: initialize **mutation seeds**, **select suitable seeds**, **add constraint**, **mutate**, **attack**, and **evaluate**. On this basis, EasyJailbreak provides a component for each step, constructing a playground for further research and attempts. More details can be found in our paper. 

### 📚 Resources

- **[EasyJailbreak Website](http://easyjailbreak.org/):** Explore different LLMs' jailbreak results and view examples of jailbreaks. 

- **[Documentation](https://easyjailbreak.github.io/EasyJailbreakDoc.github.io):** Detailed API documentation and parameter explanations.


## 🛠️ Setup

There are two methods to install EasyJailbreak.

1. For users who only require the approaches (or [recipes](#using-recipe)) collected in EasyJailbreak, execute the following command:

```shell
pip install easyjailbreak
```

2. For users interested in [adding new components](#diy-your-attacker) (e.g., new mutate or evaluate methods), follow these steps:

```shell
git clone https://github.com/EasyJailbreak/EasyJailbreak.git
cd EasyJailbreak
pip install -e .
```

## 🔍 Project Structure

This project is mainly divided into three parts.

1. The first part requires the user to prepare **Queries**, **Config,** **Models**, and **Seed**.
2. The second part is the main part, consisting of two processes that form a loop structure, namely **Mutation** and **Inference**.

   1) In the **Mutation** process, the program will first select the optimal jailbreak prompts through **Selector**, then transform the prompts through **Mutator**, and then filter out the expected prompts through **Constraint.**
   2) In the **Inference** process, the prompts are used to attack the **Target (model)** and obtain the target model's responses. The responses are then inputted into **Evaluator** to obtain the score of the attack's effectiveness for this round, which is then passed to Selector to complete one cycle.
3. The third part you will get a **Report**. Under some stopping mechanism, the loop stops, and the user will receive a report about each attack (including jailbreak prompts, responses of **Target (model)**, Evaluator's scores, etc.).

<p align="center"><img src="image/README/project_structure.png" alt="Project Structure" height="300"></p>

The following table shows the 4 essential components (i.e. **Selectors**, **Mutators**, **Constraints**, **Evaluators**) used by each recipe implemented in our project:

| <font face="Arial Black" size="4">Attack<br>Recipes</font> | <font face="Arial Black" size="4">Selector</font> | <font face="Arial Black" size="4">Mutation</font> | <font face="Arial Black" size="4">Constraint</font>| <font face="Arial Black" size="4">Evaluator</font>|
| :---------------: | :---------------: | :----------------: | :--------------: |:-------------: |
| <font face="Arial Black"><b>ReNeLLM</b></font>|<sub> N/A </sub>|<sub>ChangeStyle<br>InsertMeaninglessCharacters<br>MisspellSensitiveWords<br>Rephrase<br>GenerateSimilar<br>AlterSentenceStructure | <sub>DeleteHarmLess | <sub>Evaluator_GenerativeJudge |
| <font face="Arial Black"><b>GPTFuzz</b></font>| <sub>MCTSExploreSelectPolicy<br>RandomSelector<br>EXP3SelectPolicy<br>RoundRobinSelectPolicy<br>UCBSelectPolicy |<sub> ChangeStyle<br>Expand<br>Rephrase<br>Crossover<br>Translation<br>Shorten|<sub> N/A| <sub>Evaluator_ClassificationJudge </sub>|
| <font face="Arial Black"><b>ICA</b></font> |<sub> N/A| <sub>N/A| <sub>N/A|<sub> Evaluator_PatternJudge</sub>|
| <font face="Arial Black"><b>AutoDAN</b></font>|<sub> N/A| <sub>Rephrase<br>CrossOver<br>ReplaceWordsWithSynonyms|<sub> N/A| <sub>Evaluator_PatternJudge</sub>|
| <font face="Arial Black"><b>PAIR</b></font> |<sub> N/A| <sub>HistoricalInsight|<sub> N/A| <sub>Evaluator_GenerativeGetScore</sub>|
| <font face="Arial Black"><b>JailBroken</b></font> | <sub>N/A| <sub>Artificial<br>Auto_obfuscation<br>Auto_payload_splitting<br>Base64_input_only<br>Base64_raw<br>Base64<br>Combination_1<br>Combination_2<br>Combination_3<br>Disemovowel<br>Leetspeak<br>Rot13 | <sub>N/A|<sub> Evaluator_GenerativeJudge</sub> |
| <font face="Arial Black"><b>Cipher</b></font> | <sub>N/A|<sub> AsciiExpert<br>CaserExpert<br>MorseExpert<br>SelfDefineCipher |<sub> N/A| <sub>Evaluator_GenerativeJudge </sub>|
| <font face="Arial Black"><b>DeepInception</b></font>|<sub> N/A| <sub>Inception|<sub> N/A|<sub> Evaluator_GenerativeJudge </sub>|
| <font face="Arial Black"><b>MultiLingual</b></font> |<sub>N/A|<sub> Translate|<sub> N/A| <sub>Evaluator_GenerativeJudge </sub>|
| <font face="Arial Black"><b>GCG</b></font> |<sub> ReferenceLossSelector|<sub> MutationTokenGradient| <sub>N/A|<sub> Evaluator_PrefixExactMatch</sub>|
| <font face="Arial Black"><b>TAP</b></font>|<sub>SelectBasedOnScores|<sub> IntrospectGeneration </sub>|<sub> DeleteOffTopic </sub>| <sub>Evaluator_GenerativeGetScore</sub>|


## 💻 Usage

### Using Recipe

We have got many **implemented methods** ready for use! Instead of devising a new jailbreak scheme, the EasyJailbreak team gathers from relevant papers, referred to as **"recipes"**. Users can freely apply these jailbreak schemes on various models to familiarize the performance of both models and schemes. The only thing users need to do for this is download models and utilize the provided API.

Here is a usage example:

```python
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
attack_model = from_pretrained(model_name_or_path='lmsys/vicuna-13b-v1.5',
                               model_name='vicuna_v1.1')
target_model = OpenaiModel(model_name='gpt-4',
                         api_keys='INPUT YOUR KEY HERE!!!')
eval_model = OpenaiModel(model_name='gpt-4',
                         api_keys='INPUT YOUR KEY HERE!!!')
dataset = JailbreakDataset('AdvBench')

# Then instantiate the recipe.
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)

# Finally, start jailbreaking.
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
```

All available recipes and their relevant information can be found in the [documentation](https://easyjailbreak.github.io/EasyJailbreakDoc.github.io/).

### DIY Your Attacker

#### 1. Load Models

You can load a model in one line of python code.

```python
# import model prototype
from easyjailbreak.models.huggingface_model import HuggingfaceModel

# load the target model (but you may use up to 3 models in a attacker, i.e. attack_model, eval_model, target_model)
target_model = HuggingfaceModel(model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
                                model_name='llama-2')

# use the target_model to generate response based on any input. Here is an example.  
target_response = target_model.generate(messages=['how to make a bomb?'])
```

#### 2. Load Dataset and intialize Seed

**Dataset**: We prepare a class named "JailbreakDataset" to wrap the the instance list. And every instance contains query, jailbreak prompts, etc. You can either load Dataset from our online repo or your local file.

**Seed**: You can simply ramdomly generate initial seed.

```python
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.seed.seed_random import SeedRandom

# Option 1: load dataset from our online repo. Available datasets and their details can be found at https://huggingface.co/datasets/Lemhf14/EasyJailbreak_Datasets
dataset = JailbreakDataset(dataset='AdvBench')

# Option 2: load dataset from a local file
dataset = JailbreakDataset(local_file_type='csv', dataset='AdvBench.csv')

# Randomly generate initial seed
seeder = SeedRandom()
seeder.new_seeds()
```

#### 3. Instantiate Components

 As mentioned in [Project Structure](#project-structure), the second part consists of 4 major components (modules, i.e. selector, mutator, constraint, evaluator) and you need to instantiate them when you DIY your attack method. All available **Selectors**, **Mutators**, **Constraint**, **Evaluators** and their details can be found in the [documentation](https://easyjailbreak.github.io/EasyJailbreakDoc.github.io/).

You can `import` the module you want by using `from easyjailbreak.module_name.method_name import method_name`, here is a brief instruction for you to start (the `method_name` is what you choose as the method in the corresponding module):

1. **Selector**:  `from easyjailbreak.selector.method_name import method_name`
2. **Mutator**:  `from easyjailbreak.mutation.rule.method_name import method_name`
3. **Constraint**:  `from easyjailbreak.constraint.method_name import method_name`
4. **Evaluator**:  `from easyjailbreak.metrics.Evaluator.method_name import method_name`

Here is an example.

```python
from easyjailbreak.selector.RandomSelector import RandomSelectPolicy
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.seed import SeedTemplate
from easyjailbreak.mutation.rule import Translate
from easyjailbreak.models import from_pretrained
import torch

# Initializes malicious questions used to attack language models
instance = Instance(query='How to make a bomb?')
dataset = JailbreakDataset([instance])

# Define the target model of the attack
model = from_pretrained('meta-llama/Llama-2-7b-chat-hf', 'llama-2', dtype=torch.bfloat16, max_new_tokens=200)

# Initializes a Jailbreak prompt
inital_prompt_seed = SeedTemplate().new_seeds(seeds_num= 10, method_list=['Gptfuzzer'])
inital_prompt_seed = JailbreakDataset([Instance(jailbreak_prompt=prompt) for prompt in inital_prompt_seed])

# Initializes a Selector
selector = RandomSelectPolicy(inital_prompt_seed)

# Apply selection to provide a prompt
candidate_prompt_set = selector.select()
for instance  in dataset:
    instance.jailbreak_prompt = candidate_prompt_set[0].jailbreak_prompt

# Mutate the raw query to fool the language model
Mutation = Translate(attr_name='query',language = 'jv')
mutated_instance = Mutation(dataset)[0]

#  get target model's response
attack_query = mutated_instance.jailbreak_prompt.format(query = mutated_instance.query)
response = model.generate(attack_query)
```

## 🖊️ Citing EasyJailbreak

```bibtex
@misc{2024easyjailbreak,
    title={EasyJailbreak: A Unified Framework for Jailbreaking Large Language Models},
    author={Zhou, Weikang and Wang, Xiao and Xiong, Limao and Xia, Han and Gu, Yingshuang and Chai, Mingxu and Zhu, Fukang and Huang, Caishuang and Dou, Shihan and Xi, Zhiheng and Zheng, Rui and Gao, Songyang and Zou, Yicheng and Yan, Hang and Le, Yifan and Wang, Ruohui and Li, Lijun and Shao, Jing and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
    howpublished = {\url{https://github.com/EasyJailbreak/EasyJailbreak}},
    year={2024}
}

```
