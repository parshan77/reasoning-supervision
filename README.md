# Outcome-Process Combined Supervision of Multi-Hop Knowledge-Augmented Generation
This repository contains the codebase and the report for UCLA's CS269: Foundation Models final project. 

Team Members: Parshan Teimouri, Chengdi Cao, Tang Mohan

## Citation

```bibtex
@inproceedings{yao2023react,
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle = {International Conference on Learning Representations (ICLR) },
  year = {2023},
  html = {https://arxiv.org/abs/2210.03629},
}
```

## Setup
You need to first have an OpenAI API key and store it in the environment variable ``OPENAI_API_KEY`` (see [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)).

Package requirement: ``openai``, and install ``alfworld`` following instructions [here](https://github.com/alfworld/alfworld).

```shell
# run this for one time
conda create -n cs259 python=3.10
conda activate cs259
pip install -r requirements.txt

# run this every time before experiments
# put export OPENAI_API_KEY='...' in env.sh
conda activate cs259
source env.sh
```

## fine-tune on HotPotQA

```shell
python prepare_finetune_chat.py
python upload_file.py
# paste the uploaded filaname before proceeding
python finetune.py
```

## results on HotPotQA

`devinci-002`:
- dev: 1/100

`gpt-3.5-turbo-instruct`:
- train: 37/100
- dev: 4/10

`gpt-3.5-turbo-0125`:
- train: 40/100
- dev: 33/100

finetuned `gpt-3.5-turbo-0125`:
- train: 46/100
- dev: 35/100, 32/100 (w/o and w/ few-shot examples)
