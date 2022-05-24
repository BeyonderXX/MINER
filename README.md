

<h3 align="center">MINER: Mutual Information based Named Entity Recognition</h3>
<p align="center">
This repository contains the code for our paper [MINER: Improving Out-of-Vocabulary Named Entity Recognition from an Information Theoretic Perspective](https://aclanthology.org/2022.acl-long.383.pdf) (ACL 2022).



## Table of Contents

- [Overview](#Overview)
- [Installation](#Installation)
- [Preparation](#Preparation)
- [How to Run](#How to Run?)
- [Results](#Results at a Glance)
- [Citation](#Citation)

## Overview

NER models underperform when predicting entities that have not been seen during training, which is referred to as an Out-of-Vocabulary (OOV) problem. We propose MINER, a novel NER learning framework, to remedy this is- sue from an information-theoretic perspective.

![avatar](./pic/MINER_architecture.pdf)


## Installation

Require **python version >= 3.7**, recommend install with `pip`.

```shell
pip install -r requirements.txt
```



## Preparation

### Data Preprocessing

You need to put your data (train.txt, dev.txt, test.txt, labels.txt) in a folder . The data format refers to the CoNLL2003 dataset.

To generate new samples, we need to run the preprocessing script.

`python ./scripts/cal_vocab.py --data_dir data_dir_path --tokenizer tokenizer_name`

The script would output 'pmi.json' to your *data_dir *, which contains important substring to prevent operations on entities when they are replaced.

`python ./scripts/collect_entity.py --data_dir data_dir_path`

The script would output 'entity.json' to your *data_dir *, which contains entities and their neighbors.



### Prepare Models

Bert, ALbert and Roberta can be applied directly as our encoders.



## How to Run?

Note that the beta and gama of different datasets and models are different, it is recommended to adjust slightly when applying other scenarios. 

Here we give the parameters of **Bert-base-uncased** model, the performance will be further boosted when you use Bert-large.

```
# CoNLL2003
python -u main_conll.py --epoch 30 --do_train --batch_size 64 --gama 0.001 --beta 0.01 --gpu_id 0 --lr 0.00001 --switch_ratio 0.5 --data_dir data_dir_path --output_dir out_path 

# TwitterNER
python -u main.py --epoch 50 --batch_size 64 --do_train --do_eval --do_predict --gama 0.0001 --beta 0.0001 --gpu_id 0 --lr 0.00001 --switch_ratio 0.5 --data_dir data_dir_path --output_dir out_path 

# BioNER
python -u main.py --epoch 50 --batch_size 64 --do_train --do_eval --do_predict --gama 0.0001 --beta 0.001 --gpu_id 0 --lr 0.00001 --switch_ratio 0.5 --data_dir data_dir_path --output_dir out_path 

# WNUT2017
python -u main.py --epoch 50 --batch_size 64 --do_train --do_eval --do_predict --gama 0.0001 --beta 0.0001 --gpu_id 0 --lr 0.00001 --switch_ratio 0.5 --data_dir data_dir_path --output_dir out_path 
```

## Results at a Glance
![avatar](./pic/results.pdf)


## Citation

If you are using MINER for your work, please kindly cite our paper:

```latex
@inproceedings{wang2022miner,
  title={MINER: Improving Out-of-Vocabulary Named Entity Recognition from an Information Theoretic Perspective},
  author={Wang, Xiao and Dou, Shihan and Xiong, Limao and Zou, Yicheng and Zhang, Qi and others},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5590--5600},
  year={2022}
}
```