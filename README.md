# AutoTag

Continual Learning Model for Multi-class Text Classification based on Replay Method

## Features

- A general Data-IL classification continual learning framework with built-in Replay Method (Storing Feature Vecs \ Generating Pseudo Feature Vecs)
- Learn new samples while domain shifting (Domain Incremental Learning)
- Dynamically add new categories (Class Incremental Learning)
- A simple GUI

## Usage

run `main.py` to play with continual learning text classification (the embeddings below are needed)

- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

or run `experiment.ipynb` to see more general usage

## Thought

- calc_target_offset有很大的帮助，一个好的FXM+单纯的clustering就能解决Data-IL的任务……
- Generator很难训练好：因为Generator是单样本训练，所以在没有新样本时，将不会再有真实的feature_vec进行复习。若存储多个真实的feature_vec来陪练Generator，那还不如直接用存储的feature_vec对Classifier进行陪练
- 所以干脆存储具有代表性的feature_vec
- 样本数大一些，无方法的持续学习性能就不好了
- 维度低的情况就根本不用弄FXM，弄了反而增加学习难度，效果比None都差。
- SHUFFLE = False（一类一类学，且样本分布漂移），才能体现有优化方法与无优化方法的差别
