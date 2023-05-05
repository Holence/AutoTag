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

- calc_target_offset有很大的帮助
- Generator很难训练好：因为Generator是单样本训练，所以在没有新样本时，将不会再有真实的feature_vec进行复习。若存储多个真实的feature_vec来陪练Generator，那还不如直接用存储的feature_vec对Classifier进行陪练
- 所以干脆存储具有代表性的feature_vec
- 学习率对Generator的影响很大
- 样本数大一些，结果比较好

```
INPUT_DIM=3
CLASS_NUM=15
SAMPLE_NUM=10
SAMPLE_NUM_RANGE=...
MEAN_ARRANGE=100
STD_ARRANGE=5
SAMPLE_SHIFTING=...
```

| 无优化方法的持续学习与批量学习的Accuracy比较 | 每类的样本数量均匀<br />`SAMPLE_NUM_RANGE=0` | 每类的样本数量不均匀<br />`SAMPLE_NUM_RANGE=0.5` |
| -------------------------------------------- | -------------------------------------------- | ------------------------------------------------ |
| 样本分布不漂移<br />`SAMPLE_SHIFTING=False`  | 基本持平                                     | 基本持平                                         |
| 样本分布漂移<br />`SAMPLE_SHIFTING=True`     | 些许差距                                     | 较大差距                                         |
