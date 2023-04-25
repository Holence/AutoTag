# AutoTag

Continual Learning Model for Multi-class Text Classification base on Replay Method

## Features

- use GUI to train and predict
- train new texts with shifted domain (Domain Incremental Learning)
- dynamically add new text categories (Class Incremental Learning)

## Embeddings

- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

## Awareness

- calc_target_offset有很大的帮助
- Generator很难训练好，甚至比没有任何优化的持续学习都差（把test.ipynb的实验结果放到论文里）：因为Generator是单样本训练，所以在没有新样本时，将不会再有真实的feature_vec进行复习。若存储多个真实的feature_vec来陪练Generator，那还不如直接用存储的feature_vec对Classifier进行陪练
- 所以干脆存储具有代表性的feature_vec

## ToDo

- self.lm.eval()要加吗
