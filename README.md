# AutoTag

Continual Learning Model for Multi-class Text Classification base on Replay Method

## Features

- use GUI to train and predict
- train new texts with shifted domain (Domain Incremental Learning)
- dynamically add new text categories (Class Incremental Learning)

## Embeddings

- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

## ToDo

- lstm的gen_net如何持续训练？
- 现在持续学习的效果比forward_without_generator还差🤡，因为Generator是单样本训练
- self.lm.eval()要加吗
