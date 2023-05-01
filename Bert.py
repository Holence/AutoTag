import torch
from utils import *
from transformers import BertTokenizer, BertModel

EMBED_DIM=768
TAG_VEC_DIM=384

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(EMBED_DIM, TAG_VEC_DIM)   # 768 -> 384
        if os.path.exists("./Bert_Para/classifier.pt"):
            self.load_state_dict(torch.load("./Bert_Para/classifier.pt"))
        else:
            init_network(self)

    def forward(self, feature_vecs):
        out = self.fc(feature_vecs)
        return out

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = torch.nn.Linear(TAG_VEC_DIM, EMBED_DIM)   # 384 -> 768
        if os.path.exists("Bert_Para/generator.pt"):
            self.load_state_dict(torch.load("Bert_Para/generator.pt"))
        else:
            init_network(self)

    def forward(self, tag_vecs):
        out = self.fc(tag_vecs)
        return out

class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists("./Bert_Para/pytorch_model.bin"):
            self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.tokenizer.save_pretrained("./Bert_Para/")
            self.bert.save_pretrained("./Bert_Para/")
        else:
            self.tokenizer = BertTokenizer.from_pretrained("./Bert_Para/")
            self.bert = BertModel.from_pretrained("./Bert_Para/")
        
        # 固定Bert
        for p in self.bert.parameters():
            p.requires_grad=False

    def save(self):
        pass

    def forward(self, text):
        text=pre_process(text)
        inputs=self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs.to(self.device)
        output=self.bert(**inputs)
        return output["pooler_output"]
