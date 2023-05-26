from utils import *
import torch
import jieba

EMBED_DIM=300                   # word vec的维度
HIDDEN_DIM=300                  # lstm隐藏层的向量维度
LAYERS=2                        # lstm层数
DROPOUT=0.5                     
TAG_VEC_DIM=300                      # 输出空间的维度

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(HIDDEN_DIM * 2, TAG_VEC_DIM)
        if os.path.exists("LSTM_Para/classifier.pt"):
            self.load_state_dict(torch.load("LSTM_Para/classifier.pt"))
        else:
            init_network(self)

    def forward(self, x):
        out = self.fc(x)
        return out
        
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = torch.nn.Linear(TAG_VEC_DIM, HIDDEN_DIM * 2)
        if os.path.exists("./LSTM_Para/generator.pt"):
            self.load_state_dict(torch.load("./LSTM_Para/generator.pt"))
        else:
            init_network(self)
    
    def forward(self, x):
        out = self.fc(x)
        return out

class LSTM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 导入或生成token_id_dict
        # {
        #     "<PAD>": 0,
        #     "啊": 1,
        #     "<UNK>": ...
        # }
        self.token_id_dict=load_WordIdDict()
        self.lstm = torch.nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, bidirectional=True, batch_first=True, dropout=DROPOUT)
        
        if not os.path.exists("./LSTM_Para/lstm.pt"):

            # 生成word_embeddings，numpy array
            # [
            #     [1x300 random],
            #     [1x300 from sgns.zhihu.char],
            #     [1x300 random],
            # ]
            word_embeddings=load_WordEmbeddings(self.token_id_dict)
            self.embedding = torch.nn.Embedding.from_pretrained(word_embeddings, freeze=True)
            init_network(self.lstm)

            # 保存初始状态
            self.save()
        else:
            self.embedding = torch.nn.Embedding(len(self.token_id_dict), EMBED_DIM)
            self.load_state_dict(torch.load("./LSTM_Para/lstm.pt"))
        
    def save(self):
        torch.save(self.state_dict(), "./LSTM_Para/lstm.pt")

    def text_to_id_sequence(self,text):
        # "转换成id sequence"
        
        sequence=[]
        
        for i in jieba.cut(text,cut_all=False):
            if i.strip():
                id=self.token_id_dict.get(i, self.token_id_dict.get("<UNK>"))
                sequence.append(id)
        
        token_id_seq=torch.LongTensor(sequence).reshape(1,-1).to(self.device)

        return token_id_seq
    
    def forward(self, text):
        text=pre_process(text)
        token_id_seq=self.text_to_id_sequence(text)
        # x是真实句子转换出的token_id
        # x = [token_id1, token_id2, ...]
        # 
        # 在训练新样本时为了防止对旧tag的遗忘，要用到gen_net生成的embeddings，这里由attach_embedding传入
        # 因为attach_embedding不是token_id，所以它进来的时候要跳过embedding层

        # pytorch传入Embedding层不用one-hot的token表示法，直接用id的序列就行了
        embedded = self.embedding(token_id_seq)
        
        # 网上流传的这种取法有问题
        # output, (h_n, c_n) = self.lstm(embedded)
        # output = output[:, -1, :]
        # return output
        
        # https://github.com/yunjey/pytorch-tutorial/issues/149
        # 可以用下面的代码看一下
        # output[-1]包含的是最后一个token处正向和负向的输出
        # 而output[0]包含的是第一个token处正向和负向的输出
        # 应该取最后一个token处的正向输出+第一个token处的负向输出
        # 
        # h_n的组成是正确的，它由layer_num个最后token的输出组成
        # 如果是bidirectional，就是layer_num*2个，并且每两个都是最后一个token处正向和第一个token处的负向
        # lstm = torch.nn.LSTM(10, 7, 2, batch_first=True, bidirectional=True)
        # input=torch.randn([6, 10])
        # output,(h,c)=lstm(input)
        # print(output[-1])
        # print(h)
        
        # 所以这里取h_n的最后两个再concatenate一下就行了
        # 并且通过实验发现，按照下面这样修改后，精度大幅提升
        _, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2], h_n[-1]), -1)
        output = output.reshape(-1)
        return output
