import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EMBED_DIM=300                   # word vec的维度
HIDDEN_DIM=128                  # lstm隐藏层的向量维度
LAYERS=2                        # lstm层数
DROPOUT=0.5                     
OUT_DIM=10                      # 输出空间的维度

class VecNet(nn.Module):
    def __init__(self, word_num, word_embeddings=None):
        super(VecNet, self).__init__()
        
        if word_embeddings==None:
            self.embedding = nn.Embedding(word_num, EMBED_DIM) # freeze=False
        else:
            self.embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, bidirectional=True, batch_first=True, dropout=DROPOUT)
        self.full_connect = nn.Linear(HIDDEN_DIM * 2, OUT_DIM)

    def forward(self, x, attach_embedding=None):
        # x = [word_vec1, word_vec2, ...], [len(seq1), len(seq2), ...]
        #     shape: (batch_size, emb_dim)=(128, 300)     shape: (batch_size)=300
        
        out = self.embedding(x)
        # shape: (batch_size, pad_size, emb_dim)=(128, 32, 300)

        embedded=out

        if attach_embedding!=None:
            out=torch.cat([attach_embedding,out])

        lstm_outs, (h_t, h_c) = self.lstm(out)
        # lstm_outs 所有token经过lstm后的输出
        #   shape: (batch_size, pad_size, 2（双向）*hidden_size)=(128, 32, 2*128)
        #   128句为一个batch，每句话有32个token，每个token对应的输出为一个长度为2（双向）*128的vec
        #
        # ht 最后一个token处理完之后，所有hidden layer的状态
        #   shape: (2（双向）*num_layers, batch_size, hidden_size)=(2*2, 128, 128)
        #   双向，前向后向各2层，一共4层，每一层的输出为一个长度为*128的vec
        #
        # hc 最后一个token处理完之后，所有cell的状态
        
        out=lstm_outs[:, -1, :]# 句子最后时刻的输出，作为句子的vec
        
        # 注：ht中包含了lstm_outs中句子最后时刻的输出
        
        out = self.full_connect(out)  
        
        return out, embedded
    
    # def forward(self, seq_tensors, seq_lengths):
        
    #     embedded_seq_tensors = self.embedding(seq_tensors)
    #     packed_input = pack_padded_sequence(embedded_seq_tensors, seq_lengths.cpu().numpy(), batch_first=True)
    #     # out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
        
    #     packed_output, (ht, ct) = self.lstm(packed_input)
    #     # out, (hn, _) = self.lstm(out)
        
    #     # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
    #     out = torch.cat((ht[2], ht[3]), -1)
    #     out = self.full_connect(out)

    #     return out

class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 10 -> 128
        self.full_connect = nn.Linear(OUT_DIM, HIDDEN_DIM)
        # 128 -> 128
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, LAYERS, batch_first=True, dropout=DROPOUT)
        # 128 -> 300
        self.full_connect2 = nn.Linear(HIDDEN_DIM, EMBED_DIM)
    
    def forward(self, seq_tensors, seq_lengths):
        
        seq_out=[]
        for vec,times in zip(seq_tensors,seq_lengths):
            result=[]

            out=self.full_connect(vec).reshape(1,1,-1)
            
            lstm_outs, (ht, ct)=self.lstm(out)
            lstm_outs=lstm_outs[:, -1, :]
            result.append(lstm_outs[0])
            lstm_outs=lstm_outs.reshape(1,1,-1)
            
            for i in range(times-1):
                lstm_outs, (ht, ct)=self.lstm(lstm_outs)
                lstm_outs=lstm_outs[:, -1, :]
                result.append(lstm_outs[0])
                lstm_outs=lstm_outs.reshape(1,1,-1)

            seq_out.append(torch.stack(result))

        seq_out=torch.stack(seq_out)
        
        seq_out=self.full_connect2(seq_out)
        
        return seq_out