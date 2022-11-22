import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EMBED_DIM=300                   # word vec的维度
HIDDEN_DIM=256                  # lstm隐藏层的向量维度
LAYERS=2                        # lstm层数
DROPOUT=0.5                     
TAG_VEC_DIM=32                      # 输出空间的维度

class VecNet(nn.Module):
    def __init__(self, word_num, word_embeddings=None):
        super(VecNet, self).__init__()
        
        if word_embeddings==None:
            self.embedding = nn.Embedding(word_num, EMBED_DIM) # freeze=False
        else:
            self.embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, bidirectional=True, batch_first=True, dropout=DROPOUT)
        self.full_connect = nn.Linear(HIDDEN_DIM * 2, TAG_VEC_DIM)

    def forward(self, x, attach_embedding=None, seq_lengths=None):

        if seq_lengths==None:
            # x是真实句子转换出的word_id
            # x = [word_vec1, word_vec2, ...]
            # 
            # 在训练新样本时为了防止对旧tag的遗忘，要用到gen_net生成的embeddings，这里由attach_embedding传入
            # 因为attach_embedding不是word_id，所以它进来的时候要跳过embedding层
        
            embedded = self.embedding(x)

            embedded_orig=embedded

            if attach_embedding!=None:
                embedded=torch.cat([attach_embedding,embedded])

            lstm_outs, (h_t, h_c) = self.lstm(embedded)
            out = lstm_outs[:, -1, :]# 句子最后时刻的输出，作为句子的vec
            out = self.full_connect(out)
            
            return out, embedded_orig
        
        else:
            # 只有在大批量训练的时候才会用到
            # x和seq_lengths是text_to_id_sequence_with_padding出来的seq_tensor,seq_lengths
            embedded_seq_tensors = self.embedding(x)
            packed_input = pack_padded_sequence(embedded_seq_tensors, seq_lengths.cpu().numpy(), batch_first=True)
            # out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
            
            packed_output, (ht, ct) = self.lstm(packed_input)
            
            # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
            out = torch.cat((ht[2], ht[3]), -1)
            out = self.full_connect(out)

            return out
        

class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TAG_VEC_DIM -> 256
        self.full_connect = nn.Linear(TAG_VEC_DIM, HIDDEN_DIM)
        # 256 -> 256
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, LAYERS, batch_first=True, dropout=DROPOUT)
        # 256 -> 300
        self.full_connect2 = nn.Linear(HIDDEN_DIM, EMBED_DIM)
    
    def forward(self, tag_vecs, seq_lengths):
        
        seq_out=[]
        for vec,times in zip(tag_vecs, seq_lengths):
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