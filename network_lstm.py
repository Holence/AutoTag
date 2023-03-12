from utils import *
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import jieba

EMBED_DIM=300                   # word vec的维度
HIDDEN_DIM=768                  # lstm隐藏层的向量维度
LAYERS=2                        # lstm层数
DROPOUT=0.5                     
TAG_VEC_DIM=384                      # 输出空间的维度

TRAIN_ALONG=10

class VecNet(torch.nn.Module):
    def __init__(self, word_num, word_embeddings=None):
        super(VecNet, self).__init__()
        
        if word_embeddings==None:
            self.embedding = torch.nn.Embedding(word_num, EMBED_DIM)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(word_embeddings, freeze=True)
        
        self.lstm = torch.nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, bidirectional=True, batch_first=True, dropout=DROPOUT)
        self.full_connect = torch.nn.Linear(HIDDEN_DIM * 2, TAG_VEC_DIM)

    def forward(self, x, attach_embedding=None, seq_lengths=None):

        if seq_lengths==None:
            # x是真实句子转换出的token_id
            # x = [token_id1, token_id2, ...]
            # 
            # 在训练新样本时为了防止对旧tag的遗忘，要用到gen_net生成的embeddings，这里由attach_embedding传入
            # 因为attach_embedding不是token_id，所以它进来的时候要跳过embedding层

            # pytorch传入Embedding层不用one-hot的token表示法，直接用id的序列就行了
            embedded = self.embedding(x)

            embedded_orig=embedded

            if attach_embedding!=None:
                # current text的embedding放在最后
                embedded=torch.cat([attach_embedding,embedded])

            # 原始的版本
            # output, (h_t, h_c) = self.lstm(embedded)
            # output = output[:, -1, :]# 句子最后时刻的输出，作为句子的vec
            # output = self.full_connect(output)
            # return output, embedded_orig

            # torch的BiLSTM的output有问题
            # https://github.com/yunjey/pytorch-tutorial/issues/149
            # 可以用下面的代码看一下
            # lstm = torch.nn.LSTM(10, 7, 2, batch_first=True, bidirectional=True)
            # input=torch.randn([6, 10])
            # output,(h,c)=lstm(input)
            # print(output[-1])
            # print(h)
            # output[-1]包含的是最后一个token处正向和负向的输出，而output[0]包含的是第一个token处正向和负向的输出
            # 应该是最后一个token处的正向输出+第一个token处的负向输出

            
            # 实验发现，按照下面这样修改后，精度大幅提升
            _, (h_n, c_n) = self.lstm(embedded)
            # h_n的-2和-1的位置上是正向和负向的最后一层的输出
            output = torch.cat((h_n[-2], h_n[-1]), -1)
            output = self.full_connect(output)
            
            return output, embedded_orig

        else:
            # 只有在batch_train的时候才会用到
            # x和seq_lengths是text_to_id_sequence_with_padding出来的seq_tensor,seq_lengths
            embedded_seq_tensors = self.embedding(x)
            packed_input = pack_padded_sequence(embedded_seq_tensors, seq_lengths.numpy(), batch_first=True)
            
            _, (h_n, c_n) = self.lstm(packed_input)
            # h_n的-2和-1的位置上是正向和负向的最后一层的输出
            out = torch.cat((h_n[-2], h_n[-1]), -1)
            out = self.full_connect(out)

            return out
        
class GenNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TAG_VEC_DIM -> EMBED_DIM
        self.full_connect = torch.nn.Linear(TAG_VEC_DIM, EMBED_DIM)
        # EMBED_DIM -> EMBED_DIM
        self.lstm = torch.nn.LSTM(EMBED_DIM, EMBED_DIM, LAYERS, batch_first=True, dropout=DROPOUT)
    
    def forward(self, tag_vecs, seq_lengths):
        
        seq_out=[]
        for vec,times in zip(tag_vecs, seq_lengths):
            result=[]

            inputs=self.full_connect(vec).reshape(1,1,-1)

            lstm_outs, (ht, ct)=self.lstm(inputs)
            lstm_outs=lstm_outs[0,0]
            result.append(lstm_outs)
            inputs=lstm_outs.reshape(1,1,-1)

            for i in range(times-1):
                lstm_outs, (ht, ct)=self.lstm(inputs, (ht, ct))
                lstm_outs=lstm_outs[0,0]
                result.append(lstm_outs)
                inputs=lstm_outs.reshape(1,1,-1)

            seq_out.append(torch.stack(result))

        seq_out=torch.stack(seq_out)
        
        return seq_out

class Model():

    VEC_NET_LR=0.0005
    GEN_NET_LR=0.0005

    def __init__(self) -> None:

        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:",self.device)

        # 导入或生成token_id_dict
        # {
        #     "<PAD>": 0,
        #     "啊": 1,
        #     "<UNK>": ...
        # }
        self.token_id_dict=load_WordIdDict()

        # VecNet
        if not os.path.exists("lstm_para/vec_net_para_init.pt"):

            # 生成word_embeddings，numpy array
            # [
            #     [1x300 random],
            #     [1x300 from sgns.zhihu.char],
            #     [1x300 random],
            # ]
            word_embeddings=load_WordEmbeddings(self.token_id_dict)

            # 创建VecNet，并把word_embeddings存入VecNet的embedding层
            self.vec_net=VecNet(len(self.token_id_dict), word_embeddings)
            self.vec_net.to(self.device)

            # 初始化VecNet（除去embedding层，初始化LSTM、Linear Layer的参数）
            init_network(self.vec_net)

            for name,weight in self.vec_net.named_parameters():
                print(name,weight.shape)

            # 保存初始状态
            torch.save(self.vec_net.state_dict(), "lstm_para/vec_net_para_init.pt")
        else:
            
            self.vec_net=VecNet(len(self.token_id_dict))
            try:
                self.vec_net.load_state_dict(torch.load("lstm_para/vec_net_para.pt"))
            except:
                self.vec_net.load_state_dict(torch.load("lstm_para/vec_net_para_init.pt"))
            self.vec_net.to(self.device)
        
        # 学习率也应该保存并读取
        self.vec_net_optimizer = torch.optim.Adam(self.vec_net.parameters(), lr=self.VEC_NET_LR)
        self.vec_net_criterion = torch.nn.MSELoss()
        
        # GenNet
        self.gen_net=GenNet()
        try:
            self.gen_net.load_state_dict(torch.load("lstm_para/gen_net_para.pt"))
        except:
            init_network(self.gen_net)
        self.gen_net.to(self.device)
        
        # 学习率也应该保存并读取
        self.gen_net_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr=self.GEN_NET_LR)
        self.gen_net_criterion = torch.nn.MSELoss()
        
        if os.path.exists("lstm_para/tag_dict.pt"):
            self.tag_dict=torch.load("lstm_para/tag_dict.pt")
        else:
            self.tag_dict={}
        
        self.loadCorpus()

        self.loss_dict={
            "VecNet Batch Train Loss":[],
            "VecNet Continual Single Train Loss":[],
            "VecNet Continual Attach Train Loss":[],
            "VecNet Continual Baseline Train Loss":[],
            "GenNet Continual Single Train Loss":[],
            "GenNet Continual Attach Train Loss":[],
        }
        
        self.acc_dict={}
        for tag in self.train_dict:
            if self.acc_dict.get(tag)==None:
                self.acc_dict[tag+"_test"]=[]
                self.acc_dict[tag+"_train"]=[]
    
    def loadCorpus(self):
        self.train_pipe, self.train_dict, self.test_dict=load_corpus("corpus", 0.2)

    def save(self):
        torch.save(self.vec_net.state_dict(), "lstm_para/vec_net_para.pt")
        torch.save(self.gen_net.state_dict(), "lstm_para/gen_net_para.pt")
        torch.save(self.tag_dict, "lstm_para/tag_dict.pt")
    
    def text_to_id_sequence_with_padding(self,text_list):
        # "转换成token_id，padding，再排序"

        # 生成id序列，这里是形状如[[...], [...], ...]的python的列表
        id_seqs=[]
        for t in text_list:
            sequence=[]
            for i in jieba.cut(t,cut_all=False):
                if i.strip():
                    id=self.token_id_dict.get(i, self.token_id_dict.get("<UNK>"))
                    sequence.append(id)
            id_seqs.append(sequence)
        
        # 下面为了支持多个不同长度的输入，进行padding，以及加速运算的packing
        
        # 计算每句话的长度
        seq_lengths = torch.LongTensor(list(map(len, id_seqs)))
        
        # 生成shape为(len(id_seqs), 最大长度序列)的全0矩阵
        token_id_seq=torch.zeros((len(id_seqs), seq_lengths.max())).long().to(self.device)
        
        # 把id_seqs填进去
        for idx, (seq, seqlen) in enumerate(zip(id_seqs, seq_lengths)):
            token_id_seq[idx, :seqlen] = torch.LongTensor(seq)
        
        # 按照id_seqs中每句话的非零元素的多少排序（从多到少）
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        # seq_lengths的顺序更新了，perm_idx记录了sort的操作
        # 用perm_idx对id_seq进行重排序
        token_id_seq = token_id_seq[perm_idx]

        return token_id_seq, seq_lengths, perm_idx

    def text_to_id_sequence(self,text):
        # "转换成id sequence"
        
        sequence=[]
        
        for i in jieba.cut(text,cut_all=False):
            if i.strip():
                id=self.token_id_dict.get(i, self.token_id_dict.get("<UNK>"))
                sequence.append(id)
        
        token_id_seq=torch.LongTensor(sequence).reshape(1,-1).to(self.device)
        # token_id_seq=torch.autograd.Variable(torch.zeros((len(id_seqs), seq_lengths.max()))).long().to(self.device)

        return token_id_seq

    def forward(self, current_text, current_tag):
        # "单样本正向训练，生成向量进行陪练"

        def train_gen(embedded_sentence):
            seq_lengths = torch.LongTensor([len(token_id_seq[0])])

            self.gen_net.train()
            
            output_seq=self.gen_net(self.tag_dict[current_tag]["vec"].reshape(1,-1), seq_lengths)
            
            # loss为gen_net的output_seq和embedded_sentence的差异
            # 试图让gen_net输入tag domain的vec，拟合输出embedding domain的vec
            self.gen_net.zero_grad()
            
            loss=self.gen_net_criterion(output_seq, embedded_sentence)
            loss.backward()
            self.gen_net_optimizer.step()

            return loss.tolist()

        current_text=pre_process(current_text)
        token_id_seq=self.text_to_id_sequence(current_text)

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if type(self.tag_dict.get(current_tag))==type(None):
            # [token的个数]
            seq_lengths=[token_id_seq.shape[1]]
            
            # -------------------------------------
            
            self.vec_net.eval()
            with torch.no_grad():
                # 输入id sequence的id_seq(1xn)
                # 先用vec_net预测，输出tag domain的向量tag_vec(1 x TAG_VEC_DIM)，以及embed的vec(nx300)（用来之后训练gen_net
                tag_vec, embedded_sentence=self.vec_net(token_id_seq)

            # 把tag_vec存储到tag_dict中
            self.tag_dict[current_tag]={
                "vec":tag_vec.detach()[0,...], # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                "time":1
            }
            
            # -------------------------------------
            loss=train_gen(embedded_sentence.detach())
            self.loss_dict["GenNet Continual Single Train Loss"].append(loss)

        else:
            
            # -------------------------------------
            # 先用gen_net预测生成current_tag之外的其他tag的所对应的embedding作为训练vec_net的陪练
            # 以应对Catastrophic Forgetting
            

            # 堆叠其他tag的vec
            other_tag_vecs=[]
            for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict),TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
                if tag!=current_tag:
                    other_tag_vecs.append(value["vec"])
            
            # 如果没有其他陪练的，就取vec_net的输出为TARGET_VEC，取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
            if other_tag_vecs==[]:
                self.vec_net.train()
                output_tag_vec, embedded_sentence=self.vec_net(token_id_seq)
                
                orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
                
                # 取递减偏离的一点
                target_vec = orig_tag_vec + calc_target_offset(orig_tag_vec, output_tag_vec.detach(), self.tag_dict[current_tag]["time"])
                
                self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
                self.tag_dict[current_tag]["time"]+=1

                self.vec_net.zero_grad()
                loss=self.vec_net_criterion(output_tag_vec, target_vec)
                self.loss_dict["VecNet Continual Single Train Loss"].append(loss.tolist())
                loss.backward()
                self.vec_net_optimizer.step()
                
                # -------------------------------------
                loss=train_gen(embedded_sentence.detach())
                self.loss_dict["GenNet Continual Single Train Loss"].append(loss)
            else:
                
                other_tag_vecs=torch.stack(other_tag_vecs)
                
                # 要生成的长度就设置为current_text的token的个数
                seq_lengths = torch.LongTensor([len(token_id_seq[0]) for i in range(len(other_tag_vecs))])

                # 用gen_net生成相同长度的陪练embeddings
                self.gen_net.eval()
                with torch.no_grad():
                    # 用gen_net预测输出embedding domain的陪练seq
                    attach_embedding=self.gen_net(other_tag_vecs, seq_lengths)

                # -------------------------------------
                # 开始训练vec_net
                
                # 把current_text的id_seq
                # 以及gen_net预测输出的embedding domain的陪练seq
                # 一起传入vec_net
                self.vec_net.train()
                
                try:
                    output_tag_vecs, embedded_sentence=self.vec_net(token_id_seq, attach_embedding=attach_embedding.detach())
                except Exception as e:
                    print("ERROR occur in .line1")
                    print(e)
                    print(current_text, current_tag)
                    return
                
                orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
                # 取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
                # 取递减偏离的一点
                target_vec = orig_tag_vec + calc_target_offset(orig_tag_vec, output_tag_vecs[-1,...].detach().reshape(1,-1), self.tag_dict[current_tag]["time"])

                # 把所有的tag domain的vec都弄起来
                # 因为在vec_net中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
                target_vecs=torch.cat([ other_tag_vecs, target_vec ])
                
                self.tag_dict[current_tag]["vec"]=target_vec.reshape(-1)
                self.tag_dict[current_tag]["time"]+=1
                
                self.vec_net.zero_grad()
                loss=self.vec_net_criterion(output_tag_vecs, target_vecs)
                self.loss_dict["VecNet Continual Attach Train Loss"].append(loss.tolist())
                loss.backward()
                self.vec_net_optimizer.step()

                # -------------------------------------
                # 最后用新得到的tag_vec，训练一下gen_net
                loss=train_gen(embedded_sentence.detach())
                self.loss_dict["GenNet Continual Attach Train Loss"].append(loss)
    

    def forward_without_generator(self, current_text, current_tag):
        # "单样本正向训练，不生成向量进行陪练"

        current_text=pre_process(current_text)
        token_id_seq=self.text_to_id_sequence(current_text)

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if type(self.tag_dict.get(current_tag))==type(None):
            
            self.vec_net.eval()
            with torch.no_grad():
                # 输入id sequence的id_seq(1xn)
                # 先用vec_net预测，输出tag domain的向量tag_vec(1 x TAG_VEC_DIM)，以及embed的vec(nx300)（用来之后训练gen_net
                tag_vec, embedded_sentence=self.vec_net(token_id_seq)

            # 把tag_vec存储到tag_dict中
            self.tag_dict[current_tag]={
                "vec":tag_vec.detach()[0,...], # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                "time":1
            }

        else:
            
            self.vec_net.train()
            output_tag_vec, _=self.vec_net(token_id_seq)
            
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_vec = orig_tag_vec + calc_target_offset(orig_tag_vec, output_tag_vec.detach(), self.tag_dict[current_tag]["time"])
            
            self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
            self.tag_dict[current_tag]["time"]+=1

            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_tag_vec, target_vec)
            self.loss_dict["VecNet Continual Baseline Train Loss"].append(loss.tolist())
            loss.backward()
            self.vec_net_optimizer.step()
    
    def backward(self, current_text, current_tag):

        # 这里和forward的训练方法大致一样，区别是：target_vecs是反方向的，并且不包括对gen_net的训练

        # -------------------------------------
        # 先用gen_net预测生成current_tag之外的其他tag的所对应的embedding作为训练vec_net的陪练
        # 以应对Catastrophic Forgetting
        current_text=pre_process(current_text)

        # 堆叠其他tag的vec
        other_tag_vecs=[]
        for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict),TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
            if tag!=current_tag:
                other_tag_vecs.append(value["vec"])
        
        token_id_seq=self.text_to_id_sequence(current_text)
        
        if other_tag_vecs==[]:
            self.vec_net.train()
            output_tag_vec, _=self.vec_net(token_id_seq)
            
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_vec = orig_tag_vec - calc_target_offset(orig_tag_vec, output_tag_vec.detach(), self.tag_dict[current_tag]["time"])
            
            self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
            if self.tag_dict[current_tag]["time"]!=1:
                self.tag_dict[current_tag]["time"]-=0.5

            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_tag_vec, target_vec)
            loss.backward()
            self.vec_net_optimizer.step()
        else:
            
            other_tag_vecs=torch.stack(other_tag_vecs)
            
            # 要生成的长度就设置为current_text的token的个数
            seq_lengths = torch.LongTensor([len(token_id_seq[0]) for i in range(len(other_tag_vecs))])

            # 用gen_net生成相同长度的陪练embeddings
            self.gen_net.eval()
            with torch.no_grad():
                # 用gen_net预测输出embedding domain的陪练seq
                attach_embedding=self.gen_net(other_tag_vecs,seq_lengths)

            # -------------------------------------
            # 开始训练vec_net
            
            # 把current_text的id_seq
            # 以及gen_net预测输出的embedding domain的陪练seq
            # 一起传入vec_net
            self.vec_net.train()
            
            try:
                output_tag_vecs, _=self.vec_net(token_id_seq,attach_embedding=attach_embedding.detach())
            except Exception as e:
                print("ERROR occur in .line2")
                print(e)
                print(current_text, current_tag)
                return
            
            # 有进有退
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            offset = calc_target_offset(orig_tag_vec, output_tag_vecs[-1,...].detach().reshape(1,-1), self.tag_dict[current_tag]["time"])
            target_vecs = torch.cat( [other_tag_vecs + offset, orig_tag_vec - offset] )
            self.tag_dict[current_tag]["vec"]=(orig_tag_vec - offset).reshape(-1)
            
            if self.tag_dict[current_tag]["time"]!=1:
                self.tag_dict[current_tag]["time"]-=0.5
            
            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_tag_vecs, target_vecs)
            loss.backward()
            self.vec_net_optimizer.step()

    def batch_train(self):
        
        # 遍历一遍训练集中所有的tag，如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        self.vec_net.eval()
        with torch.no_grad():
            
            for tag, values in self.train_dict.items():

                sample_text=values[random.randint(0,len(values)-1)]
                sample_text=pre_process(sample_text)
                if self.tag_dict.get(tag)==None:
                    
                    # 生成vec序列，这里是形状如[[...], [...], ...]的python的列表
                    token_id_seq=self.text_to_id_sequence(sample_text)
                    
                    # 将 padding好的序列 以及 序列中句子的长度 输入网络
                    output_vecs, _=self.vec_net(token_id_seq)

                    # 初始参数的网络的输出作为初始的tag_vec
                    self.tag_dict[tag]={
                        "vec":output_vecs.detach()[0,...], # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                        "time":1
                    }
        
        # 开始以batch_size为一个batch，对全体训练集进行小批量训练
        self.vec_net.train()
        batch_size=32
        o=0
        while o<len(self.train_pipe):
            
            batch_texts=[]
            target_vec=[]
            for i in self.train_pipe[o:o+batch_size]:
                text=i[0]
                tag=i[1]
                text=pre_process(text)
                batch_texts.append(text)
                target_vec.append(self.tag_dict[tag]["vec"])

            target_vec=torch.stack(target_vec)

            # 生成vec序列，这里是形状如[[...], [...], ...]的python的列表
            token_id_seq, seq_lengths, perm_idx = self.text_to_id_sequence_with_padding(batch_texts)

            # 将 padding好的序列 以及 序列中句子的长度 输入网络
            output_vecs=self.vec_net(token_id_seq,seq_lengths=seq_lengths)

            # 用perm_idx对target_vec重排序
            target_vec=target_vec[perm_idx]
            
            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_vecs, target_vec)
            self.loss_dict["VecNet Batch Train Loss"].append(loss.tolist())
            print(loss)
            loss.backward()
            self.vec_net_optimizer.step()
            o+=batch_size

    def predict(self, text, top_num):
        text=pre_process(text)
        if text:
            self.vec_net.eval()
            result={}
            with torch.no_grad():
                token_id_seq=self.text_to_id_sequence(text)

                # 将 padding好的序列 以及 序列中句子的长度 输入网络
                output_vecs, _=self.vec_net(token_id_seq)

                for tag,value in self.tag_dict.items():
                    # 计算距离
                    vec=value["vec"]
                    result[tag]=torch.linalg.norm(vec-output_vecs.detach()).tolist()
                result=sorted(result.items(),key=lambda x:x[1])
    
            if top_num==-1:
                return result
            else:
                return [i[0] for i in result[:top_num]]