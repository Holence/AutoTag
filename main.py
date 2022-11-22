import os
import torch
import matplotlib.pyplot as plt
from network import VecNet, GenNet
from utils import *
from DTPySide import *
import jieba
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

from Window import Ui_Window
class window(QWidget,Ui_Window):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

class MainSession(DTSession.DTMainSession):
    
    VEC_NET_LR=0.0005
    GEN_NET_LR=0.0005
    TARGET_VEC_INIT_DEIVATION=0.3
    TARGET_VEC_DECAY_RATE=0.8
    
    def __init__(self, app):
        super().__init__(app)
    
    def initializeWindow(self):
        super().initializeWindow()
        self.module=window(self)
        self.setCentralWidget(self.module)
        self.module.plainTextEdit_pred_text.setPlaceholderText("{TAG}lorem ipsum...\n\n===\n\n{标签}测试文本...")

    def initializeSignal(self):
        super().initializeSignal()
        self.module.pushButton_forward.clicked.connect(self.forward)
        self.module.pushButton_backward.clicked.connect(self.backward)
        self.module.pushButton_pred.clicked.connect(self.predict)
        
        self.module.pushButton_load_corpus.clicked.connect(self.loadCorpus)
        self.module.pushButton_continual_train.clicked.connect(self.continual_train)
        self.module.pushButton_normal_train.clicked.connect(self.normal_train)
        self.module.pushButton_plot.clicked.connect(self.plot)
        self.module.pushButton_plot2D.clicked.connect(self.plot2D)
        self.module.pushButton_eval.clicked.connect(self.evaluate)
        self.module.pushButton_save.clicked.connect(self.saveModel)
    
    def loadData(self):
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:",self.device)

        # 导入或生成word_id_dict
        # {
        #     "<PAD>": 0,
        #     "啊": 1,
        #     "<UNK>": ...
        # }
        self.word_id_dict=load_WordIdDict()

        # VecNet
        if not os.path.exists("vec_net_para_init.pt"):

            # 生成word_embeddings，numpy array
            # [
            #     [1x300 random],
            #     [1x300 from sgns.zhihu.char],
            #     [1x300 random],
            # ]
            word_embeddings=load_WordEmbeddings(self.word_id_dict)

            # 创建VecNet，并把word_embeddings存入VecNet的embedding层
            self.vec_net=VecNet(len(self.word_id_dict), word_embeddings)
            self.vec_net.to(self.device)

            # 初始化VecNet（除去embedding层，初始化LSTM、Linear Layer的参数）
            init_network(self.vec_net)

            for name,weight in self.vec_net.named_parameters():
                print(name,weight.shape)

            # 保存初始状态
            torch.save(self.vec_net.state_dict(), "vec_net_para_init.pt")
        else:
            
            self.vec_net=VecNet(len(self.word_id_dict))
            try:
                self.vec_net.load_state_dict(torch.load("vec_net_para.pt"))
            except:
                self.vec_net.load_state_dict(torch.load("vec_net_para_init.pt"))
            self.vec_net.to(self.device)
        
        # 学习率也应该保存并读取
        self.vec_net_optimizer = torch.optim.Adam(self.vec_net.parameters(), lr=self.VEC_NET_LR)
        self.vec_net_criterion = torch.nn.MSELoss()
        
        # GenNet
        self.gen_net=GenNet()
        try:
            self.gen_net.load_state_dict(torch.load("gen_net_para.pt"))
        except:
            init_network(self.gen_net)
        self.gen_net.to(self.device)
        
        # 学习率也应该保存并读取
        self.gen_net_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr=self.GEN_NET_LR)
        self.gen_net_criterion = torch.nn.MSELoss()
        
        if os.path.exists("tag_dict.pt"):
            self.tag_dict=torch.load("tag_dict.pt")
        else:
            self.tag_dict={}

        self.loss_dict={
            "VecNet Normal Train Loss":[],
            "GenNet Continual Single Train Loss":[],
            "GenNet Continual Attach Train Loss":[],
            "VecNet Continual Single Train Loss":[],
            "VecNet Continual Attach Train Loss":[]
        }
        self.acc_dict={}
        
        # for k,v in self.tag_dict.items():
        #     print(k)
        #     print(v)
        #     print()

    def loadCorpus(self):
        self.train_pipe, self.train_dict, self.test_dict=load_corpus("corpus", 0.5)
        # train_pipe
        # [("sentence","tag1"), ..., ("sentence","tag1"), ..., ("sentence","tag10")]
        
        # train_dict
        # {
        #   "tag1": ["sentence", ..., "sentence"]
        #    ...
        #   "tag10": ["sentence", ..., "sentence"]
        # }
        
        # test_dict
        # {
        #   "tag1": ["sentence", ..., "sentence"]
        #    ...
        #   "tag10": ["sentence", ..., "sentence"]
        # }

    
    def saveModel(self):
        torch.save(self.vec_net.state_dict(), "vec_net_para.pt")
        torch.save(self.gen_net.state_dict(), "gen_net_para.pt")
        torch.save(self.tag_dict, "tag_dict.pt")

    def evaluate(self):
        if len(self.tag_dict)==0:
            print("There isn't any tag_vec in tag_dict! Please train first!")
            return
        
        try:
            self.train_dict
        except:
            print("Evaluate is only used in traning corpus!")
            return
        
        self.module.textBrowser_res.clear()
        s, train_acc = self.pred_train(return_string_and_acc=True)
        self.module.textBrowser_res.append(s)
        s, test_acc = self.pred_test(return_string_and_acc=True)
        self.module.textBrowser_res.append(s)
        self.module.label_acc.setText("Train acc: %.2f%%  Test acc: %.2f%%"%(train_acc*100, test_acc*100))
    
    def plot(self):
        plt.ion()

        plt.figure()
        for key in self.loss_dict.keys():
            plt.plot(self.loss_dict[key])
        plt.legend(self.loss_dict.keys())

        plt.figure()
        for tag in [i for i in self.acc_dict.keys() if "test" in i]:
            plt.plot(self.acc_dict[tag])
        plt.legend([i for i in self.acc_dict.keys() if "test" in i])
        
        plt.figure()
        for tag in [i for i in self.acc_dict.keys() if "train" in i]:
            plt.plot(self.acc_dict[tag])
        plt.legend([i for i in self.acc_dict.keys() if "train" in i])

    def plot2D(self):
        def euclidean(x0, x1):
            x0, x1 = np.array(x0), np.array(x1)
            d = np.sum((x0 - x1)**2)**0.5
            return d
        
        def scaledown(X, distance=euclidean, rate=0.1, itera=1000, rand_time=10, verbose=1):
            n = len(X)
            
            # calculate distances martix in high dimensional space
            realdist = np.array([[distance(X[i], X[j]) for j in range(n)] for i in range(n)])
            realdist = realdist / np.max(realdist)  # rescale between 0-1
            
            min_error = None
            for i in range(rand_time): # search for n times
                
                if verbose: print("%s/%s, min_error=%s"%(i, rand_time, min_error))
                
                # initilalize location in 2-D plane randomly
                loc = np.array([[np.random.random(), np.random.random()] for i in range(n)])

                # start iterating
                last_error = None
                for m in range(itera):

                    # calculate distance in 2D plane
                    twoD_dist = np.array([[np.sum((loc[i] - loc[j])**2)**0.5 for j in range(n)] for i in range(n)])

                    # calculate move step
                    move_step = np.zeros_like(loc)
                    total_error = 0
                    for i in range(n):
                        for j in range(n):                
                            if realdist[i, j] <= 0.01: continue               
                            error_rate = (twoD_dist[i, j] - realdist[i, j]) / twoD_dist[i, j]                
                            move_step[i, 0] += ((loc[i, 0] - loc[j, 0]) / twoD_dist[i, j])*error_rate
                            move_step[i, 1] += ((loc[i, 1] - loc[j, 1]) / twoD_dist[i, j])*error_rate
                            total_error += abs(error_rate)

                    if last_error and total_error > last_error: break  # stop iterating if error becomes worse
                    last_error = total_error

                    # update location
                    loc -= rate*move_step

                # save best location
                if min_error is None or last_error < min_error:
                    min_error = last_error
                    best_loc = loc
                
            return best_loc
        
        X=np.stack([v["vec"] for v in self.tag_dict.values()]) 
        label = [k for k in self.tag_dict.keys()]

        loc = scaledown(X, itera=20000, rand_time=500, verbose=1)
        x = loc[:,0]
        y = loc[:,1]

        plt.ion()
        plt.figure()
        plt.scatter(x,y)
        for x_, y_, s in zip(x,y,label):
            plt.annotate(s, (x_, y_))
        plt.show()

    def continual_train(self):

        try:
            self.train_dict
        except:
            print("Evaluate is only used in traning corpus!")
            return
        
        o=0
        random.shuffle(self.train_pipe)
        for i in tqdm(self.train_pipe):
            self.forward(i[0],i[1])
            # o+=1
            # if o%15==0:
            #     self.pred_train()
            #     self.pred_test()
        
        self.pred_train()
        self.pred_test()
        self.plot()
        
    def normal_train(self):
        
        try:
            self.train_dict
        except:
            print("Evaluate is only used in traning corpus!")
            return
        
        # 遍历一遍训练集中所有的tag，如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        self.vec_net.eval()
        with torch.no_grad():
            
            for tag, values in self.train_dict.items():

                # 狗皮膏药
                if self.acc_dict.get(tag)==None:
                    self.acc_dict[tag+"_test"]=[]
                    self.acc_dict[tag+"_train"]=[]

                sample_text=values[random.randint(0,len(values)-1)]

                if self.tag_dict.get(tag)==None:
                    
                    # 生成vec序列，这里是形状如[[...], [...], ...]的python的列表
                    token_id_seq=self.text_to_id_sequence(sample_text)
                    
                    # 将 padding好的序列 以及 序列中句子的长度 输入网络
                    output_vecs, _=self.vec_net(token_id_seq)

                    # 初始参数的网络的输出作为初始的tag_vec
                    self.tag_dict[tag]={
                        "vec":output_vecs.detach().cpu()[0,...], # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                        "time":1
                    }
        
        random.shuffle(self.train_pipe)
        
        # 开始以batch_size为一个batch，对全体训练集进行小批量训练
        batch_size=24
        o=0
        while o<len(self.train_pipe):
            
            self.vec_net.train()

            batch_texts=[]
            target_vec=[]
            for i in self.train_pipe[o:o+batch_size]:
                text=i[0]
                tag=i[1]

                batch_texts.append(text)
                target_vec.append(self.tag_dict[tag]["vec"])

            target_vec=torch.stack(target_vec)

            # 生成vec序列，这里是形状如[[...], [...], ...]的python的列表
            token_id_seq,seq_lengths,perm_idx=self.text_to_id_sequence_with_padding(batch_texts)

            # 将 padding好的序列 以及 序列中句子的长度 输入网络
            output_vecs=self.vec_net(token_id_seq,seq_lengths=seq_lengths)

            # 用perm_idx对target_vec重排序
            target_vec=target_vec[perm_idx]
            # center=torch.mean(torch.stack([output_vecs.detach().cpu(), target_vec]),dim=0) # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
            center=target_vec+(output_vecs.detach().cpu()-target_vec)*0.05
            
            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_vecs, center.to(self.device))
            self.loss_dict["VecNet Normal Train Loss"].append(loss.tolist())
            loss.backward()
            self.vec_net_optimizer.step()
            o+=batch_size

            self.pred_train()
            self.pred_test()
        
        self.pred_train()
        self.pred_test()
        self.plot()
    
    def pred_train(self, return_string_and_acc=False):
        if len(self.tag_dict)==0:
            print("There isn't any tag_vec in tag_dict! Please train first!")
            return
        
        s=""
        total=0
        total_correct=0
        for key, value in self.train_dict.items():
            total+=len(value)
            single_fault=1/len(value)
            acc=1
            for text in value:
                pred_tag=self.predict(text)
                if pred_tag!=key:
                    acc-=single_fault
                else:
                    total_correct+=1
            self.acc_dict[key+"_train"].append(acc)
            s+="Train Prediction for %10s ---- %.2f%%\n"%(key, acc*100)

        if return_string_and_acc==True:
            return s, total_correct/total
        else:
            print(s)
        
    def pred_test(self, return_string_and_acc=False):
        if len(self.tag_dict)==0:
            print("There isn't any tag_vec in tag_dict! Please train first!")
            return
        
        s=""
        total=0
        total_correct=0
        for key, value in self.test_dict.items():
            total+=len(value)
            single_fault=1/len(value)
            acc=1
            for text in value:
                pred_tag=self.predict(text)
                if pred_tag!=key:
                    acc-=single_fault
                else:
                    total_correct+=1
            self.acc_dict[key+"_test"].append(acc)
            s+="Test Prediction for %10s ---- %.2f%%\n"%(key, acc*100)
        
        if return_string_and_acc==True:
            return s, total_correct/total
        else:
            print(s)
    
    def saveData(self):
        # torch.save(self.vec_net.state_dict(), "model_para_01.pt")
        # Json_Save(self.tag_dict,"tag_dict.json")
        pass

    def text_to_id_sequence_with_padding(self,text_list):
        "转换成word_id，padding，再排序"

        # 生成id序列，这里是形状如[[...], [...], ...]的python的列表
        id_seqs=[]
        for t in text_list:
            sequence=[]
            for i in jieba.cut(t,cut_all=False):
                if i.strip():
                    id=self.word_id_dict.get(i, self.word_id_dict.get("<UNK>"))
                    sequence.append(id)
            id_seqs.append(sequence)
        
        # 下面为了支持多个不同长度的输入，进行padding，以及加速运算的packing
        
        # 计算每句话的长度
        seq_lengths = torch.LongTensor(list(map(len, id_seqs)))
        
        # 生成shape为(len(id_seqs), 最大长度序列)的全0矩阵
        token_id_seq=torch.autograd.Variable(torch.zeros((len(id_seqs), seq_lengths.max()))).long().to(self.device)
        
        # 把id_seqs填进去
        for idx, (seq, seqlen) in enumerate(zip(id_seqs, seq_lengths)):
            token_id_seq[idx, :seqlen] = torch.LongTensor(seq)
        
        # 按照id_seqs中每句话的非零元素的多少排序（从多到少）
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        # seq_lengths的顺序更新了，perm_idx记录了sort的操作
        # 用perm_idx对id_seq进行重排序
        token_id_seq = token_id_seq[perm_idx]

        return token_id_seq,seq_lengths,perm_idx

    def text_to_id_sequence(self,text):
        "转换成id sequence"
        
        sequence=[]
        
        for i in jieba.cut(text,cut_all=False):
            if i.strip():
                id=self.word_id_dict.get(i, self.word_id_dict.get("<UNK>"))
                sequence.append(id)
        
        token_id_seq=torch.LongTensor(sequence).reshape(1,-1).to(self.device)
        # token_id_seq=torch.autograd.Variable(torch.zeros((len(id_seqs), seq_lengths.max()))).long().to(self.device)

        return token_id_seq
    
    def forward(self, current_text=False, current_tag=False):
        "单样本正向训练"

        if current_text==False and current_tag==False:
            update_predict=True
            current_text=self.module.plainTextEdit_train_text.toPlainText()
            current_tag=self.module.lineEdit_train_tag.text()
        else:
            update_predict=False

        current_text=pre_process(current_text)

        # 狗皮膏药
        if self.acc_dict.get(current_tag)==None:
            self.acc_dict[current_tag+"_test"]=[]
            self.acc_dict[current_tag+"_train"]=[]

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if type(self.tag_dict.get(current_tag))==type(None):
            # 把sentence转换成token的id序列
            token_id_seq=self.text_to_id_sequence(current_text)
            # [token的个数]
            seq_lengths=[token_id_seq[0,...].shape[0]]
            
            # -------------------------------------
            
            self.vec_net.eval()
            with torch.no_grad():
                # 输入id sequence的id_seq(1xn)
                # 先用vec_net预测，输出tag domain的向量tag_vec(1 x TAG_VEC_DIM)，以及embed的vec(nx300)（用来之后训练gen_net
                tag_vec, embedded_sentence=self.vec_net(token_id_seq)

            # 把tag_vec存储到tag_dict中
            self.tag_dict[current_tag]={
                "vec":tag_vec.detach().cpu()[0,...], # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                "time":1
            }
            
            # -------------------------------------
            # 再用tag_vec训练gen_net
            self.gen_net.train()
            # 把tag domain的tag_vec(1 x TAG_VEC_DIM)输入gen_net，试图产生的embedding domain的output_seq(nx300)
            output_seq=self.gen_net(tag_vec.detach(),seq_lengths)

            # loss为gen_net的output_seq和embedded_sentence的差异
            # 试图让gen_net输入tag domain的vec，拟合输出embedding domain的vec
            self.gen_net.zero_grad()
            loss=self.gen_net_criterion(output_seq, embedded_sentence.detach())
            self.loss_dict["GenNet Continual Single Train Loss"].append(loss.tolist())
            loss.backward()
            self.gen_net_optimizer.step()

        else:
            
            # -------------------------------------
            # 先用gen_net预测生成current_tag之外的其他tag的所对应的embedding作为训练vec_net的陪练
            # 以应对Catastrophic Forgetting
            

            # 堆叠其他tag的vec
            other_tag_vecs=[]
            for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict),10) ): # 随机取10个来伴随训练，太多了的话内存不够用
                if tag!=current_tag:
                    other_vec=value["vec"]
                    other_tag_vecs.append(other_vec)
            
            token_id_seq=self.text_to_id_sequence(current_text)
            
            # 如果没有其他陪练的，就取vec_net的输出为TARGET_VEC，取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
            if other_tag_vecs==[]:
                self.vec_net.train()
                output_tag_vec, _=self.vec_net(token_id_seq)
                
                orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
                
                # 取递减偏离的一点
                target_vec=orig_tag_vec+(output_tag_vec.detach().cpu()-orig_tag_vec)*self.TARGET_VEC_INIT_DEIVATION*(self.TARGET_VEC_DECAY_RATE**self.tag_dict[current_tag]["time"])
                
                self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
                self.tag_dict[current_tag]["time"]+=1

                self.vec_net.zero_grad()
                loss=self.vec_net_criterion(output_tag_vec, target_vec.to(self.device))
                self.loss_dict["VecNet Continual Single Train Loss"].append(loss.tolist())
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
                    attach_embedding=self.gen_net(other_tag_vecs.to(self.device),seq_lengths)

                # -------------------------------------
                # 开始训练vec_net
                
                # 把current_text的id_seq
                # 以及gen_net预测输出的embedding domain的陪练seq
                # 一起传入vec_net
                self.vec_net.train()
                
                try:
                    output_tag_vecs, embedded_sentence=self.vec_net(token_id_seq, attach_embedding=attach_embedding.detach())
                except:
                    print("ERROR occur in .line1")
                    print(current_text, current_tag)
                    return
                
                orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
                # 取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
                # 取递减偏离的一点
                target_vec=orig_tag_vec+(output_tag_vecs[-1,...].detach().cpu().reshape(1,-1)-orig_tag_vec)*self.TARGET_VEC_INIT_DEIVATION*(self.TARGET_VEC_DECAY_RATE**self.tag_dict[current_tag]["time"])

                # 把所有的tag domain的vec都弄起来
                # 因为在vec_net中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
                target_vecs=torch.cat([ other_tag_vecs, target_vec ])
                
                self.tag_dict[current_tag]["vec"]=target_vec.reshape(-1)
                self.tag_dict[current_tag]["time"]+=1
                
                self.vec_net.zero_grad()
                loss=self.vec_net_criterion(output_tag_vecs, target_vecs.to(self.device))
                self.loss_dict["VecNet Continual Attach Train Loss"].append(loss.tolist())
                loss.backward()
                self.vec_net_optimizer.step()

                # -------------------------------------
                # 最后用新得到的tag_vec，训练一下gen_net

                seq_lengths = torch.LongTensor([len(token_id_seq[0])])

                self.gen_net.train()
                
                output_seq=self.gen_net(self.tag_dict[current_tag]["vec"].reshape(1,-1).to(self.device), seq_lengths)
                
                # loss为gen_net的output_seq和embedded_sentence的差异
                # 试图让gen_net输入tag domain的vec，拟合输出embedding domain的vec
                self.gen_net.zero_grad()
                
                loss=self.gen_net_criterion(output_seq, embedded_sentence.detach())
                self.loss_dict["GenNet Continual Attach Train Loss"].append(loss.tolist())
                loss.backward()
                self.gen_net_optimizer.step()
        
        if update_predict:
            self.predict(None)
        
    def backward(self):
        
        current_text=self.module.plainTextEdit_train_text.toPlainText()
        current_tag=self.module.lineEdit_train_tag.text()
        if type(self.tag_dict.get(current_tag))==type(None):
            DTFrame.DTMessageBox(self,"Warning","%s is not in tag_dict, please forward train first!"%current_tag)
            return
            
        # 这里和forward的训练方法大致一样，区别是：target_vecs是反方向的，并且不包括对gen_net的训练

        # -------------------------------------
        # 先用gen_net预测生成current_tag之外的其他tag的所对应的embedding作为训练vec_net的陪练
        # 以应对Catastrophic Forgetting
        

        # 堆叠其他tag的vec
        other_tag_vecs=[]
        for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict),10) ): # 随机取10个来伴随训练，太多了的话内存不够用
            if tag!=current_tag:
                other_vec=value["vec"]
                other_tag_vecs.append(other_vec)
        
        token_id_seq=self.text_to_id_sequence(current_text)
        
        if other_tag_vecs==[]:
            self.vec_net.train()
            output_tag_vec, _=self.vec_net(token_id_seq)
            
            orig_tag_vecs=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_vec=orig_tag_vecs-(output_tag_vec.detach().cpu()-orig_tag_vecs)*self.TARGET_VEC_INIT_DEIVATION*(self.TARGET_VEC_DECAY_RATE**self.tag_dict[current_tag]["time"])
            
            self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
            if self.tag_dict[current_tag]["time"]!=1:
                self.tag_dict[current_tag]["time"]-=0.5

            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_tag_vec, target_vec.to(self.device))
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
                attach_embedding=self.gen_net(other_tag_vecs.to(self.device),seq_lengths)

            # -------------------------------------
            # 开始训练vec_net
            
            # 把current_text的id_seq
            # 以及gen_net预测输出的embedding domain的陪练seq
            # 一起传入vec_net
            self.vec_net.train()
            
            try:
                output_tag_vecs, embedded_sentence=self.vec_net(token_id_seq,attach_embedding=attach_embedding.detach())
            except:
                print("ERROR occur in .line2")
                print(current_text, current_tag)
                return
            
            #####################################################################################
            # # 所有的tag_vec都向后撤

            # # 把所有的tag domain的vec都弄起来
            # # 因为在vec_net中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
            # orig_tag_vecs=torch.cat([ other_tag_vecs, self.tag_dict[current_tag]["vec"].reshape(1,-1) ])

            # # 取递增偏离的一点
            # target_vecs=orig_tag_vecs-(output_tag_vecs.detach().cpu()-orig_tag_vecs)*self.TARGET_VEC_INIT_DEIVATION*(self.TARGET_VEC_DECAY_RATE**self.tag_dict[current_tag]["time"])
            
            # self.tag_dict[current_tag]["vec"]=target_vecs[-1,...]

            #####################################################################################
            # # 只有该tag_vec后撤
            
            # orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            # # 取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
            # # 取递减偏离的一点
            # target_vec=orig_tag_vec-(output_tag_vecs[-1,...].detach().cpu().reshape(1,-1)-orig_tag_vec)*self.TARGET_VEC_INIT_DEIVATION*(self.TARGET_VEC_DECAY_RATE**self.tag_dict[current_tag]["time"])
            
            # # 把所有的tag domain的vec都弄起来
            # # 因为在vec_net中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
            # target_vecs=torch.cat([ other_tag_vecs, target_vec ])
            
            # self.tag_dict[current_tag]["vec"]=target_vec.reshape(-1)

            #####################################################################################
            # 有进有退
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            offset = (output_tag_vecs[-1,...].detach().cpu().reshape(1,-1)-orig_tag_vec)*self.TARGET_VEC_INIT_DEIVATION*(self.TARGET_VEC_DECAY_RATE**self.tag_dict[current_tag]["time"])
            target_vec = orig_tag_vec - offset
            target_vecs = torch.cat( [other_tag_vecs + offset, target_vec] )
            self.tag_dict[current_tag]["vec"]=target_vec.reshape(-1)
            
            if self.tag_dict[current_tag]["time"]!=1:
                self.tag_dict[current_tag]["time"]-=0.5
            
            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_tag_vecs, target_vecs.to(self.device))
            loss.backward()
            self.vec_net_optimizer.step()
        
        self.predict(None)

    def predict(self, text):

        if len(self.tag_dict)==0:
            print("There isn't any tag_vec in tag_dict! Please train first!")
            return
        
        if type(text)==str:
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
                        result[tag]=torch.linalg.norm(vec-output_vecs.detach().cpu()).tolist()
                    result=sorted(result.items(),key=lambda x:x[1])
                
                return result[0][0]
        else:
            text=self.module.plainTextEdit_pred_text.toPlainText()
            scroll=self.module.textBrowser_res.verticalScrollBar().value()
            self.module.textBrowser_res.clear()

            text_list=[_.strip() for _ in text.split("===") if _.strip()]
            if text_list:
                res=""
                single_fault=1/len(text_list)
                acc=1
                for text in text_list:
                    
                    head=re.findall("^\{.*?\}",text)
                    if head!=[]:
                        target_tag=head[0].replace("{","").replace("}","")
                        text=re.sub("^\{.*?\}","",text)
                    else:
                        target_tag=None
                    
                    orig_text=text
                    text=pre_process(text)
                    if text:
                        
                        result={}
                        self.vec_net.eval()
                        with torch.no_grad():
                            
                            token_id_seq=self.text_to_id_sequence(text)
                            
                            # 将 padding好的序列 以及 序列中句子的长度 输入网络
                            output_vecs, _=self.vec_net(token_id_seq)

                            for tag,value in self.tag_dict.items():
                                # 计算距离
                                vec=value["vec"]
                                result[tag]=torch.linalg.norm(vec-output_vecs.detach().cpu()).tolist()
                            result=sorted(result.items(),key=lambda x:x[1])
                        
                        pred_prob=""
                        for i in result:
                            if i[0]==target_tag:
                                pred_prob+="%.5f  %s    <------\n"%(i[1], i[0])
                            else:
                                pred_prob+="%.5f  %s\n"%(i[1], i[0])
        
                        if target_tag!=None and result[0][0]!=target_tag:
                            # 预测错误
                            res+="~~"+orig_text+"~~"+"\n\n"+pred_prob+"\n===\n\n"
                            acc-=single_fault
                        else:
                            # 预测正确
                            res+=orig_text+"\n\n"+pred_prob+"\n===\n\n"
                
                res=res.replace("\n","\n\n")
                self.module.textBrowser_res.setMarkdown(res)
                self.module.textBrowser_res.verticalScrollBar().setValue(scroll)
                self.module.label_acc.setText("acc: %.2f%%"%(acc*100))

if __name__=="__main__":

    app=DTAPP(sys.argv)

    app.setApplicationName("AutoTag")
    app.setWindowIcon(DTIcon.HoloIcon1())
    app.setAuthor("Holence")
    app.setApplicationVersion("1.0.0.0")
    app.setLoginEnable(False)

    session=MainSession(app)
    app.setMainSession(session)

    app.run()