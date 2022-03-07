# coding=UTF-8
import os
from matplotlib.pyplot import legend
import torch
from network import VecNet,GenNet
from utils import *
from DTPySide import *
import jieba

from Window import Ui_Window
class window(QWidget,Ui_Window):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

class MainSession(DTSession.DTMainSession):
    def __init__(self, app):
        super().__init__(app)
    
    def initializeWindow(self):
        super().initializeWindow()
        self.module=window(self)
        self.setCentralWidget(self.module)

    def initializeSignal(self):
        super().initializeSignal()
        self.module.pushButton_forward.clicked.connect(self.forward)
        self.module.pushButton_backward.clicked.connect(self.backward)
        self.module.pushButton_pred.clicked.connect(self.predict)
    
    def loadData(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:",self.device)

        # 先把word vec存储到network的embedding层里面保存起来，以后直接读取整个模型就行了
        if not os.path.exists("model_para_init.pt"):
            self.word_id_dict=load_WordIdDict()
            self.word_embeddings=load_WordEmbeddings(self.word_id_dict)

            self.vec_net=VecNet(len(self.word_id_dict),self.word_embeddings)
            self.vec_net.to(self.device)
            self.vec_net_optimizer = torch.optim.Adam(self.vec_net.parameters(), lr=1e-4)

            init_network(self.vec_net)

            for name,weight in self.vec_net.named_parameters():
                print(name,weight.shape)

            torch.save(self.vec_net.state_dict(), "model_para_init.pt")
        else:
            self.word_id_dict=load_WordIdDict()

            self.vec_net=VecNet(len(self.word_id_dict))
            self.vec_net.load_state_dict(torch.load("model_para_init.pt"))
            self.vec_net.to(self.device)
            self.vec_net_optimizer = torch.optim.Adam(self.vec_net.parameters(), lr=0.005)
            self.vec_net_criterion = torch.nn.MSELoss()
            
            self.gen_net=GenNet()
            self.gen_net.to(self.device)
            self.gen_net_optimizer=torch.optim.Adam(self.gen_net.parameters(), lr=0.005)
            self.gen_net_criterion = torch.nn.MSELoss()
            init_network(self.gen_net) #######
            
            if not os.path.exists("tag_dict.json"):
                self.tag_dict={}
                Json_Save(self.tag_dict,"tag_dict.json")
            else:
                self.tag_dict=Json_Load("tag_dict.json")

            # self.single_train_test()
            # self.full_train_test()
            self.show()
    
    def single_train_test(self):
        self.train_pipe,self.test_pipe=load_corpus()
        # self.forward("宅是一些什么样的人呢？宅一定就很不好么","现视研")
        # self.acc_list={"现视研":[],"Astro":[],"大蒜":[],"AGI":[],"gen_net_loss1":[],"vec_net_loss1":[],"gen_net_loss2":[],"vec_net_loss2":[]}
        self.acc_list={"俄罗斯":[],"Universe":[],"gen_net_loss1":[],"vec_net_loss1":[],"gen_net_loss2":[],"vec_net_loss2":[]}
        # n=100
        # for i in tqdm(range(n)):
        #     self.forward("宇宙是由大约70%的暗能量和30%的物质构成的。","Astro")
        #     self.pred_run()


        for j in range(5):
            for i in tqdm(self.train_pipe):
                self.forward(i[0],i[1])
                # print()
                # print(i[1])
                self.pred_run()
                # input()
        # self.pred_run()
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.acc_list["俄罗斯"])
        plt.plot(self.acc_list["Universe"])
        # plt.plot(self.acc_list["现视研"])
        # plt.plot(self.acc_list["Astro"])
        # plt.plot(self.acc_list["大蒜"])
        # plt.plot(self.acc_list["AGI"])
        # plt.legend(["现视研","Astro","大蒜","AGI"])
        plt.legend(["俄罗斯","Universe"])
        
        plt.figure()
        plt.plot(self.acc_list["gen_net_loss1"])
        plt.plot(self.acc_list["vec_net_loss1"])
        plt.plot(self.acc_list["gen_net_loss2"])
        plt.plot(self.acc_list["vec_net_loss2"])
        plt.legend(["gen_net_loss1","vec_net_loss1","gen_net_loss2","vec_net_loss2"])
        
        plt.show()
        
    def full_train_test(self):
        self.train_pipe,self.test_pipe=load_corpus()

        
        first=[
            ["大蒜","吃了几回蒜才发现，这玩意跟吸毒一样，初时诚惶诚恐，百般排斥；稍加适应便入心入骨，无法自拔。从味道来说，大蒜味冲。但就像芥末一般，若适量入口，则丹田如内力涌动。若胆子大的，用作死的节奏，将一颗或者一整头全部塞入口中，更有奇效。先是面红眼赤说不出话来，再是面目狰狞如泣如诉；高潮时浑身一紧，宛若段誉小子吃了什么剧毒又大补的神物。此关键处一定要凝神静气，挨过几秒，待头顶白烟缓缓散去，周身如打通任督二脉一般通透。此时，睁开眼睛，对众人一笑，说一句：“啊！舒服”。这头蒜算没白费。"],
            ["现视研","《现视研》并没有像它的众多后继者那样，让“宅”属性沦为附庸在角色身上的一个“萌点”，而是老老实实描写了各类只在otaku之间发生、让人会心一笑的故事：前辈们通过偷窥新生对小黄本的反应来“招贤纳才”；日常中最重大的活动是cosplay表演和“不签”的鉴赏会；腐女在部室里对男性同伴止不住的“妄想”；以社团名义参加comifes能被当成一生的光荣来炫耀……"],
            ["AGI","此外，在类脑计算整个发展的路径上，整个类脑领域又逐渐和脉冲神经网络（SNN），忆阻器，存算一体等概念产生了一定的绑定，更是给软硬件架构带来毁灭性冲击。忆阻器把模拟计算引入了进来，存算一体把非冯架构引入了进来，而SNN又一定程度把深度学习挡在了外面。这几点每一个都在给整个领域引入巨大的难度。现阶段大部分类脑方面的努力都在和这些问题做斗争，我们哪里还有精力去思考如何类脑来制造更强的智能？我博士最后一年写了一篇关于类脑解耦合的论文发在了nature正刊上，核心想法是想把类脑计算系统从这种缺乏架构思维的困局中拉回来，至少拉回到和现在的计算机系统的起点处。但说实话，这种工作其实也没有解决什么科学问题，因为即使拉回来了，类脑计算系统的能力现在也只处于计算机系统起步的年代，注意这里不是类比，类脑计算系统和计算机系统并不是平行的赛道，两者其实就是一个赛道！所以那篇论文我自己都觉得没多大用，从计算机的视角看更没有带来太多新的东西。更多是一种比较含蓄的方式尝试指出类脑计算领域现在的问题。"],
            ["Astro","但是，银河系中所有的物质，包括恒星本身以及形成恒星的物质，都是动态的。假如存在这样一种情况，地球距离太阳仍是现在这么远，但却是静止的，那么它会立即朝着太阳直线下坠；它之所以不会坠落到太阳上，是因为它处于轨道上，正在绕太阳运行。"]
        ]
        self.vec_net.eval()
        for current_text in first:
            current_tag=current_text[0]
            current_text=current_text[1]
            # 生成vec序列，这里是形状如[[...], [...], ...]的python的列表
            seq_tensor,seq_lengths,_=self.text_to_id_sequence([current_text])
            
            # 将 padding好的序列 以及 序列中句子的长度 输入网络
            output_vecs=self.vec_net(seq_tensor,seq_lengths)
            
            # print(output_vecs.detach()[0,...])

            self.tag_dict[current_tag]={
                "vec":output_vecs.detach().cpu()[0,...], # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
            }
        
        for j in range(15):
            
            self.vec_net.train()
            
            batch_texts=[]
            target_vec=[]
            for i in self.train_pipe:
                text=i[0]
                tag=i[1]

                batch_texts.append(text)
                target_vec.append(self.tag_dict[tag]["vec"])

            target_vec=torch.stack(target_vec)

            # 生成vec序列，这里是形状如[[...], [...], ...]的python的列表
            seq_tensor,seq_lengths,perm_idx=self.text_to_id_sequence(batch_texts)

            # 将 padding好的序列 以及 序列中句子的长度 输入网络
            output_vecs=self.vec_net(seq_tensor,seq_lengths)

            # 用perm_idx对target_vec重排序
            target_vec=target_vec[perm_idx]
            center=torch.mean(torch.stack([output_vecs.detach().cpu(), target_vec]),dim=0) # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
            
            self.vec_net.zero_grad()
            loss=self.vec_net_criterion(output_vecs, center.to(self.device))
            loss.backward()
            self.vec_net_optimizer.step()
    
            self.pred_run()
            print()
    
    def pred_run(self):
        for key, value in self.test_pipe.items():
        # for key, value in [("现视研",["宅是一些什么样的人","宅一定就很不好么","宅是很不好的么"]),("Astro",["宇宙是由大约70%的暗能量构成的。","宇宙是由30%的物质构成的。","宇宙是由暗能量和物质构成的。"])]:
            single_fault=1/len(value)
            acc=1
            for text in value:
                pred_tag=self.predict(text)
                if pred_tag!=key:
                    acc-=single_fault
            self.acc_list[key].append(acc)
            # print("Prediction for",key," ---- ","%.2f%%"%(acc*100))
    
    def saveData(self):
        # torch.save(self.vec_net.state_dict(), "model_para_01.pt")
        # Json_Save(self.tag_dict,"tag_dict.json")
        pass

    # def text_to_id_sequence_with_padding(self,text_list):
    #     "转换成word_id，padding，再排序"

    #     # 生成id序列，这里是形状如[[...], [...], ...]的python的列表
    #     id_seqs=[]
    #     for t in text_list:
    #         sequence=[]
    #         for i in jieba.cut(t,cut_all=False):
    #             id=self.word_id_dict.get(i, self.word_id_dict.get("<UNK>"))
    #             sequence.append(id)
    #         id_seqs.append(sequence)
        
    #     # 下面为了支持多个不同长度的输入，进行padding，以及加速运算的packing
        
    #     # 计算每句话的长度
    #     seq_lengths = torch.LongTensor(list(map(len, id_seqs)))
        
    #     # 生成shape为(len(id_seqs), 最大长度序列)的全0矩阵
    #     seq_tensor=torch.autograd.Variable(torch.zeros((len(id_seqs), seq_lengths.max()))).long().to(self.device)
        
    #     # 把id_seqs填进去
    #     for idx, (seq, seqlen) in enumerate(zip(id_seqs, seq_lengths)):
    #         seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        
    #     # 按照id_seqs中每句话的非零元素的多少排序（从多到少）
    #     seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

    #     # seq_lengths的顺序更新了，perm_idx记录了sort的操作
    #     # 用perm_idx对seq_tensor进行重排序
    #     seq_tensor = seq_tensor[perm_idx]

    #     return seq_tensor,seq_lengths,perm_idx

    def text_to_id_sequence(self,text):
        "转换成id sequence"
        
        sequence=[]
        for i in jieba.cut(text,cut_all=False):
            id=self.word_id_dict.get(i, self.word_id_dict.get("<UNK>"))
            sequence.append(id)
        
        seq_tensor=torch.LongTensor(sequence).reshape(1,-1).to(self.device)
        # seq_tensor=torch.autograd.Variable(torch.zeros((len(id_seqs), seq_lengths.max()))).long().to(self.device)

        return seq_tensor
    
    def forward(self, current_text=False, current_tag=False):

        if current_text==False and current_tag==False:
            current_text=self.module.lineEdit_train_text.text()
            current_tag=self.module.lineEdit_train_tag.text()

        current_text=pre_process(current_text)

        if type(self.tag_dict.get(current_tag))==type(None):
            
            seq_tensor=self.text_to_id_sequence(current_text)
            seq_lengths=[seq_tensor[0,...].shape[0]]
            
            # -------------------------------------
            
            self.vec_net.eval()
            # 输入id sequence的seq_tensor(1xn)
            # 先用vec_net预测，输出tag domain的向量tag_vec(1x10)，以及embed的vec(1x300)（用来之后训练gen_net
            tag_vec, embedded_sentence=self.vec_net(seq_tensor)

            # 把tag_vec存储到tag_dict中
            self.tag_dict[current_tag]={
                "vec":tag_vec.detach().cpu()[0,...] # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
            }

            # -------------------------------------
            # 再用tag_vec训练gen_net
            self.gen_net.train()
            # 把tag domain的tag_vec(1x10)输入gen_net，试图产生的embedding domain的output_seq(1x300)
            output_seq=self.gen_net(tag_vec.detach(),seq_lengths)

            # loss为gen_net的output_seq和embedded_sentence的差异
            # 试图让gen_net输入tag domain的vec，拟合输出embedding domain的vec
            self.gen_net.zero_grad()
            loss=self.gen_net_criterion(output_seq, embedded_sentence.detach())
            self.acc_list["gen_net_loss1"].append(loss.tolist())
            loss.backward()
            self.gen_net_optimizer.step()

            # -------------------------------------

        else:

            # -------------------------------------
            # 先用gen_net预测生成current_tag之外的其他tag的所对应的embedding作为训练vec_net的陪葬品
            # 以应对Catastrophic Forgetting
            

            # 堆叠其他tag的vec
            other_tag_vecs=[]
            for tag,value in self.tag_dict.items():
                if tag!=current_tag:
                    other_vec=value["vec"]
                    other_tag_vecs.append(other_vec)
            
            seq_tensor=self.text_to_id_sequence(current_text)
            
            if other_tag_vecs==[]:
                self.vec_net.train()
                output_tag_vecs, _=self.vec_net(seq_tensor)
                
                orig_tag_vecs=self.tag_dict[current_tag]["vec"].reshape(1,-1)
                # target_vecs=torch.mean(torch.stack([output_tag_vecs.detach().cpu(),orig_tag_vecs]),dim=0) # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                target_vecs=orig_tag_vecs+(output_tag_vecs.detach().cpu()-orig_tag_vecs)*0.1
                self.tag_dict[current_tag]["vec"]=target_vecs[-1,...]

                self.vec_net.zero_grad()
                loss=self.vec_net_criterion(output_tag_vecs, target_vecs.to(self.device))
                self.acc_list["vec_net_loss1"].append(loss.tolist())
                loss.backward()
                self.vec_net_optimizer.step()
            else:
                
                other_tag_vecs=torch.stack(other_tag_vecs)
                
                # 要生成的长度就设置为current_text的长度
                seq_lengths = torch.LongTensor([len(seq_tensor[0]) for i in range(len(other_tag_vecs))])

                self.gen_net.eval()
                # 用gen_net预测输出embedding domain的seq
                output_seq=self.gen_net(other_tag_vecs.to(self.device),seq_lengths)

                # -------------------------------------
                # 开始训练vec_net
                
                
                # 把current_text的embedding -> seq_tensor传入vec_net
                # 另外把gen_net预测输出的embedding domain的seq作为另一个参数传入vec_net
                self.vec_net.train()
                output_tag_vecs, embedded_sentence=self.vec_net(seq_tensor,attach_embedding=output_seq.detach())
                
                # 把所有的tag domain的vec都弄起来
                # 因为在vec_net中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
                orig_tag_vecs=torch.cat([ other_tag_vecs, self.tag_dict[current_tag]["vec"].reshape(1,-1) ])

                # target_vecs=torch.mean(torch.stack([output_tag_vecs.detach().cpu(),orig_tag_vecs]),dim=0) # 这里一定要detach掉，不然在计算loss之后，loss.backward会出错
                target_vecs=orig_tag_vecs+(output_tag_vecs.detach().cpu()-orig_tag_vecs)*0.1
                self.tag_dict[current_tag]["vec"]=target_vecs[-1,...]
                
                self.vec_net.zero_grad()
                loss=self.vec_net_criterion(output_tag_vecs, target_vecs.to(self.device))
                self.acc_list["vec_net_loss2"].append(loss.tolist())
                loss.backward()
                self.vec_net_optimizer.step()

                # -------------------------------------
                # 最后训练一下gen_net

                seq_lengths = torch.LongTensor([len(seq_tensor[0])])

                self.gen_net.train()
                
                output_seq=self.gen_net(self.tag_dict[current_tag]["vec"].reshape(1,-1).to(self.device),seq_lengths)
                
                # loss为gen_net的output_seq和embedded_sentence的差异
                # 试图让gen_net输入tag domain的vec，拟合输出embedding domain的vec
                self.gen_net.zero_grad()
                
                loss=self.gen_net_criterion(output_seq, embedded_sentence.detach())
                self.acc_list["gen_net_loss2"].append(loss.tolist())
                loss.backward()
                self.gen_net_optimizer.step()
        
    def backward(self):
        text=self.module.lineEdit_train_text.text()
        pass

    def predict(self, text):
        
        if type(text)==str or type(text)==list:
            text=pre_process(text)
            self.vec_net.eval()
            with torch.no_grad():
                seq_tensor=self.text_to_id_sequence(text)

                # 将 padding好的序列 以及 序列中句子的长度 输入网络
                output_vecs, _=self.vec_net(seq_tensor)

                result={}
                for tag,value in self.tag_dict.items():
                    # 计算距离
                    vec=value["vec"]
                    result[tag]=torch.linalg.norm(vec-output_vecs.detach().cpu())
                result=sorted(result.items(),key=lambda x:x[1])
            
            return result[0][0]
        else:
            text=self.module.lineEdit_pred_text.text()
            text=pre_process(text)
            self.vec_net.eval()
            with torch.no_grad():
                
                seq_tensor=self.text_to_id_sequence(text)

                # 将 padding好的序列 以及 序列中句子的长度 输入网络
                output_vecs, _=self.vec_net(seq_tensor)

                result={}
                for tag,value in self.tag_dict.items():
                    # 计算距离
                    vec=value["vec"]
                    result[tag]=torch.linalg.norm(vec-output_vecs.detach().cpu())
                result=sorted(result.items(),key=lambda x:x[1])
            
            text=""
            for i in result:
                text+=i[0]+" "+str(i[1])+"\n"
            self.module.plainTextEdit_pred_tag.setPlainText(text)

if __name__=="__main__":

    app=DTAPP(sys.argv)

    app.setApplicationName("Tagger")
    app.setWindowIcon(DTIcon.HoloIcon1())
    app.setAuthor("Holence")
    app.setApplicationVersion("1.0.0.0")
    app.setLoginEnable(False)

    session=MainSession(app)
    app.setMainSession(session)

    app.run()