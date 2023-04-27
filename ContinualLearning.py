import torch
from utils import *

class ContinualLearningModel():
    
    def __init__(self, LanguageModel, Classifier, Generator, CLS_LR=0.0005, GEN_LR=0.0005, TRAIN_ALONG=10) -> None:
        self.LanguageModel=LanguageModel
        self.Classifier=Classifier
        self.Generator=Generator
        self.CLS_LR=CLS_LR
        self.GEN_LR=GEN_LR
        self.TRAIN_ALONG=TRAIN_ALONG
        self.initilize()
    
    def initilize(self):
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:",self.device)

        self.save_path=self.LanguageModel.__name__+"_Para"

        self.lm=self.LanguageModel()
        self.lm.to(self.device)

        # Classifier
        self.cls=self.Classifier()
        self.cls.to(self.device)
        # 学习率也应该保存并读取
        self.cls_optimizer = torch.optim.Adam(self.cls.parameters(), lr=self.CLS_LR)
        self.cls_criterion = torch.nn.MSELoss()
        
        # Generator
        self.gen=self.Generator()
        self.gen.to(self.device)
        # 学习率也应该保存并读取
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.GEN_LR)
        self.gen_criterion = torch.nn.MSELoss()

        # tag_dict
        if os.path.exists(f"{self.save_path}/tag_dict.pt"):
            self.tag_dict=torch.load(f"{self.save_path}/tag_dict.pt")
        else:
            self.tag_dict={}
        
        # loss_dict
        if os.path.exists(f"{self.save_path}/loss_dict.pt"):
            self.loss_dict=torch.load(f"{self.save_path}/loss_dict.pt")
        else:
            self.loss_dict={
                "Classifier Batch Train Loss":[],
                "Classifier Continual Single Train Loss":[],
                "Classifier Continual Attach Train Loss":[],
                "Classifier Continual Baseline Train Loss":[],
                "Generator Continual Single Train Loss":[],
            }
        
    def save(self):
        torch.save(self.cls.state_dict(), f"{self.save_path}/classifier_para.pt")
        torch.save(self.gen.state_dict(), f"{self.save_path}/generator_para.pt")
        torch.save(self.tag_dict, f"{self.save_path}/tag_dict.pt")
        torch.save(self.loss_dict, f"{self.save_path}/loss_dict.pt")
    
    def predict(self, sample_list, top_num=1):
        if sample_list:
            self.cls.eval()
            with torch.no_grad():
                feature_vecs=[]
                for i in sample_list:
                    feature_vec=self.lm(i).reshape(-1)
                    feature_vecs.append(feature_vec)
                feature_vecs=torch.stack(feature_vecs)
                
                output_tag_vecs=self.cls(feature_vecs)

            tag_vecs=torch.stack([i["tag_vec"] for i in self.tag_dict.values()])
            tags=[i for i in self.tag_dict.keys()]

            result=[]
            for output_tag_vec in output_tag_vecs:
                distance = torch.linalg.norm(tag_vecs-output_tag_vec, dim=1)
                _, sorted_index = distance.sort()
                if top_num==-1:
                    result.append([[tags[i], distance[i]] for i in sorted_index])
                else:
                    result.append([tags[i] for i in sorted_index][:top_num])
            
            return result

    
    def continual_forward(self, train_text, train_tag):
        # "单样本正向训练，生成向量进行陪练"
        
        def train_generator(tag_vec, target_feature_vec):
            #### 没用了
            #### 如果存储feature_dict，进行多样本训练Generator，那么在训练Classifier时所要的陪练该用Generator生成的还是feature_dict里存储的？
            #### 显然这样Generator就没有存在的必要了
            # 用一些tag的feature_vecs训练Generator
            # tag_vecs=[]
            # target_feature_vecs=[]
            # for tag, value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict), self.TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
            #     tag_vecs.append(value["tag_vec"])
            #     target_feature_vecs.append(self.feature_dict[tag])
            # tag_vecs=torch.stack(tag_vecs)
            # target_feature_vecs=torch.stack(target_feature_vecs)
            ####

            # 单样本训练Generator
            self.gen.train()
            output_feature_vecs=self.gen(tag_vec)
            self.gen_optimizer.zero_grad()
            loss=self.gen_criterion(output_feature_vecs, target_feature_vec)
            loss.backward()
            self.loss_dict["Generator Continual Single Train Loss"].append(loss.tolist())
            self.gen_optimizer.step()

        train_feature_vec=self.lm(train_text).detach() # 这里就detach掉，BERT是不用训练的，LSTM的话怎么弄

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if not self.tag_dict.get(train_tag):
            
            self.cls.eval()
            with torch.no_grad():
                tag_vec=self.cls(train_feature_vec)
            self.tag_dict[train_tag]={
                "tag_vec": tag_vec.detach().reshape(-1),
                "time": 1
            }

            train_generator(self.tag_dict[train_tag]["tag_vec"], train_feature_vec.detach().reshape(-1))

        # 如果tag_dict有该tag
        else:
            
            # -------------------------------------
            # 先用Generator预测生成train_tag之外的其他tag所对应的feature_vec，作为训练Classifier的陪练（以应对Catastrophic Forgetting）

            # 堆叠其他tag的tag_vec
            other_tag_vecs=[]
            for tag, value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict), self.TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
                if tag!=train_tag:
                    other_tag_vecs.append(value["tag_vec"])
            
            # 如果没有其他陪练的
            if other_tag_vecs==[]:
                self.cls.train()
                output_tag_vec=self.cls(train_feature_vec)
                
                original_tag_vec=self.tag_dict[train_tag]["tag_vec"].reshape(1,-1)
                
                # 取递减偏离的一点
                target_tag_vec = original_tag_vec + calc_target_offset(original_tag_vec, output_tag_vec.detach(), self.tag_dict[train_tag]["time"])

                self.tag_dict[train_tag]["tag_vec"]=target_tag_vec.reshape(-1)
                self.tag_dict[train_tag]["time"]+=1

                self.cls_optimizer.zero_grad()
                loss=self.cls_criterion(output_tag_vec, target_tag_vec)
                self.loss_dict["Classifier Continual Single Train Loss"].append(loss.tolist())
                loss.backward()
                self.cls_optimizer.step()

                train_generator(self.tag_dict[train_tag]["tag_vec"], train_feature_vec.detach().reshape(-1))
            
            # 如果有其他tag的tag_vec
            else:
                
                other_tag_vecs=torch.stack(other_tag_vecs)

                # 用generator生成陪练feature_vecs
                self.gen.eval()
                with torch.no_grad():
                    other_feature_vecs=self.gen(other_tag_vecs)
                cat_feature_vecs = torch.cat([other_feature_vecs, train_feature_vec])
                
                # 开始训练classifier
                self.cls.train()
                
                try:
                    output_tag_vecs=self.cls(cat_feature_vecs)
                except:
                    print("ERROR occur in .line1")
                    print(train_text, train_tag)
                    return
                
                original_tag_vec=self.tag_dict[train_tag]["tag_vec"].reshape(1,-1)
            
                # 取original_tag_vec与TARGET_tag_VEC的连线上的一点为优化目标
                # 取递减偏离的一点
                target_tag_vec = original_tag_vec + calc_target_offset(original_tag_vec, output_tag_vecs[-1].detach().reshape(1,-1), self.tag_dict[train_tag]["time"])

                self.tag_dict[train_tag]["tag_vec"]=target_tag_vec.reshape(-1)
                self.tag_dict[train_tag]["time"]+=1

                # 把所有的tag domain的vec都弄起来
                # 因为在classifier中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
                target_tag_vecs=torch.cat([ other_tag_vecs, target_tag_vec ])
                
                self.cls_optimizer.zero_grad()
                loss=self.cls_criterion(output_tag_vecs, target_tag_vecs)
                self.loss_dict["Classifier Continual Attach Train Loss"].append(loss.tolist())
                loss.backward()
                self.cls_optimizer.step()

                train_generator(self.tag_dict[train_tag]["tag_vec"], train_feature_vec.detach().reshape(-1))
    
    def continual_forward_without_generator(self, train_text, train_tag):
        # "单样本正向训练，不带生成向量的陪练"

        train_feature_vec=self.lm(train_text).detach() # 这里就detach掉，BERT是不用训练的，LSTM的话怎么弄

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if not self.tag_dict.get(train_tag):
            
            self.cls.eval()
            with torch.no_grad():
                tag_vec=self.cls(train_feature_vec)
            self.tag_dict[train_tag]={
                "tag_vec": tag_vec.detach().reshape(-1),
                "time": 1
            }

        else:
            self.cls.train()
            output_tag_vec=self.cls(train_feature_vec)
            
            original_tag_vec=self.tag_dict[train_tag]["tag_vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_tag_vec = original_tag_vec + calc_target_offset(original_tag_vec, output_tag_vec.detach(), self.tag_dict[train_tag]["time"])
            
            self.tag_dict[train_tag]["tag_vec"]=target_tag_vec.reshape(-1)
            self.tag_dict[train_tag]["time"]+=1

            self.cls_optimizer.zero_grad()
            loss=self.cls_criterion(output_tag_vec, target_tag_vec)
            self.loss_dict["Classifier Continual Baseline Train Loss"].append(loss.tolist())
            loss.backward()
            self.cls_optimizer.step()
    
    def continual_backward(self, train_text, train_tag):
        
        # 这里和forward的训练方法大致一样，区别是：target_tag_vecs是反方向的，并且不包括对generator的训练

        # -------------------------------------
        # 先用generator预测生成train_tag之外的其他tag的所对应的embedding作为训练classifier的陪练
        # 以应对Catastrophic Forgetting
        train_feature_vec=self.lm(train_text).detach() # 这里就detach掉，BERT是不用训练的，LSTM的话怎么弄

        # 堆叠其他tag的vec
        other_tag_vecs=[]
        for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict), self.TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
            if tag!=train_tag:
                other_tag_vecs.append(value["tag_vec"])
        
        if other_tag_vecs==[]:
            self.cls.train()
            output_tag_vec=self.cls(train_feature_vec)
            
            original_tag_vec=self.tag_dict[train_tag]["tag_vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_tag_vec = original_tag_vec + calc_target_offset(original_tag_vec, output_tag_vec.detach(), self.tag_dict[train_tag]["time"], forward=False)
            
            self.tag_dict[train_tag]["tag_vec"]=target_tag_vec.reshape(-1)
            
            self.cls_optimizer.zero_grad()
            loss=self.cls_criterion(output_tag_vec, target_tag_vec)
            loss.backward()
            self.cls_optimizer.step()
        else:
            
            other_tag_vecs=torch.stack(other_tag_vecs)

            # 用generator生成陪练feature_vecs
            self.gen.eval()
            with torch.no_grad():
                other_feature_vecs=self.gen(other_tag_vecs)
            cat_feature_vecs = torch.cat([other_feature_vecs, train_feature_vec])

            # 开始训练classifier
            self.cls.train()
            
            try:
                output_tag_vecs=self.cls(cat_feature_vecs)
            except:
                print("ERROR occur in .line2")
                print(train_text, train_tag)
                return
            
            original_tag_vec=self.tag_dict[train_tag]["tag_vec"].reshape(1,-1)
            
            # 有进有退
            offset = calc_target_offset(original_tag_vec, output_tag_vecs[-1].detach().reshape(1,-1), self.tag_dict[train_tag]["time"])
            target_tag_vecs = torch.cat( [other_tag_vecs + offset, original_tag_vec - offset] )
            
            self.tag_dict[train_tag]["tag_vec"]=(original_tag_vec - offset).reshape(-1)
            
            self.cls_optimizer.zero_grad()
            loss=self.cls_criterion(output_tag_vecs, target_tag_vecs)
            loss.backward()
            self.cls_optimizer.step()
        
        if self.tag_dict[train_tag]["time"]>1:
            self.tag_dict[train_tag]["time"]-=1
    
    def batch_train(self, train_pipe, batch_size=16):
        # 遍历一遍训练集中所有的tag，如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        temp_train_dict={}
        for i in train_pipe:
            text=i[0]
            tag=i[1]
            if temp_train_dict.get(tag)==None:
                temp_train_dict[tag]=[text]
            else:
                temp_train_dict[tag].append(text)

        self.cls.eval()
        with torch.no_grad():
            for key, value in temp_train_dict.items():
                if self.tag_dict.get(key)==None:
                    feature_text=random.sample(value, 1)[0]
                    feature_vec=self.lm(feature_text).detach()
                    tag_vec=self.cls(feature_vec)
                    self.tag_dict[key]={
                        "tag_vec": tag_vec.detach().reshape(-1),
                        "time": 1
                    }

        # 开始以batch_size为一个batch，对全体训练集进行小批量训练
        self.cls.train()
        o=0
        while o<len(train_pipe):

            feature_vecs=[]
            target_tag_vecs=[]
            for i in train_pipe[o:o+batch_size]:
                text=i[0]
                tag=i[1]
                feature_vec=self.lm(text).detach().reshape(-1)
                feature_vecs.append(feature_vec)
                target_tag_vecs.append(self.tag_dict[tag]["tag_vec"])
            feature_vecs=torch.stack(feature_vecs)
            target_tag_vecs=torch.stack(target_tag_vecs)

            output_tag_vecs=self.cls(feature_vecs)

            self.cls_optimizer.zero_grad()
            loss=self.cls_criterion(target_tag_vecs, output_tag_vecs)
            self.loss_dict["Classifier Batch Train Loss"].append(loss.tolist())
            # print(loss)
            loss.backward()
            self.cls_optimizer.step()

            o+=batch_size