import torch
from utils import *
from typing import Type

class BaseModel():
    def __init__(self, FeatureExtractionModel: Type[torch.nn.Module], Classifier: Type[torch.nn.Module], CLS_LR: float) -> None:
        self.FeatureExtractionModel=FeatureExtractionModel
        self.Classifier=Classifier
        self.CLS_LR=CLS_LR
        self.initilize()
    
    def initilize(self):
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:",self.device)

        self.save_path=self.FeatureExtractionModel.__name__+"_Para"

        self.fxm=self.FeatureExtractionModel()
        self.fxm.to(self.device)

        # Classifier
        self.cls=self.Classifier()
        self.cls.to(self.device)
        # 学习率也应该保存并读取
        self.cls_optimizer = torch.optim.Adam(self.cls.parameters(), lr=self.CLS_LR)
        self.cls_criterion = torch.nn.MSELoss()

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
    
        self.iteration=0
    
    def save(self):
        torch.save(self.cls.state_dict(), f"{self.save_path}/classifier_para.pt")
        torch.save(self.tag_dict, f"{self.save_path}/tag_dict.pt")
        torch.save(self.loss_dict, f"{self.save_path}/loss_dict.pt")

    def continual_forward_baseline(self, train_text, train_tag):
        "单样本，正向训练，不带任何持续学习优化算法，作为效果比较的底线"
        self.iteration+=1

        self.fxm.train()
        train_feature_vec=self.fxm(train_text)

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if not self.tag_dict.get(train_tag):
            
            self.cls.eval()
            with torch.no_grad():
                tag_vec=self.cls(train_feature_vec)
            self.tag_dict[train_tag]={
                "tag_vec": tag_vec.detach().reshape(-1),
                "time": 1
            }
            if self.__class__.__name__=="ContinualLearningModel_Store":
                self.tag_dict[train_tag]["feature_vec"]=train_feature_vec.detach().reshape(-1)

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
            self.loss_dict["Classifier Continual Baseline Train Loss"].append((self.iteration, loss.tolist()))
            loss.backward()
            self.cls_optimizer.step()

            if self.__class__.__name__=="ContinualLearningModel_Store":
                # 更新feature_vec
                # 实验表明feature_vec用calc_target_offset更新后的效果更好
                # self.tag_dict[train_tag]["feature_vec"]=train_feature_vec.detach().reshape(-1)
                original_feature_vec=self.tag_dict[train_tag]["feature_vec"].reshape(1,-1)
                new_feature_vec = original_feature_vec + calc_target_offset(original_feature_vec, train_feature_vec.detach(), self.tag_dict[train_tag]["time"])
                self.tag_dict[train_tag]["feature_vec"]=new_feature_vec.reshape(-1)
    
    def batch_train(self, train_pipe):
        "批量，正向训练，作为效果比较的顶线"
        self.iteration+=1
        
        # 遍历一遍训练集中所有的tag，如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        need_text_tag=[]
        for tag in set([i[1] for i in train_pipe]):
            if self.tag_dict.get(tag)==None:
                need_text_tag.append(tag)
        if need_text_tag:
            temp_train_dict={}
            for i in train_pipe:
                text=i[0]
                tag=i[1]
                if tag in need_text_tag:
                    if temp_train_dict.get(tag)==None:
                        temp_train_dict[tag]=[text]
                    else:
                        temp_train_dict[tag].append(text)
            self.cls.eval()
            self.fxm.eval()
            with torch.no_grad():
                for key, text_list in temp_train_dict.items():
                    feature_vecs=torch.stack([self.fxm(text).detach() for text in text_list])
                    tag_vecs=self.cls(feature_vecs)
                    self.tag_dict[key]={
                        "tag_vec": tag_vecs.mean(dim=0).reshape(-1),
                        "time": 1
                    }
                    if self.__class__.__name__=="ContinualLearningModel_Store":
                        self.tag_dict[key]["feature_vec"]=feature_vecs.mean(dim=0).reshape(-1)
        
        index_dict={}
        for i in range(len(train_pipe)):
            tag=train_pipe[i][1]
            if index_dict.get(tag)==None:
                self.tag_dict[tag]["time"]+=1
                index_dict[tag]=[i]
            else:
                index_dict[tag].append(i)

        self.cls.train()
        self.fxm.train()
        train_feature_vecs=[]
        original_tag_vecs=[]
        for i in train_pipe:
            text=i[0]
            tag=i[1]
            feature_vec=self.fxm(text).reshape(-1)
            train_feature_vecs.append(feature_vec)
            original_tag_vecs.append(self.tag_dict[tag]["tag_vec"])
        train_feature_vecs=torch.stack(train_feature_vecs)
        original_tag_vecs=torch.stack(original_tag_vecs)

        output_tag_vecs=self.cls(train_feature_vecs)
        
        for tag, indexs in index_dict.items():
            new_tag_vec = self.tag_dict[tag]["tag_vec"] + calc_target_offset(self.tag_dict[tag]["tag_vec"], output_tag_vecs[indexs].detach().mean(dim=0), self.tag_dict[tag]["time"])
            original_tag_vecs[indexs] = new_tag_vec
            self.tag_dict[tag]["tag_vec"] = new_tag_vec

            if self.__class__.__name__=="ContinualLearningModel_Store":
                original_feature_vec = self.tag_dict[tag]["feature_vec"]
                new_feature_vec = original_feature_vec + calc_target_offset(original_feature_vec, train_feature_vecs[indexs].detach().mean(dim=0), self.tag_dict[tag]["time"])
                self.tag_dict[tag]["feature_vec"] = new_feature_vec

        self.cls_optimizer.zero_grad()
        loss=self.cls_criterion(output_tag_vecs, original_tag_vecs)
        self.loss_dict["Classifier Batch Train Loss"].append((self.iteration, loss.tolist()))
        loss.backward()
        self.cls_optimizer.step()

    def predict(self, sample_list, top_num=1):
        if sample_list:
            self.cls.eval()
            self.fxm.eval()
            with torch.no_grad():
                feature_vecs=[]
                for i in sample_list:
                    feature_vec=self.fxm(i).reshape(-1)
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

class ContinualLearningModel_Store(BaseModel):
    def __init__(self, FeatureExtractionModel: Type[torch.nn.Module], Classifier: Type[torch.nn.Module], CLS_LR: float, TRAIN_ALONG: int) -> None:
        super().__init__(FeatureExtractionModel, Classifier, CLS_LR)
        self.TRAIN_ALONG=TRAIN_ALONG
    
    def continual_forward(self, train_text, train_tag):
        "单样本，正向训练，储存样本回放"
        self.iteration+=1

        self.fxm.train()
        train_feature_vec=self.fxm(train_text)

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if not self.tag_dict.get(train_tag):
            
            self.cls.eval()
            with torch.no_grad():
                tag_vec=self.cls(train_feature_vec)
            self.tag_dict[train_tag]={
                "tag_vec": tag_vec.detach().reshape(-1),
                "feature_vec": train_feature_vec.detach().reshape(-1),
                "time": 1
            }

        # 如果tag_dict有该tag
        else:

            # 堆叠其他tag的tag_vec
            # 堆叠其他tag的feature_vec
            other_tag_vecs=[]
            other_feature_vecs=[]
            for tag, value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict), self.TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
                if tag!=train_tag:
                    other_tag_vecs.append(value["tag_vec"])
                    other_feature_vecs.append(value["feature_vec"])
            
            # 如果没有其他陪练的
            if other_tag_vecs==[]:
                self.cls.train()
                output_tag_vec=self.cls(train_feature_vec)
                
                # 更新tag_vec
                original_tag_vec=self.tag_dict[train_tag]["tag_vec"].reshape(1,-1)
                target_tag_vec = original_tag_vec + calc_target_offset(original_tag_vec, output_tag_vec.detach(), self.tag_dict[train_tag]["time"])
                self.tag_dict[train_tag]["tag_vec"]=target_tag_vec.reshape(-1)

                # 训练Classifier
                self.cls_optimizer.zero_grad()
                loss=self.cls_criterion(output_tag_vec, target_tag_vec)
                self.loss_dict["Classifier Continual Single Train Loss"].append((self.iteration, loss.tolist()))
                loss.backward()
                self.cls_optimizer.step()
            
            # 如果有其他tag的feature_vec
            else:
                
                other_tag_vecs=torch.stack(other_tag_vecs)
                
                other_feature_vecs=torch.stack(other_feature_vecs)
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
                target_tag_vec = original_tag_vec + calc_target_offset(original_tag_vec, output_tag_vecs[-1].detach().reshape(1,-1), self.tag_dict[train_tag]["time"])
                self.tag_dict[train_tag]["tag_vec"]=target_tag_vec.reshape(-1)

                # 把所有的tag domain的vec都弄起来
                # 因为在classifier中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
                target_tag_vecs=torch.cat([ other_tag_vecs, target_tag_vec ])
                
                # 训练Classifier
                self.cls_optimizer.zero_grad()
                loss=self.cls_criterion(output_tag_vecs, target_tag_vecs)
                self.loss_dict["Classifier Continual Attach Train Loss"].append((self.iteration, loss.tolist()))
                loss.backward()
                self.cls_optimizer.step()

            self.tag_dict[train_tag]["time"]+=1

            # 更新feature_vec
            # 实验表明feature_vec用calc_target_offset更新后的效果更好
            # self.tag_dict[train_tag]["feature_vec"]=train_feature_vec.detach().reshape(-1)
            original_feature_vec=self.tag_dict[train_tag]["feature_vec"].reshape(1,-1)
            new_feature_vec = original_feature_vec + calc_target_offset(original_feature_vec, train_feature_vec.detach(), self.tag_dict[train_tag]["time"])
            self.tag_dict[train_tag]["feature_vec"]=new_feature_vec.reshape(-1)
    
    def continual_backward(self, train_text, train_tag):
        "单样本，反向训练，储存样本回放"
        
        # 这里和forward的训练方法大致一样，区别是：target_tag_vecs是反方向的

        self.fxm.train()
        train_feature_vec=self.fxm(train_text)

        # 堆叠其他tag的tag_vec
        # 堆叠其他tag的feature_vec
        other_tag_vecs=[]
        other_feature_vecs=[]
        for tag, value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict), self.TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
            if tag!=train_tag:
                other_tag_vecs.append(value["tag_vec"])
                other_feature_vecs.append(value["feature_vec"])
        
        # 如果没有其他陪练的
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
        
        # 如果有其他tag的tag_vec
        else:
            
            other_tag_vecs=torch.stack(other_tag_vecs)
                
            other_feature_vecs=torch.stack(other_feature_vecs)
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
    
class ContinualLearningModel_Generate(BaseModel):
    def __init__(self, FeatureExtractionModel: Type[torch.nn.Module], Classifier: Type[torch.nn.Module], CLS_LR: float, TRAIN_ALONG: int, Generator: Type[torch.nn.Module], GEN_LR: float) -> None:
        self.TRAIN_ALONG=TRAIN_ALONG
        self.Generator=Generator
        self.GEN_LR=GEN_LR
        super().__init__(FeatureExtractionModel, Classifier, CLS_LR)
    
    def initilize(self):
        super().initilize()
        # Generator
        self.gen=self.Generator()
        self.gen.to(self.device)
        # 学习率也应该保存并读取
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.GEN_LR)
        self.gen_criterion = torch.nn.MSELoss()
    
    def save(self):
        super().save()
        torch.save(self.gen.state_dict(), f"{self.save_path}/generator_para.pt")
    
    def continual_forward(self, train_text, train_tag):
        "单样本，正向训练，生成旧样本回放"
        self.iteration+=1

        def train_generator(tag_vec, target_feature_vec):
            # 单样本训练Generator
            self.gen.train()
            output_feature_vecs=self.gen(tag_vec)
            self.gen_optimizer.zero_grad()
            loss=self.gen_criterion(output_feature_vecs, target_feature_vec)
            loss.backward()
            self.loss_dict["Generator Continual Single Train Loss"].append((self.iteration, loss.tolist()))
            self.gen_optimizer.step()

        train_feature_vec=self.fxm(train_text)

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
                self.loss_dict["Classifier Continual Single Train Loss"].append((self.iteration, loss.tolist()))
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
                self.loss_dict["Classifier Continual Attach Train Loss"].append((self.iteration, loss.tolist()))
                loss.backward()
                self.cls_optimizer.step()

                train_generator(self.tag_dict[train_tag]["tag_vec"], train_feature_vec.detach().reshape(-1))
    
    def continual_backward(self, train_text, train_tag):
        "单样本，反向训练，生成旧样本回放"
        
        # 这里和forward的训练方法大致一样，区别是：target_tag_vecs是反方向的，并且不包括对generator的训练

        train_feature_vec=self.fxm(train_text)

        # 堆叠其他tag的vec
        other_tag_vecs=[]
        for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict), self.TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
            if tag!=train_tag:
                other_tag_vecs.append(value["tag_vec"])
        
        # 如果没有其他陪练的
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
        # 如果有其他tag的tag_vec
        else:
        # 用generator预测生成train_tag之外的其他tag的所对应的embedding作为训练classifier的陪练
        # 以应对Catastrophic Forgetting
            
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
