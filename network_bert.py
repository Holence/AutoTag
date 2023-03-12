from utils import *
import torch
from transformers import BertTokenizer, BertModel

TRAIN_ALONG=10

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(768, 384)   # 768 -> 384

    def forward(self, sentences_vectors):
        out = self.fc(sentences_vectors)
        return out

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = torch.nn.Linear(384, 768)   # 384 -> 768

    def forward(self, sentences_vectors):
        out = self.fc(sentences_vectors)
        return out

class Model():

    CLA_LR=0.0005
    GEN_LR=0.05

    def __init__(self) -> None:

        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:",self.device)

        # Bert
        if not os.path.exists("bert_para/pytorch_model.bin"):
            self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.tokenizer.save_pretrained("./bert_para/")
            # 保存初始状态
            self.bert.save_pretrained("./bert_para/")
        else:
            self.tokenizer = BertTokenizer.from_pretrained("./bert_para/")
            self.bert = BertModel.from_pretrained("./bert_para/")
            self.bert.to(self.device)
        
        # 固定Bert
        for p in self.bert.parameters():
            p.requires_grad=False
        
        # classifier
        self.classifier=Classifier()
        try:
            self.classifier.load_state_dict(torch.load("bert_para/classifier.pt"))
        except:
            init_network(self.classifier)
        self.classifier.to(self.device)

        # 学习率也应该保存并读取
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.CLA_LR)
        self.classifier_criterion = torch.nn.MSELoss(reduction = 'sum')
        
        # generator
        self.generator=Generator()
        try:
            self.generator.load_state_dict(torch.load("bert_para/generator.pt"))
        except:
            init_network(self.generator)
        self.generator.to(self.device)

        # 学习率也应该保存并读取
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.GEN_LR)
        self.generator_criterion = torch.nn.MSELoss(reduction = 'sum')
        
        if os.path.exists("bert_para/tag_dict.pt"):
            self.tag_dict=torch.load("bert_para/tag_dict.pt")
        else:
            self.tag_dict={}
        
        self.sentence_vec_dict={}
        
        self.loadCorpus()

        self.loss_dict={
            "Classifier Batch Train Loss":[],
            "Classifier Continual Single Train Loss":[],
            "Classifier Continual Attach Train Loss":[],
            "Classifier Continual Baseline Train Loss":[],
            "Generator Continual Single Train Loss":[],
            "Generator Continual Attach Train Loss":[],
        }
    
    def loadCorpus(self):
        self.train_pipe, self.train_dict, self.test_dict=load_corpus("corpus", 0.5)

    def save(self):
        torch.save(self.classifier.state_dict(), "bert_para/classifier_para.pt")
        torch.save(self.generator.state_dict(), "bert_para/generator_para.pt")
        torch.save(self.tag_dict, "bert_para/tag_dict.pt")
    
    def sentences_vectors(self, sentences):
        inputs=self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        inputs.to(self.device)
        output=self.bert(**inputs)
        return output["pooler_output"]

    def forward(self, current_text, current_tag):
        # "单样本正向训练，生成向量进行陪练"
        
        def train_generator():
            # 堆叠其他tag的sentences_vec
            select=[]
            for tag in random.sample(list(self.tag_dict.keys()), min(len(self.tag_dict), TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
                select.append(tag)
            
            tag_vecs=[]
            sentences_vec=[]
            for tag in select:
                    tag_vecs.append(self.tag_dict[tag]["vec"])
                    sentences_vec.append(self.sentence_vec_dict[tag])
            
            tag_vecs=torch.stack(tag_vecs)
            sentences_vec=torch.stack(sentences_vec)
            
            self.generator.train()
            
            output_seq=self.generator(tag_vecs)
            self.generator.zero_grad()
            loss=self.generator_criterion(output_seq, sentences_vec)
            loss.backward()
            self.generator_optimizer.step()
            return loss.tolist()

        current_text=pre_process(current_text)
        sentences_vec=self.sentences_vectors([current_text])

        orig_vec=self.sentence_vec_dict.get(current_tag, None)
        new_vec=sentences_vec[-1,...]
        if orig_vec==None:
            self.sentence_vec_dict[current_tag]=new_vec
        else:
            self.sentence_vec_dict[current_tag] = orig_vec + calc_target_offset(orig_vec, new_vec, self.tag_dict[current_tag]["time"])

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if type(self.tag_dict.get(current_tag))==type(None):
            
            self.classifier.eval()
            with torch.no_grad():
                classes_vecs=self.classifier(sentences_vec)
            self.tag_dict[current_tag]={
                "vec":classes_vecs.detach()[0,...],
                "time":1
            }
            
            # -------------------------------------
            loss=train_generator()
            self.loss_dict["Generator Continual Single Train Loss"].append(loss)

        else:
            
            # -------------------------------------
            # 先用generator预测生成current_tag之外的其他tag的所对应的embedding作为训练classifier的陪练
            # 以应对Catastrophic Forgetting

            # 堆叠其他tag的vec
            other_tag_vecs=[]
            for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict),TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
                if tag!=current_tag:
                    other_tag_vecs.append(value["vec"])
            
            # 如果没有其他陪练的，就取classifier的输出为TARGET_VEC，取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
            if other_tag_vecs==[]:
                self.classifier.train()
                output_tag_vec=self.classifier(sentences_vec)
                
                orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
                
                # 取递减偏离的一点
                target_vec = orig_tag_vec + calc_target_offset(orig_tag_vec, output_tag_vec.detach(), self.tag_dict[current_tag]["time"])

                self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
                self.tag_dict[current_tag]["time"]+=1

                self.classifier.zero_grad()
                loss=self.classifier_criterion(output_tag_vec, target_vec)
                self.loss_dict["Classifier Continual Single Train Loss"].append(loss.tolist())
                loss.backward()
                self.classifier_optimizer.step()

                # -------------------------------------
                loss=train_generator()
                self.loss_dict["Generator Continual Single Train Loss"].append(loss)
            else:
                
                other_tag_vecs=torch.stack(other_tag_vecs)

                # 用generator生成陪练sentences_vecs
                self.generator.eval()
                with torch.no_grad():
                    other_sentences_vecs=self.generator(other_tag_vecs)

                cat_sentences_vecs = torch.cat([other_sentences_vecs, sentences_vec])
                
                # -------------------------------------
                # 开始训练classifier

                self.classifier.train()
                
                try:
                    output_tag_vecs=self.classifier(cat_sentences_vecs)
                except:
                    print("ERROR occur in .line1")
                    print(current_text, current_tag)
                    return
                
                orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
                # 取orig_tag_vec与TARGET_VEC的连线上的一点为优化目标
                # 取递减偏离的一点
                target_vec = orig_tag_vec + calc_target_offset(orig_tag_vec, output_tag_vecs[-1,...].detach().reshape(1,-1), self.tag_dict[current_tag]["time"])

                # 把所有的tag domain的vec都弄起来
                # 因为在classifier中current text的embedding放在了最后一个，这里也把current tag的vec放在最后一个
                target_vecs=torch.cat([ other_tag_vecs, target_vec ])
                
                self.tag_dict[current_tag]["vec"]=target_vec.reshape(-1)
                self.tag_dict[current_tag]["time"]+=1
                
                self.classifier.zero_grad()
                loss=self.classifier_criterion(output_tag_vecs, target_vecs)
                self.loss_dict["Classifier Continual Attach Train Loss"].append(loss.tolist())
                loss.backward()
                self.classifier_optimizer.step()

                # -------------------------------------
                loss=train_generator()
                self.loss_dict["Generator Continual Attach Train Loss"].append(loss)

    def forward_without_generator(self, current_text, current_tag):
        # "单样本正向训练，不带生成向量的陪练"

        current_text=pre_process(current_text)
        sentences_vec=self.sentences_vectors([current_text])

        # 如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        if type(self.tag_dict.get(current_tag))==type(None):
            
            self.classifier.eval()
            with torch.no_grad():
                classes_vecs=self.classifier(sentences_vec)
            self.tag_dict[current_tag]={
                "vec":classes_vecs.detach()[0,...],
                "time":1
            }

        else:
            
            self.classifier.train()
            output_tag_vec=self.classifier(sentences_vec)
            
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_vec = orig_tag_vec + calc_target_offset(orig_tag_vec, output_tag_vec.detach(), self.tag_dict[current_tag]["time"])
            
            self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
            self.tag_dict[current_tag]["time"]+=1

            self.classifier.zero_grad()
            loss=self.classifier_criterion(output_tag_vec, target_vec)
            self.loss_dict["Classifier Continual Baseline Train Loss"].append(loss.tolist())
            loss.backward()
            self.classifier_optimizer.step()
    
    def backward(self, current_text, current_tag):
        
        # 这里和forward的训练方法大致一样，区别是：target_vecs是反方向的，并且不包括对generator的训练

        # -------------------------------------
        # 先用generator预测生成current_tag之外的其他tag的所对应的embedding作为训练classifier的陪练
        # 以应对Catastrophic Forgetting
        current_text=pre_process(current_text)
        sentences_vec=self.sentences_vectors([current_text])

        # 堆叠其他tag的vec
        other_tag_vecs=[]
        for tag,value in random.sample(list(self.tag_dict.items()), min(len(self.tag_dict),TRAIN_ALONG) ): # 随机取TRAIN_ALONG个来伴随训练，太多了的话内存不够用
            if tag!=current_tag:
                other_tag_vecs.append(value["vec"])
        
        if other_tag_vecs==[]:
            self.classifier.train()
            output_tag_vec=self.classifier(sentences_vec)
            
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            # 取递减偏离的一点
            target_vec = orig_tag_vec - calc_target_offset(orig_tag_vec, output_tag_vec.detach(), self.tag_dict[current_tag]["time"])
            
            self.tag_dict[current_tag]["vec"]=target_vec[-1,...]
            if self.tag_dict[current_tag]["time"]!=1:
                self.tag_dict[current_tag]["time"]-=0.5

            self.classifier.zero_grad()
            loss=self.classifier_criterion(output_tag_vec, target_vec)
            loss.backward()
            self.classifier_optimizer.step()
        else:
            
            other_tag_vecs=torch.stack(other_tag_vecs)

            # 用generator生成陪练sentences_vecs
            self.generator.eval()
            with torch.no_grad():
                other_sentences_vecs=self.generator(other_tag_vecs)

            cat_sentences_vecs = torch.cat([other_sentences_vecs, sentences_vec])

            # -------------------------------------
            # 开始训练classifier
            
            self.classifier.train()
            
            try:
                output_tag_vecs=self.classifier(cat_sentences_vecs)
            except:
                print("ERROR occur in .line2")
                print(current_text, current_tag)
                return
            
            orig_tag_vec=self.tag_dict[current_tag]["vec"].reshape(1,-1)
            
            offset = calc_target_offset(orig_tag_vec, output_tag_vecs[-1,...].detach().reshape(1,-1), self.tag_dict[current_tag]["time"])
            
            # 有进有退
            target_vecs = torch.cat( [other_tag_vecs + offset, orig_tag_vec - offset] )
            
            self.tag_dict[current_tag]["vec"]=(orig_tag_vec - offset).reshape(-1)
            
            if self.tag_dict[current_tag]["time"]!=1:
                self.tag_dict[current_tag]["time"]-=0.5
            
            self.classifier.zero_grad()
            loss=self.classifier_criterion(output_tag_vecs, target_vecs)
            loss.backward()
            self.classifier_optimizer.step()

    def batch_train(self):
        
        # 遍历一遍训练集中所有的tag，如果tag_dict中没有该tag，就用网络的输出作为该tag的初始tag_vec
        self.classifier.eval()
        with torch.no_grad():
            for key, value in self.train_dict.items():
                if self.tag_dict.get(key)==None:
                    sample_text=value[random.randint(0,len(value)-1)]
                    sample_text=pre_process(sample_text)
                    sentences_vec=self.sentences_vectors([sample_text])

                    classes_vecs=self.classifier(sentences_vec)
                    self.tag_dict[key]={
                        "vec":classes_vecs.detach()[0,...],
                        "time":1
                    }

        # 开始以batch_size为一个batch，对全体训练集进行小批量训练
        self.classifier.train()
        batch_size=16
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

            sentences_vecs=self.sentences_vectors(batch_texts)
            classes_vecs=self.classifier(sentences_vecs)

            self.classifier_optimizer.zero_grad()
            loss=self.classifier_criterion(target_vec, classes_vecs)
            self.loss_dict["Classifier Batch Train Loss"].append(loss.tolist())
            print(loss)
            loss.backward()
            self.classifier_optimizer.step()

            o+=batch_size

    def predict(self, text, top_num):
        
        text=pre_process(text)
        if text:
            self.classifier.eval()
            result={}
            with torch.no_grad():
                inputs=self.sentences_vectors([text])
                sentences_vec=self.classifier(inputs)

                for tag,value in self.tag_dict.items():
                    # 计算距离
                    vec=value["vec"]
                    result[tag]=torch.linalg.norm(vec-sentences_vec).tolist()
                result=sorted(result.items(),key=lambda x:x[1])
    
            if top_num==-1:
                return result
            else:
                return [i[0] for i in result[:top_num]]