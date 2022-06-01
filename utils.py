from sklearn.model_selection import train_test_split
import os
import random
import pickle
import torch
from tqdm import tqdm
import numpy as np
import re

WORD_VEC_RAW_FILE="sgns.zhihu.char"
WORD_ID_DICT_FILE="sgns.zhihu.wi.pkl"
WORD_EMBED_FILE="sgns.zhihu.embed.npz"

def generateWordIdDict():
    """
    {
        "一":0,
        "三十":1,
        "<UNK>":2,
    }
    """
    word_id_dict={}
    
    # '<PAD>' id为0
    word_id_dict.update({"<PAD>": len(word_id_dict)})
    
    i=1
    with open("%s"%WORD_VEC_RAW_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()[1:]):
            word = line.strip().split(" ")[0]
            # 竟然还有重复的……
            if word_id_dict.get(word)==None:
                word_id_dict[word]=i
                i+=1
    
    # 最后加上'<UNK>'
    word_id_dict.update({"<UNK>": len(word_id_dict)})
    
    with open(WORD_ID_DICT_FILE,'wb') as f:
        pickle.dump(word_id_dict,f)

def generateWordVecDict(word_to_id):
    """
    {
        0: np.array([...]),
        1: np.array([...]),
        2: np.array([...])
    }
    """
    embed_dim = 300
    word_embeddings = np.random.rand(len(word_to_id), embed_dim) # 初始随机
    with open("%s"%WORD_VEC_RAW_FILE, 'r', encoding='utf-8') as f:
        
        for line in tqdm(f.readlines()[1:]):
            lin = line.strip().split(" ")

            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1:]]
                word_embeddings[idx] = np.asarray(emb, dtype='float32')
    
    # '<UNK>'和'<PAD>'保留随机值即可
    np.savez_compressed(WORD_EMBED_FILE, embeddings=word_embeddings)

def load_WordIdDict():
    if not os.path.exists(WORD_ID_DICT_FILE):
        print("Generating %s from %s"%(WORD_ID_DICT_FILE,WORD_VEC_RAW_FILE))
        generateWordIdDict()
    else:
        print("Found word_id_dict in %s"%WORD_ID_DICT_FILE)
    
    with open(WORD_ID_DICT_FILE,'rb') as f:
        word_id_dict=pickle.load(f)
    
    return word_id_dict

def load_WordEmbeddings(word_id_dict):
    if not os.path.exists(WORD_EMBED_FILE):
        print("Generating %s from %s"%(WORD_EMBED_FILE,WORD_VEC_RAW_FILE))
        generateWordVecDict(word_id_dict)
    else:
        print("Found word_embeddings in %s"%WORD_EMBED_FILE)
    
    word_embeddings=torch.tensor(np.load(WORD_EMBED_FILE)["embeddings"].astype('float32'))

    return word_embeddings

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        # print(name,w)
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    torch.nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    torch.nn.init.kaiming_normal_(w)
                else:
                    torch.nn.init.normal_(w)
            elif 'bias' in name:
                torch.nn.init.constant_(w, 0)
            else:
                pass


def load_corpus(name, test_size=0.2):

    train_pipe=[]
    test_set={}
    tag_list=[]
    for tag in os.listdir("./%s"%name):
        
        with open("./%s/%s"%(name,tag),"r",encoding='utf-8') as f:
            text=[_.strip() for _ in f.read().split("===") if _.strip()]
            if len(text)>1:
                
                train,test=train_test_split(text,test_size=test_size)
                train_pipe.extend([(_,tag) for _ in train])
                test_set[tag]=test
        
                tag_list.append(tag)

    random.shuffle(train_pipe)
    
    return train_pipe, test_set, tag_list

def pre_process(content):
    def is_chinese(uchar):
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        else:
            return False
    
    # content_str = ''
    # for i in content:
    #     if is_chinese(i):
    #         content_str = content_str + ｉ
    content_str = re.sub(r'[^\w\s]','',content)
    return content_str