import matplotlib.pyplot as plt
from utils import *
from DTPySide import *

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

from Ui_Window import Ui_Window
class WindowModule(QWidget,Ui_Window):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

class MainWindow(DTSession.DTMainSession):
    
    def __init__(self, app: DTAPP):
        super().__init__(app)
        self.model_name="BERT"
    
    def initialize(self):
        self.loadCorpus()

        from ContinualLearning import ContinualLearningModel
        if self.model_name=="LSTM":
            pass
            # self.app.setApplicationName(self.app.applicationName()+" - LSTM")
        elif self.model_name=="BERT":
            from Bert import Bert, Classifier, Generator
            self.Model=ContinualLearningModel(Bert, Classifier, Generator)
            self.app.setApplicationName(self.app.applicationName()+" - BERT")

        self.acc_dict={}
        for tag in self.train_dict:
            if self.acc_dict.get(tag)==None:
                self.acc_dict[tag+"_test"]=[]
                self.acc_dict[tag+"_train"]=[]
        
        super().initialize()
    
    def loadCorpus(self):
        self.train_pipe, self.train_dict, self.test_dict=load_corpus("corpus", 0.5)

    def setModel(self, model_name):
        """设置选用的语言模型
        1) "LSTM"\n
            train VecNet:\n
                from pretrained Word2Vec get Word Embeddings\n
                put Word Embeddings into LSTM get Sentence Embeddings\n
                put Sentence Embeddings into LSTM get Tag Vector\n
            train GenNet:\n
                put Tag Vector into Linear+LSTM+Linear get Generated Word Embeddings\n
        
        2) "BERT"\n
            train Classifier:\n
                from pretrained BERT get Sentence Embeddings\n
                put Sentence Embeddings into Linear get Tag Vector\n
            train Generator:\n
                put Tag Vector into Linear get Generated Word Embeddings\n
        """
        self.model_name=model_name
    
    def initializeWindow(self):
        super().initializeWindow()
        self.module=WindowModule(self)
        self.setCentralWidget(self.module)
        self.module.plainTextEdit_pred_text.setPlaceholderText("{TAG}lorem ipsum...\n\n===\n\n{标签}测试文本...")

    def initializeSignal(self):
        super().initializeSignal()
        self.module.pushButton_forward.clicked.connect(self.forward)
        self.module.pushButton_backward.clicked.connect(self.backward)
        self.module.pushButton_pred.clicked.connect(self.predict)
        
        self.module.pushButton_load_corpus.clicked.connect(self.loadCorpus)
        self.module.pushButton_continual_train.clicked.connect(self.continual_train)
        self.module.pushButton_continual_train2.clicked.connect(self.continual_train_without_generator)
        self.module.pushButton_batch_train.clicked.connect(self.batch_train)
        self.module.pushButton_plot.clicked.connect(self.plot)
        self.module.pushButton_plot2D.clicked.connect(self.plot2D)
        self.module.pushButton_eval.clicked.connect(self.evaluate)
        self.module.pushButton_save.clicked.connect(self.saveModel)
    
    def saveModel(self):
        self.Model.save()
    
    def checkModel(self):
        if len(self.Model.tag_dict)==0:
            print("There isn't any tag_vec in tag_dict! Please train first!")
            return False
        else:
            return True

    def evaluate(self):
        if not self.checkModel():
            return
        
        self.module.textBrowser_res.clear()
        s, train_acc = self.predict_trainset(return_string_and_acc=True)
        self.module.textBrowser_res.append(s)
        s, test_acc = self.predict_testset(return_string_and_acc=True)
        self.module.textBrowser_res.append(s)
        self.module.label_acc.setText("Train acc: %.2f%%  Test acc: %.2f%%"%(train_acc*100, test_acc*100))
    
    def plot(self):
        plt.close('all')
        plt.ion()

        plt.figure()
        keys1=[]
        for key in self.Model.loss_dict.keys():
            if "Batch" in key and self.Model.loss_dict[key]:
                plt.plot(self.Model.loss_dict[key])
                keys1.append(key)
        plt.legend(keys1)

        plt.figure()
        keys2=[]
        for key in self.Model.loss_dict.keys():
            if "Batch" not in key and self.Model.loss_dict[key]:
                plt.plot(self.Model.loss_dict[key])
                keys2.append(key)
        plt.legend(keys2)

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
        
        if not self.checkModel():
            return
        
        X=np.stack([v["tag_vec"].cpu() for v in self.Model.tag_dict.values()]) 
        label = [k for k in self.Model.tag_dict.keys()]

        loc = scaledown(X, itera=30000, rand_time=650, verbose=1)
        x = loc[:,0]
        y = loc[:,1]

        plt.ion()
        plt.figure()
        plt.scatter(x,y)
        for x_, y_, s in zip(x,y,label):
            plt.annotate(s, (x_, y_))
        plt.show()
    
    def forward(self, current_text=False, current_tag=False):
        if current_text==False and current_tag==False:
            update_predict=True
            current_text=self.module.plainTextEdit_train_text.toPlainText()
            current_tag=self.module.lineEdit_train_tag.text()
        else:
            update_predict=False
        
        self.Model.continual_forward(current_text, current_tag)
        
        if update_predict:
            self.predict(None)
    
    def backward(self):
        
        current_text=self.module.plainTextEdit_train_text.toPlainText()
        current_tag=self.module.lineEdit_train_tag.text()
        if type(self.Model.tag_dict.get(current_tag))==type(None):
            DTFrame.DTMessageBox(self,"Warning","%s is not in tag_dict, please forward train first!"%current_tag)
            return
            
        self.Model.continual_backward(current_text, current_tag)
        
        self.predict(None)

    def continual_train(self):
        random.shuffle(self.train_pipe)
        for i in tqdm(self.train_pipe):
            self.forward(i[0],i[1])
        self.plot()
    
    def continual_train_without_generator(self):
        random.shuffle(self.train_pipe)
        for i in tqdm(self.train_pipe):
            self.Model.continual_forward_without_generator(i[0],i[1])
        self.plot()

    def batch_train(self):
        random.shuffle(self.train_pipe)
        self.Model.batch_train(self.train_pipe)
        self.plot()

    def predict(self, text):

        if not self.checkModel():
            return
        
        if type(text)==str:
            return self.Model.predict(text, self.module.spinBox_top.value())
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
                    
                    original_text=text
                    
                    result=self.Model.predict(text, -1)
                    if result:
                        pred_prob=""
                        for i in result:
                            if i[0]==target_tag:
                                pred_prob+="%.5f  %s    <------\n"%(i[1], i[0])
                            else:
                                pred_prob+="%.5f  %s\n"%(i[1], i[0])
        
                        if target_tag!=None:
                            if target_tag in [i[0] for i in result[:self.module.spinBox_top.value()]]:
                                res+=original_text+"\n\n"+pred_prob+"\n===\n\n"
                            else:
                                # 预测错误
                                res+="~~"+original_text+"~~"+"\n\n"+pred_prob+"\n===\n\n"
                                acc-=single_fault
                        else:
                            # 预测正确
                            res+=original_text+"\n\n"+pred_prob+"\n===\n\n"
                            acc=0
                
                res=res.replace("\n","\n\n")
                self.module.textBrowser_res.setMarkdown(res)
                self.module.textBrowser_res.verticalScrollBar().setValue(scroll)
                self.module.label_acc.setText("acc: %.2f%%"%(acc*100))

    def predict_trainset(self, return_string_and_acc=False):
        if not self.checkModel():
            return
        
        s=""
        total=0
        total_correct=0
        for key, value in self.train_dict.items():
            total+=len(value)
            single_fault=1/len(value)
            acc=1
            for text in value:
                pred_tags=self.predict(text)
                if key in pred_tags:
                    total_correct+=1
                else:
                    acc-=single_fault
            self.acc_dict[key+"_train"].append(acc)
            s+="Train Prediction for %10s ---- %.2f%%\n"%(key, acc*100)

        if return_string_and_acc==True:
            return s, total_correct/total
        else:
            print(s, total_correct/total)
    
    def predict_testset(self, return_string_and_acc=False):
        if not self.checkModel():
            return
        
        s=""
        total=0
        total_correct=0
        for key, value in self.test_dict.items():
            total+=len(value)
            single_fault=1/len(value)
            acc=1
            for text in value:
                pred_tags=self.predict(text)
                if key in pred_tags:
                    total_correct+=1
                else:
                    acc-=single_fault
            self.acc_dict[key+"_test"].append(acc)
            s+="Test Prediction for %10s ---- %.2f%%\n"%(key, acc*100)
        
        if return_string_and_acc==True:
            return s, total_correct/total
        else:
            print(s, total_correct/total)
