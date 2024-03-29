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
        self.lm_model="BERT"
        self.cl_method="Store"
        self.seed=114514
    
    def initialize(self):
        set_seed(self.seed)
        self.loadCorpus()
        
        if self.lm_model=="LSTM":
            from LSTM import LSTM, Classifier, Generator
            LanguageModel = LSTM
        elif self.lm_model=="BERT":
            from Bert import Bert, Classifier, Generator
            LanguageModel = Bert
        else:
            raise("Wrong lm_model")

        if self.cl_method=="Store":
            from ContinualLearning import ContinualLearningModel_Store
            self.CL_Model=ContinualLearningModel_Store(LanguageModel, Classifier, 0.0005, 10)
        elif self.cl_method=="Generate":
            from ContinualLearning import ContinualLearningModel_Generate
            self.CL_Model=ContinualLearningModel_Generate(LanguageModel, Classifier, 0.0005, 10, Generator, 0.0005)
        else:
            raise("Wrong cl_method")

        self.app.setApplicationName(f"{self.app.applicationName()} - {LanguageModel.__name__} - {self.CL_Model.__class__.__name__}")

        self.acc_dict={}
        for tag in self.train_dict:
            if self.acc_dict.get(tag)==None:
                self.acc_dict[tag+"_test"]=[]
                self.acc_dict[tag+"_train"]=[]
        
        super().initialize()
    
    def loadCorpus(self):
        self.train_pipe, self.train_dict, self.test_dict=load_corpus("corpus", 0.2)

    def setSeed(self, seed):
        self.seed=seed
    
    def setModel(self, lm_model: str, cl_method: str):
        """设置选用的模型

        lm_model (str): 
            1) "LSTM"
            2) "BERT"
        cl_method (str): 
            1) "Store"
            2) "Generate"
        """
        self.lm_model=lm_model
        self.cl_method=cl_method
    
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
        
        self.module.pushButton_reset.clicked.connect(self.reset)
        self.module.pushButton_load_corpus.clicked.connect(self.loadCorpus)
        self.module.pushButton_batch_train.clicked.connect(self.batch_train)
        self.module.pushButton_continual_train_basic.clicked.connect(self.continual_train_basic)
        self.module.pushButton_continual_train.clicked.connect(self.continual_train)
        self.module.pushButton_plot.clicked.connect(self.plot)
        self.module.pushButton_plot2D.clicked.connect(self.plot2D)
        self.module.pushButton_eval.clicked.connect(self.evaluate)
        self.module.pushButton_save.clicked.connect(self.saveModel)

        self.module.actionPredict.triggered.connect(self.predict)
        self.addAction(self.module.actionPredict)
    
    def initializeMenu(self):
        self.addActionToMainMenu(self.module.actionPredict)
        super().initializeMenu()
    
    def saveModel(self):
        self.CL_Model.save()
    
    def reset(self):
        self.CL_Model.initilize()
    
    def checkModel(self):
        if len(self.CL_Model.tag_dict)==0:
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
        self.module.label_acc.setText("Train Average Acc: %.2f%%\nTest Average Acc: %.2f%%"%(train_acc*100, test_acc*100))
    
    def plot(self):
        plt.close('all')
        plt.ion()

        plt.figure()
        keys=[]
        for key in self.CL_Model.loss_dict.keys():
            if self.CL_Model.loss_dict[key]:
                t=torch.tensor(self.CL_Model.loss_dict[key])
                plt.plot(t[:,0], t[:,1])
                keys.append(key)
        plt.legend(keys)

        plt.figure()
        for key in [i for i in self.acc_dict.keys() if "test" in i]:
            plt.plot(self.acc_dict[key])
        plt.legend([i for i in self.acc_dict.keys() if "test" in i])
        
        plt.figure()
        for key in [i for i in self.acc_dict.keys() if "train" in i]:
            plt.plot(self.acc_dict[key])
        plt.legend([i for i in self.acc_dict.keys() if "train" in i])

    def plot2D(self):
        def euclidean(x0, x1):
            x0, x1 = np.array(x0), np.array(x1)
            d = np.sum((x0 - x1)**2)**0.5
            return d
        
        def scaledown(X, distance=euclidean, rate=0.1, iter=1000, rand_time=10, verbose=1):
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
                for m in range(iter):

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
        
        X=np.stack([v["tag_vec"].cpu() for v in self.CL_Model.tag_dict.values()]) 
        label = [k for k in self.CL_Model.tag_dict.keys()]

        loc = scaledown(X, iter=30000, rand_time=650, verbose=1)
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
        
        self.CL_Model.continual_forward(current_text, current_tag)
        
        if update_predict:
            self.predict(None)
    
    def backward(self):
        current_text=self.module.plainTextEdit_train_text.toPlainText()
        current_tag=self.module.lineEdit_train_tag.text()
        if type(self.CL_Model.tag_dict.get(current_tag))==type(None):
            DTFrame.DTMessageBox(self,"Warning","%s is not in tag_dict, please forward train first!"%current_tag)
            return
            
        self.CL_Model.continual_backward(current_text, current_tag)
        
        self.predict(None)
    
    def continual_train_basic(self):
        set_seed(self.seed)
        pipe = random.sample(self.train_pipe, len(self.train_pipe))
        for i in tqdm(pipe):
            self.CL_Model.continual_forward_baseline(i[0],i[1])
        self.plot()

    def continual_train(self):
        set_seed(self.seed)
        pipe = random.sample(self.train_pipe, len(self.train_pipe))
        for i in tqdm(pipe):
            self.forward(i[0],i[1])
        self.plot()

    def batch_train(self):
        set_seed(self.seed)
        batch_size=self.module.spinBox_batchsize.value()
        pipe = random.sample(self.train_pipe, len(self.train_pipe))
        for i in tqdm(range(0, len(self.train_pipe)+batch_size, batch_size)):
            if pipe[i:i+batch_size]:
                self.CL_Model.batch_train(pipe[i:i+batch_size])
        self.plot()

    def predict(self, text):

        if not self.checkModel():
            return
        
        if type(text)==str:
            return self.CL_Model.predict([text], self.module.spinBox_top.value())
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
                    
                    result=self.CL_Model.predict([text], -1)[0]
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
        acc_list=[]
        for key, value in tqdm(self.train_dict.items()):
            correct=0
            pred_tags_list=self.CL_Model.predict(value, self.module.spinBox_top.value())
            for pred_tags in pred_tags_list:
                if key in pred_tags:
                    correct+=1
            acc=correct/len(value)
            self.acc_dict[key+"_train"].append(acc)
            acc_list.append(acc)
            s+="Train Prediction for %10s ---- %.2f%%\n"%(key, acc*100)

        average_acc=sum(acc_list)/len(acc_list)
        if return_string_and_acc==True:
            return s, average_acc
        else:
            print(s, average_acc)
    
    def predict_testset(self, return_string_and_acc=False):
        if not self.checkModel():
            return
        
        s=""
        acc_list=[]
        for key, value in tqdm(self.test_dict.items()):
            correct=0
            pred_tags_list=self.CL_Model.predict(value, self.module.spinBox_top.value())
            for pred_tags in pred_tags_list:
                if key in pred_tags:
                    correct+=1
            acc=correct/len(value)
            self.acc_dict[key+"_test"].append(acc)
            acc_list.append(acc)
            s+="Test Prediction for %10s ---- %.2f%%\n"%(key, acc*100)

        average_acc=sum(acc_list)/len(acc_list)
        if return_string_and_acc==True:
            return s, average_acc
        else:
            print(s, average_acc)
