import pandas as pd
import numpy as np
import torch
from time import time, strftime, localtime, clock
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import joblib

from classify_knn import Knn
from bayes import Gaussian_Bayes
from BP import Net
from tree import Tree
from pyecharts.charts import Line
from pyecharts.charts import Bar
import pyecharts.options as opts
import matplotlib.pyplot as plt

class Classify():
    def __init__(self, category=5):
        self.datafile = 'data/kddcup.data_10_percent_corrected_save_8w.csv'
        # self.datafile = 'data/corrected_save2.csv'
        self.category = category

        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus']=False

        self.time=strftime("%Y-%m-%d_%H-%M-%S", localtime()) 

    def train_test(self):
        df_all = pd.read_csv(self.datafile)
        drop_name = [str(i) for i in range(31, 42)]
        y = np.array(df_all['41'])                              # 表示分类的列
        df_all = df_all.drop(drop_name, axis=1)                 # 删除后十个
        x = np.array(df_all)
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(x, y, test_size=0.2, random_state=1)        # 划分数据集

    # 处理为二分类
    def classify_two(self):
        for i in range(len(self.train_target)):
            if self.train_target[i]!=0:
                self.train_target[i]=1          # 不是正常的标签就置为1
        for i in range(len(self.test_target)):
            if self.test_target[i]!=0:
                self.test_target[i]=1          # 不是正常的标签就置为1

    def classify(self):
        model_evaluation = {}
        model_confusion = {}
        model_score = {}
        path = 'model/'+str(self.category)+'_'

        print("真实结果：",self.test_target)
        # knn = Knn()
        # k = knn.choose_K(5, self.train_data, self.train_target)
        # print("最终选择的k为："+str(k))
        # knn = Knn(3)
        # knn.fit(self.train_data, self.train_target)
        # knn_result, knn_result_pro = knn.predict(self.test_data)
        # print("KNN预测结果：",knn_result)
        # knn_evaluation = self.Evaluation(self.test_target, knn_result)
        # model_evaluation['KNN']=knn_evaluation

        # knn = KNeighborsClassifier(n_neighbors=3)
        # knn.fit(self.train_data, self.train_target)
        # joblib.dump(knn, path+"knn_8w.pkl")

        knn = joblib.load(path+"knn_8w.pkl")
        knn_result = knn.predict(self.test_data)
        knn_result_pro = knn.predict_proba(self.test_data)
        print("KNN预测结果：",knn_result)
        knn_evaluation, knn_confu = self.Evaluation(self.test_target, knn_result)
        model_evaluation['KNN']=knn_evaluation
        model_confusion['KNN']=knn_confu
        model_score['KNN']=knn_result_pro

        # bayes = GaussianNB()
        # bayes.fit(self.train_data, self.train_target)
        # joblib.dump(bayes, path+"bayes_8w.pkl")

        bayes = joblib.load(path+"bayes_8w.pkl")
        bayes_result = bayes.predict(self.test_data)
        bayes_result_pro = bayes.predict_proba(self.test_data)
        print("贝叶斯预测结果：",bayes_result)
        bayes_evaluation, bayes_confu = self.Evaluation(self.test_target, bayes_result)
        model_evaluation['贝叶斯']=bayes_evaluation
        model_confusion['贝叶斯']=bayes_confu
        model_score['贝叶斯']=bayes_result_pro

        # bp = Net(self.train_data.shape[1], [20], self.category)
        # bp.train(self.train_data, self.train_target, 10000)
        # torch.save(bp,path+'bp_8w.pth')

        bp = torch.load(path+'bp_8w.pth')
        bp_result, bp_result_pro = bp.test(self.test_data)
        print("BP神经网络预测结果：",bp_result)
        bp_evaluation, bp_confu = self.Evaluation(self.test_target, bp_result)
        model_evaluation['BP神经网络']=bp_evaluation
        model_confusion['BP神经网络']=bp_confu
        model_score['BP神经网络']=bp_result_pro

        # tree = Tree()
        # tree.choose(self.train_data, self.train_target, self.test_data, self.test_target)
        # tree = DecisionTreeClassifier()
        # tree.fit(self.train_data, self.train_target)
        # joblib.dump(tree, path+"tree_8w.pkl")

        tree = joblib.load(path+"tree_8w.pkl")
        tree_result = tree.predict(self.test_data)
        tree_result_pro = tree.predict_proba(self.test_data)
        print("决策树预测结果：",tree_result)
        tree_result = tree_result.astype(int)
        tree_evaluation, tree_confu = self.Evaluation(self.test_target, tree_result)
        model_evaluation['决策树']=tree_evaluation
        model_confusion['决策树']=tree_confu
        model_score['决策树']=tree_result_pro
        

        self.draw_evaluation(model_evaluation)
        self.draw_confusion(model_confusion)
        multi = True
        if self.category==2:
            multi = False
        self.ROC(self.test_target, model_score, multi=multi)
        self.P_R(self.test_target, model_score, multi=multi)

    def Evaluation(self, target, test):
        self.loss = []
        for i in range(self.category):
            if i not in target and i not in test:
                self.loss.append(i)
        result = []
        # 混淆矩阵
        confusion = confusion_matrix(target, test)
        if len(self.loss)!=0:
            for l in self.loss:
                b = np.zeros(confusion.shape[0])
                confusion = np.insert(confusion, l, b, axis=1)
                b = np.zeros(confusion.shape[1])
                confusion = np.insert(confusion, l, b, axis=0)
        # 准确率
        ACC = round(100*(accuracy_score(target, test)),2)
        result.append(ACC)
        # 精确率
        P = round(100*(precision_score(target, test, average="weighted")),2)
        result.append(P)
        # 召回率
        R = round(100*(recall_score(target, test, average="weighted")),2)
        result.append(R)
        # F1-Score
        F = round(100*(f1_score(target, test, average="weighted")),2)
        result.append(F)
        return result, confusion

    def ROC(self, target, model_score, multi=False):
        fig = plt.figure(figsize=(8,8))
        for model, socre in model_score.items():
            if multi:
                new_target = []
                for t in target:
                    y = np.zeros(self.category-len(self.loss))
                    y[t]=1
                    new_target.append(y)
                new_target = np.array(new_target)
                fpr = {}
                tpr = {}
                roc_auc = {}
                for i in range(self.category-len(self.loss)):
                    fpr[i], tpr[i], _ = roc_curve(new_target[:, i], socre[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.category-len(self.loss))]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(self.category-len(self.loss)):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= (self.category-len(self.loss))
                FPR = all_fpr
                TPR = mean_tpr
            else:
                FPR, TPR, _ = roc_curve(target, socre[:,1])
            roc_auc = auc(FPR, TPR)
            plt.plot(FPR, TPR, lw=2, label=model+' - ROC (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', size=20)
            plt.ylabel('True Positive Rate', size=20)
            plt.title('ROC Curve', size=22)
            plt.legend(loc="lower right")
        plt.savefig('result/'+self.time+'ROC.png')
        plt.clf()

    def P_R(self, target, model_score, multi=False):
        fig = plt.figure(figsize=(8,8))
        for model, socre in model_score.items():
            if multi:
                new_target = []
                for t in target:
                    y = np.zeros(self.category-len(self.loss))
                    y[t]=1
                    new_target.append(y)
                new_target = np.array(new_target)
                precision = {}
                recall = {}
                roc_auc = {}
                for i in range(self.category-len(self.loss)):
                    precision[i], recall[i], _ = precision_recall_curve(new_target[:, i], socre[:, i])
                all_precision = np.unique(np.concatenate([precision[i] for i in range(self.category-len(self.loss))]))
                mean_recall = np.zeros_like(all_precision)
                for i in range(self.category-len(self.loss)):
                    mean_recall += np.interp(all_precision, precision[i], recall[i])
                mean_recall /= (self.category-len(self.loss))
                precision = all_precision
                recall = mean_recall
            else:
                precision, recall, _ = precision_recall_curve(target, socre[:,1])
            plt.plot(precision, recall, label=model+' - P_R曲线', lw=2)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('precision', size=20)
            plt.ylabel('recall', size=20)
            plt.title('P_R Curve', size=22)
            plt.legend(loc="lower right")
        plt.savefig('result/'+self.time+'P_R.png')
        plt.clf()

    def draw(self, x, y, name):
        a = Line()
        a.add_xaxis(x)
        a.add_yaxis(name,
                    y,
                    markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]),
        )
        a.set_global_opts(title_opts=opts.TitleOpts(title=name))
        a.render('result/'+name+'.html')

    def draw_evaluation(self, model_evaluation):
        a = Bar()
        a.add_xaxis(['准确率','精确率','召回率','F1-Score'])
        for model, evaluation in model_evaluation.items():
            a.add_yaxis(model,
                        evaluation,
            )
        a.set_global_opts(title_opts=opts.TitleOpts(title='性能评价'))
        a.render('result/'+self.time+'性能评价.html')

    def draw_confusion(self, model_confusion):
        if self.category==5:
            classes = ['normal','dos','probe','R2L','U2R']
        else:
            classes = ['normal','attack']
        num = len(list(model_confusion.keys()))
        fig = plt.figure(figsize=(16,int(16/num)))
        now_i = 1
        for model, confusion in model_confusion.items():
            plt.subplot(1,num,now_i)
            plt.imshow(confusion, cmap=plt.cm.Blues)
            plt.title(model, size=22)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, size=18)
            plt.yticks(tick_marks, classes, size=18)
            
            thresh = confusion.max() / 2.
            iters = np.reshape([[[i,j] for j in range(self.category)] for i in range(self.category)],(confusion.size,2))
            for i, j in iters:
                plt.text(j, i, format(confusion[i, j]), 
                        size=18, va = 'center',ha = 'center',
                        color="white" if confusion[i, j] > thresh else "black")   #显示对应的数字
            plt.ylabel('真实值', size=20)
            plt.xlabel('预测值', size=20)
            now_i+=1

        fig.tight_layout()
        plt.savefig('result/'+self.time+'混淆矩阵.png')
        plt.clf()

start = clock()
classify = Classify(5)
classify.train_test()
if classify.category==2:
    classify.classify_two()
classify.classify()
end = clock()
print("共耗时："+str(round((end-start)/60,3))+" min")