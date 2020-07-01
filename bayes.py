import numpy as np
from math import sqrt, exp, pi

class Gaussian_Bayes():
    def __init__(self, category=5):
        self.category = category
        self.prior = {}
        self.foundation = {}

    def fit(self, train_x, train_y):
        labels = {}
        for c in range(self.category):
            labels.setdefault(c, sum(train_y==c))

        # 计算先验概率
        for label, amount in labels.items():
            prior = (amount+1)/(len(train_y)+self.category)
            self.prior.setdefault(label, prior)

        train_x_classify = {}
        for i in range(len(train_y)):
            if train_y[i] not in train_x_classify.keys():
                train_x_classify.setdefault(train_y[i],[])
            train_x_classify[train_y[i]].append(train_x[i])
            
        for c in range(self.category):
            self.foundation.setdefault(c, {})
            x = train_x_classify[c]
            # 均值
            avg = np.mean(x, axis=0)
            self.foundation[c].setdefault('avg', avg)
            # 方差
            var = np.var(x, axis=0)
            self.foundation[c].setdefault('var', var)
            
            
    def predict(self, test_x):
        result = []
        result_pro = []
        for x in test_x:
            label_pro = []
            pro_all=0
            for label, prior in self.prior.items():
                prior = np.log(prior)
                for f in range(len(x)):
                    if self.foundation[label]['var'][f]==0:
                        continue
                    p = (1/sqrt(2*pi*self.foundation[label]['var'][f]))*\
                        exp(-(x[f]-self.foundation[label]['avg'][f])**2/(2*self.foundation[label]['var'][f]))
                    prior += np.log(p)
                label_pro.append(prior)
                pro_all+=prior
            label_pro = [pro/pro_all for pro in label_pro]
            result_pro.append(label_pro)
            result.append(label_pro.index(max(label_pro)))
        return result, result_pro
