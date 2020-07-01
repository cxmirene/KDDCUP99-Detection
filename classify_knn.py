import numpy as np
from pyecharts.charts import Line
import pyecharts.options as opts

class Knn():
    def __init__(self, k=3, category=5):
        self.k = k
        self.category = category

    def fit(self, train_x, train_y):
        self.x = train_x
        self.y = train_y

    def predict(self, test_x, cv=False):
        result = []
        result_pro = []
        i=1
        for x in test_x:
            distance = []       # 存距离
            for train_x in self.x:
                dis = np.linalg.norm(x-train_x)         # 欧氏距离
                distance.append(dis)                    # 将距离存入列表中

            sort_dis = sorted(range(len(distance)), key=lambda k: distance[k])  # 排序，得到的结果是对应的索引列表
            label = np.zeros(self.category)                         # 类别出现次数
            for k in range(self.k):                     # 前k个
                label[self.y[sort_dis[k]]]+=1           # 对应类别索引加一
            maxindex = np.argmax(label)                 # 最多的那个类别
            result.append(maxindex)                     # 结果
            label = [l/self.k for l in label]
            result_pro.append(label)
            print("K："+str(self.k)+" cv: "+str(cv)+" "+str(i)+" 已处理，剩余"+str(len(test_x)-i)+" 未处理")
            i+=1
        result = np.array(result)
        return result, result_pro

    def choose_K(self, cv, train_data, train_target):
        train_num = int(len(train_data)/cv)
        accuracy_list = []
        k_list = []
        for k in range(3, 16, 2):
            self.k = k                              # 设定K值
            k_list.append(str(k))
            accuracy = 0
            for i in range(cv):

                # 创建出新的测试集和训练集及其对应类别
                if i==cv-1:
                    new_test_data = train_data[i*train_num:]
                    new_test_target = train_target[i*train_num:]
                    new_train_data = train_data[0:i*train_num-1]
                    new_train_target = train_target[0:i*train_num]
                else:
                    new_test_data = train_data[i*train_num:(i+1)*train_num]
                    new_test_target = train_target[i*train_num:(i+1)*train_num]
                    new_train_data = np.vstack((train_data[0:i*train_num], train_data[(i+1)*train_num+1:]))
                    new_train_target = np.hstack((train_target[0:i*train_num], train_target[(i+1)*train_num+1:]))

                self.fit(new_train_data, new_train_target)      # 使用knn算法并放入训练集及对应类别
                result,_ = self.predict(new_test_data, cv)      # 对测试集进行预测
                accuracy += self.Accuracy(new_test_target, result)      # 计算正确率
            accuracy = round(accuracy/cv,2)             # 平均正确率
            accuracy_list.append(accuracy)
            print(str(k)+" 值测试完毕")
        print(accuracy_list)
        print(k_list)
        self.draw(k_list, accuracy_list, '正确率')
        sort_acc = sorted(range(len(accuracy_list)), key=lambda k: accuracy_list[k], reverse=True)      # 对每一个k值测出来的平均正确率进行排序
        return 3+sort_acc[0]*2          # 选取正确率最高的那一个对应的k值

    def Accuracy(self, target, test):
        result = (target==test)
        right = np.sum(result==True)
        accuracy = round(100*(right/len(target)),2)
        return accuracy

    def draw(self, x, y, name):
        a = Line()
        a.add_xaxis(x)
        a.add_yaxis(name,
                    y,
                    markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]),
        )
        a.set_global_opts(title_opts=opts.TitleOpts(title=name))
        a.render(name+'.html')