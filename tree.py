from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
from pyecharts.charts import Line
import pyecharts.options as opts

class Tree():
    def __init__(self):
        self.tree = DecisionTreeClassifier()

    def fit(self, train_data, train_target):
        self.tree.fit(train_data, train_target)

    def predict(self, test_data):
        # tree = joblib.load("model/tree_8w.pkl")
        tree_result = self.tree.predict(test_data)
        return tree_result

    def predict_proba(self, test_data):
        tree_result_pro = self.tree.predict_proba(test_data)
        return tree_result_pro

    def choose(self, train_data, train_target, test_data, test_target):
        accuracy_list = []
        name = [str(i) for i in range(2,21)]
        max_accuracy = 0
        for i in range(2,21):
            print("正在测试：",str(i))
            self.tree = DecisionTreeClassifier(max_depth=26, min_samples_split=i)#
            self.fit(train_data, train_target)
            result = self.predict(test_data)
            accuracy = self.Accuracy(test_target, result)
            accuracy_list.append(accuracy)
            if accuracy>max_accuracy:
                joblib.dump(self.tree, "model/tree_8w.pkl")

        self.draw(name, accuracy_list, 'min_samples_split')


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
        a.set_global_opts(title_opts=opts.TitleOpts(title=name),
        yaxis_opts=opts.AxisOpts(
            min_=99,
        ),)
        a.render(name+'.html')

        