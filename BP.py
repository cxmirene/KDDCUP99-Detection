from sklearn import datasets
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
class Net(nn.Module):
    def __init__(self, features, hiddens, output):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(features, hiddens[0]),
            nn.Sigmoid(),
            nn.Linear(hiddens[0], output)
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(params=self.parameters(), lr=0.01)
        self.sch = lr_scheduler.MultiStepLR(self.opt, [8000, 9000], 0.1)

    def forward(self, x):
        x = self.linear(x)
        return x

    def train(self, train_x, train_y, iters):
        train_y = torch.from_numpy(train_y).long()
        input = torch.from_numpy(train_x).float()
        iters_list = []
        loss_list = []
        for i in range(iters):
            self.sch.step()
            iters_list.append(i+1)
            out = self.forward(input)
            loss = self.loss(out, train_y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            print('iter = ',i,' loss = ',str(round(loss.item(),5)),' lr = ',str(self.sch.get_lr()[0]))
            loss_list.append(loss.item())
        import matplotlib.pyplot as plt
        plt.plot(iters_list, loss_list, lw=2)
        plt.xlabel('iter', size=20)
        plt.ylabel('loss', size=20)
        plt.title('Loss Curve', size=22)
        plt.savefig('Loss.png')
        plt.clf()

    def test(self, test_x):
        input = torch.from_numpy(test_x).float()    # numpy转tensor
        out = self.linear(input)                    # 各类别的概率
        prediction = torch.max(out, 1)[1]           # 选择最大的一个作为最终结果,
        result = prediction.numpy()                 # 结果转为numpy
        out = nn.functional.softmax(out)
        result_pro = out.data.numpy()               # 概率结果转numpy
        return result, result_pro

# dataset = datasets.load_iris()
# data = dataset['data']
# iris_type = dataset['target']

# net = Net(4, [20], 3)
# net.train(data, iris_type, 1000)
# # torch.save(net,'model/model.pth')

# # net = torch.load('model/model.pth')
# result, pro = net.test(data)
# right=0
# all=0
# for i in range(len(iris_type)):
#     if result[i]==iris_type[i]:
#         right+=1
#     all+=1
# print("测试：",result)
# print("真实：",iris_type)
# print("正确率：",round(100*right/all, 2))

