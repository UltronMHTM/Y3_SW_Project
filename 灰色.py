
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#路径目录
#当前目录
baseDir = 'D:/test/'
#静态文件目录
staticDir = os.path.join(baseDir,'Static')
#结果文件目录
resultDir = os.path.join(baseDir,'Result')

#读取数据
data = pd.read_csv(staticDir+'/1.csv',encoding='gbk')
train = data['收盘价'].values[-15:-10]#训练数据
test = data['收盘价'].values[-10:]#测试数据
data.head()

#GM11模型
def GM11(x,n):
    '''
    灰色预测
    x：序列，numpy对象
    n:需要往后预测的个数
    '''
    x1 = x.cumsum()#一次累加  
    z1 = (x1[:len(x1) - 1] + x1[1:])/2.0#紧邻均值  
    z1 = z1.reshape((len(z1),1))  
    B = np.append(-z1,np.ones_like(z1),axis=1)  
    Y = x[1:].reshape((len(x) - 1,1))
    #a为发展系数 b为灰色作用量
    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)#计算参数  
    result = (x[0]-b/a)*np.exp(-a*(n-1))-(x[0]-b/a)*np.exp(-a*(n-2))  
    S1_2 = x.var()#原序列方差
    e = list()#残差序列
    for index in range(1,x.shape[0]+1):
        predict = (x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2))
        e.append(x[index-1]-predict)
    S2_2 = np.array(e).var()#残差方差
    C = S2_2/S1_2#后验差比
    if C<=0.35:
        assess = '后验差比<=0.35，模型精度等级为好'
    elif C<=0.5:
        assess = '后验差比<=0.5，模型精度等级为合格'
    elif C<=0.65:
        assess = '后验差比<=0.65，模型精度等级为勉强'
    else:
        assess = '后验差比>0.65，模型精度等级为不合格'
    #预测数据
    predict = list()
    for index in range(x.shape[0]+1,x.shape[0]+n+1):
        predict.append((x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2)))
    predict = np.array(predict)
    return {
            'a':{'value':a,'desc':'发展系数'},
            'b':{'value':b,'desc':'灰色作用量'},
            'predict':{'value':result,'desc':'第%d个预测值'%n},
            'C':{'value':C,'desc':assess},
            'predict':{'value':predict,'desc':'往后预测%d个的序列'%(n)},
            }

#GM11动态建模，进行预测
yPre = []
for i in range(test.shape[0]):
    #只预测1个数
    result = GM11(train,1)
    yPre.append(result['predict']['value'][0])
    #更新训练集
    train = train.tolist()[:-1]
    train.append(test[i])
    train = np.array(train).reshape(-1)
#计算MAE
MAE = mean_absolute_error(test,yPre)
#打印模型
print(result['C']['desc'])
print(result['a']['desc'],np.round(result['a']['value'],2))
print(result['b']['desc'],np.round(result['b']['value'],2))