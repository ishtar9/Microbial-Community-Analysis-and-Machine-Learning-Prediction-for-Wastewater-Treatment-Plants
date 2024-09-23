"""
使用随机森林进行预测,然后把重要性进行排序，这里主要是回归预测相关的
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
name = 'DO'
data = pd.read_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据_特征处理完成\%s.csv" %name)
labels = np.array(data[name]) # 获取标签
data = data.drop(name, axis=1) # 删除标签
data_array = np.array(data) # 转化为数据形式
# 划分训练和测试集
train_x, test_x, train_labels, test_labels = train_test_split(data_array, labels, test_size=0.25, random_state = 42)
print('Training Features Shape:', train_x.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_x.shape)
print('Testing Labels Shape:', test_labels.shape)
# 随机森林模型，默认参数
rf = RandomForestRegressor(n_estimators = 500, random_state=42)
rf.fit(train_x, train_labels)
print("训练完成！")
predictions = rf.predict(test_x)

import matplotlib.pyplot as plt

# 假设 y_test 为测试集的真实值，y_pred 为预测值
plt.scatter(test_labels, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()



score = rf.score(test_x, test_labels)
print('随机森林模型得分:', score)
ResidualSquare = (predictions - test_labels)**2 #计算残差平方
RSS = sum(ResidualSquare) #残差平方和
MSE = np.mean(ResidualSquare) # 计算均方差
print(f'均方误差(MSE)={MSE}')
print(f'残差平方和(RSS)={RSS}')

# 用来保存特征
Graph_X_OTU = [] # OTU_ID
Graph_Y_importance = [] # OTU特征的重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1] # 对特征进行排序，每一个特征值对应一个序号

# 特征的重要性进行评估
Graph_X_OTU = []
Graph_Y_importance = []
# OTU的一些ID
feature_list = list(data.columns)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(train_x.shape[1]):
    Graph_X_OTU.append(feature_list[indices[f]])
    Graph_Y_importance.append(importances[indices[f]])


importance_data = pd.DataFrame({'OTU_ID': Graph_X_OTU, name: Graph_Y_importance})

importance_data.to_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\特征重要性\%s.csv"%name, index=False)

print("保存完成！")