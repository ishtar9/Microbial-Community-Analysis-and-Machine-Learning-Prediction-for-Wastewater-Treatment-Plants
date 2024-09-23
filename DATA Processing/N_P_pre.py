"""
主要是用来测试：环境因子来预测 BOD,COD,N,P等去除率，探究加入细菌群落是否能够有什么很大的影响
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

names = ['BOD removal rate', 'COD removal rate', 'NH4-N removal rate', 'TN removal rate']
# 含有环境因子、BOD,COD，N.P等去除率
path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\理化性质\预测值.csv"
data = pd.read_csv(path)

for name in names:
    data = data.drop(name, axis=1) # 删除标签

# 删除所有的空行
data.dropna(inplace=True)
# 获取标签
labels = np.array(data['TP removal rate'])

data_array = np.array(data)


train_features, test_features, train_labels, test_labels = train_test_split(data_array, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#================================================================
#使用最佳参数进行训练模型，然后再用测试集来进行测试最终效果。
rf = RandomForestRegressor(n_estimators = 500)
print("开始训练")
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
score = rf.score(test_features, test_labels)
print(score)









