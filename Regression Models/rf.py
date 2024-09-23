"""
使用随机森林回归进行计算最终的分数，用于方法的比较
"""

from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score



name = "HRT in Aeration tank"

path = r'D:\WorkSpace\jupyter_notebook\Data\数据集\插补完数据和otu放到一起\all_data.csv'
# 读入文件
features = pd.read_csv(path)
# 提取标签
labels = features[name]
# 删除最后十列数据
features = features.drop(features.columns[-10:], axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# 转换成数组
features_array = np.array(features)
labels_array = np.array(labels)
# Split the data into training and testing sets
# 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了
train_features, test_features, train_labels, test_labels = train_test_split(features_array,labels_array, test_size = 0.25,random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print("划分数据集和测试集完成==============================================")

print("开始训练模型！")
# 建立一个随机森林对象
rf = RandomForestRegressor( random_state=42)
# 进行训练
rf.fit(train_features, train_labels)
pre = rf.predict(test_features)


# 这个score就是决定系数
score = rf.score(test_features, test_labels)

print(f"{name}:最终得分为：", score)

r2 = r2_score(test_labels, pre)
print(f"{name}:r2最终得分为：", r2)

# true_pre = pd.DataFrame({'true': test_labels, 'Pre': pre})
# save_path = f"D:\\WorkSpace\\jupyter_notebook\\Data\\数据集\\5组数据(测试集中的预测值与真实值)和散点图\{name}.csv"
# # true_pre.to_csv(save_path, index=False) # 表示不加索引
# true_pre.to_csv(save_path)
# print("保存完成！")

