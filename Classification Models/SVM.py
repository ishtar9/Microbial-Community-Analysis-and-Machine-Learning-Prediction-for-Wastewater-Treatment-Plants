"""
使用SVM进行分类预测，然后输出最终准确率
"""
import pandas as pd  # 导入pandas

import numpy as np

import matplotlib

import xgboost as xgb
from sklearn.metrics import accuracy_score

# Matplotlib 有多个 GUI 后端可用,这里设置Qt5Agg后端
matplotlib.use('Qt5Agg')

print("开始读入数据")
# 读入文件
path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据(处理完异常值，不带NA)\continent_num.csv"
data = pd.read_csv(path)


label_1 = data['continent_num']
# 转化成 one-hot编码，把非数字的值转化为数字，即是one-hot编码
#label_2 = pd.get_dummies(label_1)

# 删除标签
data = data.drop('continent_num', axis = 1)
print("删除标签列完成==============================================")
feature_list = list(data.columns)
#转换成数组
data_array = np.array(data)
from sklearn.model_selection import train_test_split, cross_val_score
# Split the data into training and testing sets
# 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了

from sklearn.utils import shuffle

# 打乱数据和标签的顺序
shuffled_data, shuffled_labels = shuffle(data_array, label_1, random_state=42)


from sklearn.svm import SVC

svm = SVC()

# 进行十倍交叉验证
scores = cross_val_score(svm, shuffled_data, shuffled_labels, cv=10)

# 输出每次交叉验证的准确率
for i, score in enumerate(scores):
    print(f"Cross Validation {i+1} Accuracy: {score}")

# 输出平均准确率
print("Average Accuracy:", scores.mean())





