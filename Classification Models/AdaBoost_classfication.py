"""
使用AdaBoostClassifier进行分类预测，然后输出最终结果
"""
# 导入库
import numpy as np  # numpy库
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import train_test_split
print("开始读入数据")
# 读入文件
path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据(处理完异常值，不带NA)\continent_num.csv"
data = pd.read_csv(path)


label_1 = data['continent_num']
# 转化成 one-hot编码，把非数字的值转化为数字，即是one-hot编码
label_2 = pd.get_dummies(label_1)

# 删除标签
data = data.drop('continent_num', axis = 1)
print("删除标签列完成==============================================")
feature_list = list(data.columns)
#转换成数组
data_array = np.array(data)
# from sklearn.model_selection import train_test_split
# # Split the data into training and testing sets
# # 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了
# train_features, test_features, train_labels, test_labels = train_test_split(data_array,label_2,test_size = 0.2,random_state = 18,stratify=label_2 )
# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
# 打乱数据和标签的顺序
shuffled_data, shuffled_labels = shuffle(data_array, label_1, random_state=42)
from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier()
# 进行十倍交叉验证
scores = cross_val_score(classifier, shuffled_data, shuffled_labels, cv=10)
# 输出每次交叉验证的准确率
for i, score in enumerate(scores):
    print(f"Cross Validation {i+1} Accuracy: {score}")
# 输出平均准确率
print("Average Accuracy:", scores.mean())





























