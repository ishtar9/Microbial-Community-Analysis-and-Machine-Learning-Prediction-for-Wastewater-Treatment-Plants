"""
进行XGBoost分类模型,得出最终的结果，并输出准确率
"""
# 导入库

import pandas as pd  # 导入pandas

import numpy as np

import matplotlib


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
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



# 假设X是特征数据，y是标签数据
# 数据集划分为特征和标签
X = data_array
y = label_1
# 初始化十倍交叉验证对象
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 6,  # 类别数
    'eta': 0.1,  # 学习率
    'max_depth': 3  # 树的最大深度
}

accuracy_scores = []

for train_index, val_index in kfold.split(X, y):
    # 划分训练集和验证集
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 将数据转换为DMatrix对象
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 训练模型
    model = xgb.train(params, dtrain)

    # 在验证集上进行预测
    y_pred = model.predict(dval)

    # 计算准确率并保存
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
    accuracy_scores.append(accuracy)

# 计算平均准确率
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print("平均准确率:", mean_accuracy)




















