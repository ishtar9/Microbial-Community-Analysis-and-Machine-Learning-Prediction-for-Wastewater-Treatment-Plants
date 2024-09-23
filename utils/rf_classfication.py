"""
进行随机森林分类模型，然后获取最终的特征图
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
path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据_特征处理完成\continent_num.csv"
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
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
# 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了
train_features, test_features, train_labels, test_labels = train_test_split(data_array,label_2,test_size = 0.25,random_state = 18,stratify=label_2 )
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

from sklearn.ensemble import RandomForestClassifier
#================================================================
#使用最佳参数进行训练模型，然后再用测试集来进行测试最终效果。
rf = RandomForestClassifier(n_estimators = 500)
print("开始训练")
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
score = rf.score(test_features, test_labels)
print(score)


Graph_X_OTU = []
Graph_Y_importance = []
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(train_features.shape[1]):

    # 选取前20个重要的特征
    if f < 30:
        Graph_X_OTU.append(feature_list[indices[f]])
        Graph_Y_importance.append(importances[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, feature_list[indices[f]], importances[indices[f]]))

name = 'continent'
importance_data = pd.DataFrame({'OTU_ID': Graph_X_OTU, name: Graph_Y_importance})
importance_data.to_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\特征重要性\%s.csv" % name, index=False)
print("保存完成！")