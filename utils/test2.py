"""
用来测试分类，就是把所有的类分为两类，然后进行重要性评分，分析那个otu对这个二分类重要性高。
这样可以分析出每个大陆独特性的一个otu也就是独特性的细菌群落。
然后保存特征
然后绘制直方图
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

name = "Australasia"

print("start")
# 读取文件，这里会改变关于谁的二分类文件。
path = f"classficaition_5\continent_{name}.csv"
features = pd.read_csv(path)
# 获取标签
labels = np.array(features['Continent'])
# 删除标签列
features = features.drop('Continent', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# 转换成数组
features_array = np.array(features)


# Split the data into training and testing sets
# 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了
train_features, test_features, train_labels, test_labels = train_test_split(features_array, labels,
                                                                            test_size=0.25, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# 随机森林分类器
rf = RandomForestClassifier()
# 训练
rf.fit(train_features, train_labels)
# 预测
predictions = rf.predict(test_features)
# 得分
score = rf.score(test_features, test_labels)
print("预测的得分为：", score)

# 两个空列表
Graph_X_OTU = []
Graph_Y_importance = []

# 获取重要性排序
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(train_features.shape[1]):
    # 选取前20个重要的特征
    if f < 20:
        Graph_X_OTU.append(feature_list[indices[f]])
        Graph_Y_importance.append(importances[indices[f]])
        # 打印前20个重要的特征
        # print("%2d) %-*s %f" % (f + 1, 30, feature_list[indices[f]], importances[indices[f]]))

# 新建一个pandas数据集
re_data = pd.DataFrame({'OTU_ID': Graph_X_OTU, f'{name}_importance': Graph_Y_importance})
# 插入一列
#re_data = pd.concat([re_data, Graph_X_OTU], axis = 1)
# 插入一列
#re_data = pd.concat([re_data, Graph_Y_importance], axis = 1)
re_data.to_csv(f"D:\WorkSpace\jupyter_notebook\Data\数据集\双分类(亚洲，非亚洲等)特征重要性柱状图\{name}.csv", index=False)

#绘制特征重要性直方图
import matplotlib.pyplot as plt
# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(Graph_Y_importance)))

# Make a bar chart
plt.bar(x_values, Graph_Y_importance, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, Graph_X_OTU, rotation='30')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('DO_Variable'); plt.title('SouthAmerica_Importances');



