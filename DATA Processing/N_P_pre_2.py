"""
这里主要是使用细菌群落对BOD,COD,N,P等去除率的预测,  选择TP、NH4-N、BOD
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
name = 'TP'

path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据(带NA)\TP.csv"
data = pd.read_csv(path)
data.dropna(inplace=True) # 删除空行
label = np.array(data[f'{name} removal rate']) # 转为数组
data = data.drop(f'{name} removal rate', axis=1) # 删除标签
feature_list = list(data.columns)
data_array = np.array(data)

train_features, test_features, train_labels, test_labels = train_test_split(data_array, label, test_size=0.25, random_state=42)

rf = RandomForestRegressor()
print("开始训练")
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
score = rf.score(test_features, test_labels)
print(f"得分为:{score}")


Graph_X_OTU = []
Graph_Y_importance = []
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(train_features.shape[1]):

    # 选取前20个重要的特征
    if f < 5:
        Graph_X_OTU.append(feature_list[indices[f]])
        Graph_Y_importance.append(importances[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, feature_list[indices[f]], importances[indices[f]]))


importance_data = pd.DataFrame({'OTU_ID': Graph_X_OTU, name: Graph_Y_importance})
importance_data.to_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\特征重要性\%s.csv" % name, index=False)
print("保存完成！")




