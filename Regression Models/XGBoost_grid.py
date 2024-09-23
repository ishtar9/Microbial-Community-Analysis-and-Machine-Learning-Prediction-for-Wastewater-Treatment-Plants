"""
使用XGBoost，然后进行网格搜索得出最优参数
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


label_name = ['TP removal rate', 'NH4-N removal rate', 'BOD removal rate', 'SRT', 'MLSS' ,'DO','pH','Temp',	'HRT in Plant',	'HRT in Aeration tank'
]
name = 'TP removal rate'
score_dict = {}

path = r'D:\WorkSpace\jupyter_notebook\Data\数据集\插补完数据和otu放到一起\all_data.csv'

features = pd.read_csv(path)

new_fea = features.drop(features.columns[-10:], axis=1)

# 获取标签
label = features[name]
# 转换成数组
features_array = np.array(new_fea)
labels_array = np.array(label)
# 保存列名,为以后的后续使用
train_features, test_features, train_labels, test_labels = train_test_split(features_array, labels_array,
                                                                            test_size=0.25, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print("划分数据集和测试集完成==============================================")


# 定义参数空间
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# 创建XGBoost回归模型
xgb_model = xgb.XGBRegressor()

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=4)
grid_search.fit(train_features, train_labels)

# 输出最优参数组合和对应的评估指标
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
