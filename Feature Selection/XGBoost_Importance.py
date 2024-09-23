"""
使用XGBoost获取特征重要性，然后进行特征重要性的保存,然后保存每一个OTU对环境参数的特征重要性
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
label_name = ['TP removal rate', 'NH4-N removal rate', 'BOD removal rate', 'SRT', 'MLSS' ,'DO','pH','Temp',	'HRT in Plant',	'HRT in Aeration tank'
]
score_dict = {}
path = r'D:\WorkSpace\jupyter_notebook\Data\数据集\插补完数据和otu放到一起\all_data.csv'

features = pd.read_csv(path)

new_fea = features.drop(features.columns[-10:], axis=1)

# 新建一个pandas数据，用来存储otu的特征
OTU_importance = pd.DataFrame()
OTU_importance['OTU_id'] = new_fea.columns

for name in label_name:
    # 获取标签
    label = features[name]
    # 转换成数组
    features_array = np.array(new_fea)
    labels_array = np.array(label)
    # 保存列名,为以后的后续使用
    train_features, test_features, train_labels, test_labels = train_test_split(features_array, labels_array,
                                                                                test_size=0.1, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    print("划分数据集和测试集完成==============================================")

    # 定义XGBoost回归模型
    model = xgb.XGBRegressor()
    # 进行训练
    model.fit(train_features, train_labels)
    # 获取特征重要性
    importance = model.feature_importances_
    OTU_importance[name] = importance
    # 打印每个特征及其重要性得分
    for feature, score in zip(new_fea.columns, importance):
        print(feature, score)

# 不添加索引
OTU_importance.to_csv("all_feature.csv", index=False)

print(score_dict)