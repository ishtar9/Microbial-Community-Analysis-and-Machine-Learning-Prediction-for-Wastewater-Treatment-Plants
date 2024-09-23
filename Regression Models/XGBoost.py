"""
使用XGBoost进行回归预测，使用内核方法也是梯度提升方法，所以这里选择用XGBoost方法，不使用之前写的GradientBoosting_re方法
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

for name in label_name:

    # 用来存储真实值和预测值，最后用来绘制散点图
    true_pre = pd.DataFrame()
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

    # 随机森林训练
    print("开始训练")
    # 定义XGBoost回归模型
    model = xgb.XGBRegressor()

    model.fit(train_features, train_labels)
    # 验证模型
    pre = model.predict(test_features)
    # Train the model on training data

    r2 = r2_score(test_labels, pre)
    print(f"{name}:r2最终得分为：", r2)

    score_dict[name] = r2
    # 可以进行保存测试预测的结果
    # true_pre['true'] = test_labels
    # true_pre['pre'] = pre
    # true_pre.to_csv(f"D:\\WorkSpace\\jupyter_notebook\\Data\\数据集\\5组数据(测试集中的预测值与真实值)和散点图\\{name}.csv")

print(score_dict)
