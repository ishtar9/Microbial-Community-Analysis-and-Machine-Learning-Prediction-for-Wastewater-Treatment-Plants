"""
K最近邻算法（KNN）算法补全缺失值的代码
"""

import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd

path = r'D:\WorkSpace\jupyter_notebook\Data\数据集\原始的理化指标和插补之后的数据\原始选择的20个数据.xlsx'

data = pd.read_excel(path)

X = np.array(data)


# 创建包含缺失值的数据集
# X = np.array([[1, 2, np.nan],
#               [3, np.nan, 4],
#               [5, 6, 7],
#               [np.nan, 8, 9]])

# 创建KNNImputer对象
imputer = KNNImputer(n_neighbors=3)

# 使用KNN算法补全缺失值
X_imputed = imputer.fit_transform(X)

# 输出补全后的数据集
print(X_imputed)

re = pd.DataFrame(X_imputed)
# 新建的re的pandas列名和data的列名一样
re.columns = data.columns

save_path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\原始的理化指标和插补之后的数据\插补数据.csv"
# true_pre.to_csv(save_path, index=False) # 表示不加索引
re.to_csv(save_path, index=False)
print("保存完成！")
