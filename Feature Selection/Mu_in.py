"""
互信息法进行变量特征筛选打分。
"""

from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd


inputfile = r'..\数据集\6组数据(处理完异常值，不带NA)\Temp.csv' #输入的数据文件

data = pd.read_csv(inputfile) #读取数据

x = data.iloc[:, 0:-1]
y = data['Temp']

h = mutual_info_regression(x, y)

print(h)

new = pd.DataFrame(h)
# 对数据取绝对值
new_2 = new.abs()
# 进行归一化
new_3 = new_2/new_2.sum()
new_3.to_csv("info.csv")
print("保存完成！")
