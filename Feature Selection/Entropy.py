"""
使用熵值法进行对每一个特征进行打分
"""

import pandas as pd
import numpy as np
import math
from numpy import array

# 读取文件
# test2.csv这个数据集是没有标签，就是去除多零之后的特征
data = pd.read_csv(r'..\数据集\6组数据(处理完异常值，不带NA)\Temp.csv')
# 提取第一行到倒数第二行，[0:-1]表示从0到倒数第二列，不包含倒数第一列
Poverty_Alleviation_Satisfaction = data[data.columns[0:-1]]

# 1.归一化，算K值
row_no = Poverty_Alleviation_Satisfaction.shape[0] # 样本，表示多少行，就是有多少个样本
col_no = Poverty_Alleviation_Satisfaction.shape[1] # 指标，表示多少列，应该是多少个特征的意思

# lambda 表示一个匿名函数 冒号左边为参数，冒号右边为返回值
"""
lambda x:x/sum(x) 函数的作用是：
  每一列中的数值➗这一列数值的总数，类似归一化，为什么选择列，
  因为这在对特征进行选择，一个特征就是表示列，所以进行列的选择。
"""
Normalization_data = Poverty_Alleviation_Satisfaction.apply(lambda x: x / sum(x))

# 这个K表示什么意思
K = 1/np.log(row_no) # 1186个样本

# 2.计算Ej，每个属性的贡献度
# np.log() 表示以e为底
P_matrix = Normalization_data.apply(lambda x: x*np.log(x))
Ej = -K*P_matrix.apply(lambda x: x.sum())
# 3.求Dj
Dj = 1-Ej
# 4.求Wj
# Wj表示 对每一个特征进行权重的打分，
Wj = Dj/sum(Dj)

Wj.to_csv("entropy.csv")
print("保存完成！")

