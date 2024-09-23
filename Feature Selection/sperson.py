"""
这段代码主要是用来进行计算person相关系数的一个测试,
计算文件中person相关系数，
通过自己的理解，选择spearman的相关性分析。
进行相关性分析之后，对所有值进行取绝对值操作，然后再进行归一化。


但是需要变量才能进行，特征选择，这样的话，有多少个特征，就需要多少个特征选择。
"""


import pandas as pd

# r表示不需要进行转义,返回上一层用..\。两个点
test_data = pd.read_csv(r"..\数据集\6组数据(处理完异常值，不带NA)\Temp.csv")
"""
# 数据统计,用来统计数据中每一列的一些统计量，比如每一列的列数、平均值、标准差等等。
Desc = test_data.describe()


计算所有列中其他列中的person相关系数，
method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}
Pearson相关系数样本必须是正态分布,衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
spearman：非线性的，非正太分布的数据的相关系数,斯皮尔曼相关系数评估两个连续变量之间的单调关系,斯皮尔曼相关系数
斯皮尔曼相关系数：强调两个变量之间的单调关系，变量增加，因变量也增加，虽然增加的量不一样，spearman相关系数也为1

"""
result_1 = test_data.corr(method='spearman')

re = result_1['Temp']
# 对数据取绝对值
re_abs = re.abs()

# 删除最后一行
re_abs = re_abs.drop(re_abs.index[-1])
# 进行归一化
re_abs_1 = re_abs/re_abs.sum()
re_abs_1.to_csv("Temp_spearson.csv")
print("保存完成！")