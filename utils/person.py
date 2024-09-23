"""
这段代码主要是用来进行计算person相关系数的一个测试,
计算文件中person相关系数，
通过自己的理解，选择spearman的相关性分析。
进行相关性分析之后，对所有值进行取绝对值操作，然后再进行归一化。
"""


import pandas as pd

# r表示不需要进行转义,返回上一层用..\。两个点
test_data = pd.read_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\插补完数据和otu放到一起\all_data.csv")
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
# 使用spearman进行分析
result_1 = test_data.corr(method='spearman')


label_name = ['TP removal rate', 'NH4-N removal rate', 'BOD removal rate', 'SRT', 'MLSS' ,'DO','pH','Temp',	'HRT in Plant',	'HRT in Aeration tank'
]

re_spearman = pd.DataFrame()



for name in label_name:

    re = result_1[name]
    # 对数据 取绝对值
    re_abs = re.abs()
    # 删除后面11行
    re_abs = re_abs[0:-11]
    # 进行归一化
    re_abs_1 = re_abs/re_abs.sum()

    re_spearman[name] = re_abs_1

re_spearman.to_csv(r"D:\WorkSpace\jupyter_notebook\Data\Code\Feature_Importance\all_spearman.csv")

print("保存完成！")
