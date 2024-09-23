"""
这里的代码，主要是进行特征选择方法的一个整合，把多种的特征选择方法进行整合（spearman、互信息法、熵值法）
每一个特征选择方法选择的特征，进行归一化，然后进行累加。
最后会生成一个特征排序的csv文件
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression # 互信息法
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

name = 'DO'

"""spearman特征提取,不需要删除标签值,返回归一化之后的数据"""
def spearman_fe(data):

    spearman_1 = data.corr(method='spearman')
    re = spearman_1[name]
    # 对数据取绝对值
    re_abs = re.abs()
    # 删除最后一行
    re_abs = re_abs.drop(re_abs.index[-1])
    # 进行归一化
    spearman_re = re_abs / re_abs.sum()
    spearman_re = pd.DataFrame(spearman_re)
    # 转化为pandas类似的数据
    spearman_re = np.array(spearman_re)
    print("spearman特征提取")
    #print(spearman_re)
    return spearman_re

"""熵值法进行特征提取，需要删除带有标签值，最后计算的值 满足归一化 不用再进行归一化。"""
def Entro_fe(data):

    # 提取第一列到倒数第二列，[0:-1]表示从0到倒数第二列，不包含倒数第一列
    Poverty_Alleviation_Satisfaction = data[data.columns[0:-1]]

    # 1.归一化，算K值
    row_no = Poverty_Alleviation_Satisfaction.shape[0]  # 样本，表示多少行，就是有多少个样本
    col_no = Poverty_Alleviation_Satisfaction.shape[1]  # 指标，表示多少列，应该是多少个特征的意思

    # lambda 表示一个匿名函数 冒号左边为参数，冒号右边为返回值
    """
    lambda x:x/sum(x) 函数的作用是：
      每一列中的数值➗这一列数值的总数，类似归一化，为什么选择列，
      因为这在对特征进行选择，一个特征就是表示列，所以进行列的选择。
    """
    Normalization_data = Poverty_Alleviation_Satisfaction.apply(lambda x: x / sum(x))

    # 这个K表示什么意思
    K = 1 / np.log(row_no)  # 1186个样本

    # 2.计算Ej，每个属性的贡献度
    # np.log() 表示以e为底
    P_matrix = Normalization_data.apply(lambda x: x * np.log(x+1e-15))
    Ej = -K * P_matrix.apply(lambda x: x.sum())
    # 3.求Dj
    Dj = 1 - Ej
    # 4.求Wj
    # Wj表示 对每一个特征进行权重打分，
    Wj = Dj / sum(Dj)
    # 转化为pandas类型的数据
    Wj = pd.DataFrame(Wj)
    re = np.array(Wj)
    print("熵值法特征提取")
    #print(re)
    return re

"""互信息法进行特征提取，需要分类标签和数据集，相当于训练之后然后再进行特征评价 """
def Mu_fe(data):
    # 去除标签值
    x = data.iloc[:, 0:-1]
    # 获取标签
    y = data[name]
    # 应该是一种训练，和随机森林类似
    h = mutual_info_regression(x, y)
    new = pd.DataFrame(h)
    # 对数据取绝对值
    new_2 = new.abs()
    # 进行归一化
    new_3 = new_2 / new_2.sum()
    re = np.array(new_3)
    print("互信息法特征提取！")
    #print(re)
    return re


# r表示不需要进行转义,返回上一层用..\。两个点
test_data = pd.read_csv(r"..\数据集\6组数据(不带NA)\%s.csv" %name)
# Saving feature names for later use 保存特征名为以后使用
feature_list = list(test_data.iloc[:, 0:-1].columns)



fe1 = Mu_fe(test_data)
fe2 = Entro_fe(test_data)
fe3 = spearman_fe(test_data)


fe_re = fe1+fe2+fe3
# 给特征重要值打分添加索引
fe_re = pd.DataFrame(fe_re, index=feature_list)
# 添加一个列名
fe_re.columns = ['Importance']
# 进行一个降序排序。
indices = fe_re.sort_values('Importance', ascending = False)


indices.to_csv("result.csv")



""" 接下来获取前两百个特征，然后进行独立性校验 """
new_data = pd.DataFrame()
# 获取重要性排序的OTU
feature_list = list(indices.index)

for i in range(100):
    temp = test_data[feature_list[i]]
    # 两个pandas数据进行合并
    new_data = pd.concat([new_data, temp], axis = 1)
    print("插入%d列成功" %i)

# 需要把标签加入进来吗？
# 完成了对所有的特征进行特征提取，取了200个，然后需要对200个特征进行独立性检测
threshold = 0.6 # 设定相关性阈值
cor = new_data.corr(method='pearson') # 相关系数矩阵



# 基于相关系数矩阵选择相关性小于阈值的特征
temp = 0
selected_features = list(new_data.columns);
for i in range(len(cor.columns)-1):

    for j in range(i+1, len(cor.columns)): # 这个循环中列数永远小于行数，上三角形
        if cor.index[i] not in selected_features:
            break
        if abs(cor.iloc[i, j]) >= 0.6:
            # 如果两个特征相关性大于等于0.8，则选择第一个特征
            if cor.columns[j] in selected_features:
                temp = temp + 1
                print("删除", cor.columns[j])
                selected_features.remove(cor.columns[j])

# 上面的循环完成减少相关性大于0.6的两个特征。然后再进行绘制相关性热力图
re_data = pd.DataFrame()
# 生成新的数据集
for otu in selected_features:
    re_data = pd.concat([re_data, new_data[otu]], axis = 1)



# 相关系数矩阵
re_cor = re_data.corr(method='pearson')
ax = plt.subplots(figsize=(20, 16))#调整画布大小
ax = sns.heatmap(re_cor, # 系数矩阵
                 vmin=0, vmax=1, #设置数值的最大值和最小值
                 square=True, # 每个方形都为正方形
                 annot=True, # 显示相关系数的数据
                 cmap='coolwarm_r',
                 fmt='.2f',  # 只显示两位小数
                 annot_kws={'size': 2} # 相关系数的值的大小
                 )

# 设置边距和宽高比例
plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
plt.gcf().set_size_inches(10, 10)
plt.gca().set_aspect('equal', adjustable='box')
# 设置刻度字体大小
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
re_cor.abs().to_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\headmap\%s.csv" %name)
plt.savefig(r"D:\WorkSpace\jupyter_notebook\Data\数据集\headmap\%s.png" %name, dpi=1000,  bbox_inches='tight')#保存图片，分辨率为600
plt.show()

li = test_data.iloc[:, -1]
re_data = pd.concat([re_data, li], axis = 1)
# 不带索引
re_data.to_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据_特征处理完成\%s.csv" %name, index=False)
print("保存完成！")












