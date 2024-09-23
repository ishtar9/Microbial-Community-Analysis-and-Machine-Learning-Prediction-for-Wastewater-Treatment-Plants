"""
绘制散点图，并将多个散点图组装起来，
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

# 准备数据
all_feature = pd.read_csv(r"D:\WorkSpace\jupyter_notebook\Data\Code\Feature_Importance\all_feature.csv")
all_spearman = pd.read_csv(r"D:\WorkSpace\jupyter_notebook\Data\Code\Feature_Importance\all_spearman.csv")

label_name = ['TP removal rate', 'NH4-N removal rate', 'BOD removal rate', 'SRT', 'MLSS' ,'DO','pH','Temp',	'HRT in Plant',	'HRT in Aeration tank'
]

# 创建一个2行5列的子图布局
fig, axes = plt.subplots(2, 5, figsize=(12, 6))


# 遍历子图并绘制散点图
for i, ax in enumerate(axes.flatten()):

    x = all_feature[label_name[i]]
    y = all_spearman[label_name[i]]

    # 绘制散点图
    ax.scatter(x, y)

    # 添加标题和标签
    ax.set_title(label_name[i])
    ax.set_xlabel('importance')
    ax.set_ylabel('spearman')
    # 添加标识文本
    label = chr(ord('a') + i)  # 使用字母标识，从'a'开始递增
    ax.annotate(label, xy=(-0.55, 1.1), xycoords='axes fraction', fontsize=12, fontweight='bold')

    # 线性拟合
    coefficients = np.polyfit(x, y, 1)
    p = np.poly1d(coefficients)
    y_fit = p(x)

    # 去除右边和上边的边界线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 计算r2值
    r2 = r2_score(y, y_fit)
    # 添加r2值
    ax.annotate(f'r2 = {r2:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)

# # 线性拟合
# coefficients = np.polyfit(x, y, 1)
# p = np.poly1d(coefficients)
# y_fit = p(x)
#
# # 绘制散点图和拟合线
# plt.scatter(x, y)
# plt.plot(x, y_fit, color='red')
#
# # 添加标题和标签
# plt.title('1')
# plt.xlabel('importance')
# plt.ylabel('spearson')
#
# # 去除右边和上边的边界线
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
#
# # 计算r2值
# r2 = r2_score(y, y_fit)
# # 添加r2值
# plt.annotate(f'r2 = {r2:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)

# 调整子图之间的间距
plt.tight_layout()


plt.savefig('权重与spearman的相关性.jpg', dpi=2000)

# 显示图形
plt.show()
