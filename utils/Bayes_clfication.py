"""
进行贝叶斯分类模型，然后获取最终的特征图,分类完成之后，然后获取混淆矩阵，然后进行绘制混淆矩阵
"""
# 导入库
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
from sklearn.ensemble import RandomForestClassifier
# Matplotlib 有多个 GUI 后端可用,这里设置Qt5Agg后端
matplotlib.use('Qt5Agg')

print("开始读入数据")
# 读入文件
path = r"D:\WorkSpace\jupyter_notebook\Data\数据集\6组数据(处理完异常值，不带NA)\continent_num.csv"
data = pd.read_csv(path)


label_1 = data['continent_num']
# 转化成 one-hot编码，把非数字的值转化为数字，即是one-hot编码
#label_2 = pd.get_dummies(label_1)

# 删除标签
data = data.drop('continent_num', axis = 1)
print("删除标签列完成==============================================")
feature_list = list(data.columns)
#转换成数组
data_array = np.array(data)
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
# 这里划分数据集的时候，两个输入和标签 都应该是数组，在前面的代码 已经把features和labels 转换成数组了
train_features, test_features, train_labels, test_labels = train_test_split(data_array, label_1, test_size = 0.25, random_state = 18, stratify=label_1 )
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

from sklearn.naive_bayes import GaussianNB
print("开始训练！")
print(test_labels)
clf = GaussianNB()
rf = RandomForestClassifier()
rf.fit(train_features, train_labels)
pre = rf.predict(test_features)
print(pre)
sc = rf.score(test_features, test_labels)
cm = confusion_matrix(test_labels, pre)
print(sc)
print("混淆矩阵为：", cm)
# 类别
classes = ['Africa', 'Asia', 'Australasia', 'Europe', 'North America', 'South America']


proportion = []
length = len(cm)
print(length)
for i in cm:
    for j in i:
        temp = j / (np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
# print(proportion)
pshow = []
for i in proportion:
    pt = "%.2f%%" % (i * 100)
    pshow.append(pt)
proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
pshow = np.array(pshow).reshape(length, length)
# print(pshow)
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
# plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=8, rotation = 45)
plt.yticks(tick_marks, classes, fontsize=8)

thresh = cm.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (cm.size, 2))
for i, j in iters:
    if (i == j):
        plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=10, color='white',
                 weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=5, color='white')
    else:
        plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=5)

plt.ylabel('True label', fontsize=10)
plt.xlabel('Predict label', fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig('混淆矩阵.png', dpi = 1000)