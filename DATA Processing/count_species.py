"""
用来统计输入的OTU都是属于什么菌种，统计输入的菌种数量，
为了以后的分析对比
"""

import pandas as pd

# 读取OTU表格
df1 = pd.read_csv(r"OTU_id.csv")

# 读取OTU ID与菌种信息的对应表格
df2 = pd.read_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\菌种信息\OTU菌种信息_两列(细菌下面一级).csv")
# 将第二个文件转换成字典形式，以便于替换,字典形式
tax_dict = dict(zip(df2["OTU_ID"], df2["Tax"]))

# 替换df1中的OTU_ID为Tax
df1["OTU_id"] = df1["OTU_id"].map(tax_dict)

# 按照 OTU_ID 分组并求和
re = df1.groupby("OTU_id").sum()

# 重置索引
re = re.reset_index()

# 将结果保存到文件
re.to_csv(r"输入的菌种数量.csv", index=False)



