"""
用来合并5个特征csv文件，对共有的OTU_ID进行合并，不共有的用0填充
这里设置了读取前10个最重要的OTU进行合并，合并之后大概就只有50个以下。
"""

import pandas as pd



# 定义合并函数
def merge_files(file_paths):
    # 设置读取前多少行，这里是读取前十行
    row_nums = 5
    # 读取第一个文件
    path = "D:\\WorkSpace\\jupyter_notebook\\Data\\数据集\\特征重要性\\"
    # 设置只读取前10个最重要的OTU进行合并
    merged = pd.read_csv(path+file_paths[0], index_col=0, nrows=row_nums)

    # 循环读取并合并剩余的文件
    for file_path in file_paths[1:]:
        df = pd.read_csv(path+file_path, index_col=0, nrows=row_nums)
        merged = merged.join(df, how="outer").fillna(0)

    # 重新排列列的顺序，将DO和SRT列放在最后一列
    merged = merged[[col for col in merged.columns if col not in ["DO", "SRT", "Temp", "MLSS", "pH"]] + ["DO", "SRT", "Temp", "MLSS", "pH"]]

    # 将索引恢复为一列
    merged.reset_index(inplace=True)

    return merged


# 调用合并函数对5个文件进行合并
file_paths = ["DO.csv", "SRT.csv", "MLSS.csv", "pH.csv", "Temp.csv"]
merged = merge_files(file_paths)

# 将结果保存到文件
merged.to_csv(r"D:\WorkSpace\jupyter_notebook\Data\数据集\特征重要性\merged_TOP5.csv", index=False)
print("保存完成")
