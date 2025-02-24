# 数据预处理
# 步骤1：初步数据查看
import pandas as pd

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 进行查看
df = pd.read_csv("pre.csv")
print(df.head())


# 分析各个特征与患病的关系
def feature_to_plot(column):
    s0 = df[column][df["Class/ASD"] == "YES"].value_counts()
    s1 = df[column][df["Class/ASD"] == "NO"].value_counts()

    # 画图
    df1 = pd.DataFrame({u"患病": s0, "不患病": s1})
    df1.plot(kind="bar")
    plt.title("{}对于患病的分析".format(column))
    plt.xlabel(column)
    plt.ylabel("人数")
    plt.tight_layout()
    plt.show()


feature_columns = ["age", "gender", 'ethnicity',
                   'jundice', 'austim', 'contry_of_res',
                   'used_app_before', 'result',
                   'age_desc', 'relation']

for i in feature_columns:
    feature_to_plot(i)

# 字段的标签化
df["gender"] = df["gender"].map({"f": 0, "m": 1})
df["jundice"] = df["jundice"].map({"no": 0, "yes": 1})
df["austim"] = df["austim"].map({"no": 0, "yes": 1})
df["used_app_before"] = df["used_app_before"].map({"no": 0, "yes": 1})


# 对特征多的字段定义函数进行标签化
def fn_ethnicity(x):
    if x == "Black" or x == "Hispanic" or x == "White-European":
        return 0
    elif x == "Latio" or x == "Others":
        return 1
    else:
        return 2


df["ethnicity"] = df["ethnicity"].map(fn_ethnicity)


def fn_contry_of_res(x):
    if x in ['Australia', 'Canada',
             'United States', 'United Kingdom', 'France', 'Brazil']:
        return 0
    else:
        return 1


df["contry_of_res"] = df["contry_of_res"].map(fn_contry_of_res)


# 字段relation分为三个数值，按照信息比例进行划分
def fn_relation(x):
    if x == "Self":
        return 0
    elif x == "Parent" or x == "Relative":
        return 1
    else:
        return 2


df["relation"] = df["relation"].map(fn_relation)

# 步骤5：对其他字段进行处理
del df["result"]
del df["age_desc"]

df["Class/ASD"] = df["Class/ASD"].map({"YES": 1, "NO": 0})
df.to_csv("pre1.csv", index=None)
