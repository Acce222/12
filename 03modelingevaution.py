#第三部分：建模与评估
#对之前处理的数据进行操作，完成建模与评估，查看最终的预测效果

#步骤一：导入相关库
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

#步骤二：数据处理
df = pd.read_csv("pre1.csv")
y = df[["Class/ASD"]]
x = df.drop("Class/ASD", 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=324)

#对字段age特征进行标准化处理
std = StandardScaler()
x_std_train = std.fit_transform(x_train[["age"]])
x_std_test = std.fit_transform(x_test[["age"]])

#将离散特征转换为独热编码形式
onehot = OneHotEncoder()
x_label_train = x_train.drop("age", 1)
x_label_test = x_test.drop("age", 1)

x_onehot_train = onehot.fit_transform(x_label_train).toarray()
x_onehot_test = onehot.fit_transform(x_label_test).toarray()

x_train = np.c_[x_std_train, x_onehot_train]
x_test = np.c_[x_std_test, x_onehot_test]

#步骤三：模型调参
#定义模型调参函数
def adjust_model(estimator, param_grid, model_name):
    model = GridSearchCV(estimator, param_grid)
    model.fit(x_train, y_train)
    print("{}模型最优得分：".format(model_name), model.best_score_)
    print("{}模型最优参数：".format(model_name), model.best_params_)

lr = LogisticRegression()
pg = {"C": [1, 2, 5, 10, 20, 50]}
adjust_model(lr, pg, "逻辑回归")

knn = KNeighborsClassifier()
pg = {"n_neighbors": [3, 4, 5, 6, 7]}
adjust_model(knn, pg, "k-NN")

dt = DecisionTreeClassifier()
pg = {"max_depth": [3, 4, 5, 6]}
adjust_model(dt, pg, "决策树")

rf = RandomForestClassifier()
pg = {"max_depth": [3, 4, 5, 6],
      "n_estimators": [50, 100, 150, 200]}
adjust_model(rf, pg, "随机森林")

#步骤四：选择最优模型，重新建模并评估
lr = LogisticRegression(C=10)
lr.fit(x_train, y_train)
y_ = lr.predict(x_test)
print("分类报告：\n", classification_report(y_test, y_))
print("混淆矩阵：\n", confusion_matrix(y_test, y_))

#求预测类别概率，用于ROC计算
y_pre = lr.predict_proba(x_test)[:, 1]
roc_score = roc_auc_score(y_test, y_pre)
fpr, tpr, th = roc_curve(y_test, y_pre)
plt.plot(fpr, tpr)
plt.title("ROC曲线得分{}".format(roc_score))
plt.show()