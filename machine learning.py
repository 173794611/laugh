#由文件breast-cancer-wisconsin.names中可知数据集的相关性质
#数据集共699条信息/数据集由16出缺失值，缺失值使用‘?’表示/数据集中良性数据有458条，恶性数据有241条


###缺失值的处理和分割数据集
#因为缺失的数据不多，所以暂时先采用丢弃带有‘？’的数据，加上前面读取数据、添加表头的操作，代码如下
# import the packets
import numpy as np
import pandas as pd

DATA_PATH = "breast-cancer-wisconsin.data"

# create the column names
#文件有11个列，第一个列为ID号，2-10为特征，11列为标签（良性/恶性）
columnNames = [
    'Sample code number',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
]

data = pd.read_csv(DATA_PATH, names = columnNames)
# show the shape of data
# print data.shape

# use standard missing value to replace "?"
data = data.replace(to_replace = "?", value = np.nan)
# then drop the missing value
data = data.dropna(how = 'any')

# print data.shape
print (data.shape)

#将数据集分为两部分：训练数据集和测试数据集，
# then we split this dataset in to 2 parts:
#  - train dataset
#  - test  dataset
# here we use `train_test_split` to split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data[ columnNames[1:10] ], # features
    data[ columnNames[10]   ], # labels
    test_size = 0.25,
    random_state = 43
)



#应用机器学习的模型前，应该将每个特征值的数值均转化为值为0，方差为1的数据，从而使得训练处的模型不会被某个维度较大的值主导
# let's have a look at the distribution of the train data
# print y_train.value_counts()
# and the test's dataset
# print y_test.value_counts()

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)




#建立逻辑回归以及支持向量机的模型
# use logestic-regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y = lr.predict(X_test)
# use svm
from sklearn.svm import LinearSVC

lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
svm_y = lsvc.predict(X_test)

# now we will check the performance of the classifier
from sklearn.metrics import classification_report
# use the classification_report to present result
# `.score` method can be used to test the accuracy
print 'Accuracy of the LogesticRegression: ', lr.score(X_test, y_test)
print classification_report(y_test, lr_y, target_names = ['Benign', 'Malignant'])


# print 'Accuracy on the train dataset: ', lr.score(X_train, y_train)
# print 'Accuracy on the predict result (should be 1.0): ', lr.score(X_test, lr_y)

print 'Accuracy of the SVM: ' , lsvc.score(X_test, y_test)
print classification_report(y_test, svm_y, target_names = ['Benign', 'Malignant'])