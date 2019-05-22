#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:WWF
# datetime:2019/5/22 11:31
"""
1. sklearn 中分类方法（Naive Bayes, Logistic Regression, XGboost, lightGBM, SVM, KNN, \
   Random Forest, Decision Tree, GBDT）测试文本二分类效果，所用数据（neg.txt, pos.txt），各个分类器
   的测试效果如下：
    Naive Bayes Result: 0.9032786885245901
    Logistic Regression Result: 0.8737704918032787
    SVM Result: 0.8836065573770492
    Xgboost Result: 0.8442622950819673
    LightGBM Result: 0.8508196721311475
    knn Result: 0.6081967213114754
    RFC Result: 0.8721311475409836
    DT Result: 0.8
    GBDT Result: 0.8508196721311475

2. 准备测试集成方法效果
"""
import warnings
warnings.filterwarnings('ignore')
import random
from imblearn.combine import SMOTEENN
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error


# 随机抽取80%的数据作为训练集，20%的数据作为测试集
def get_train_test_data(data_path):
    data_X = []
    data_y = []
    with open(data_path, encoding='UTF-8') as f:
        data_list = f.readlines()
        random.shuffle(data_list)
        for line in data_list:
            row = line.rstrip('\n').split('\t')
            if len(row) < 2:
                continue
            feature = ' '.join(row[0].split(','))
            data_X.append(feature)
            data_y.append(row[1])

        # 80%作为训练集，20%作为测试集
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
        return X_train, X_test, y_train, y_test


# 训练并测试模型-NB
def train_model_NB():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    clf = MultinomialNB(alpha=0.01)
    joblib.dump(clf, '../model/naive.bayes.pkl')
    clf = joblib.load('../model/naive.bayes.pkl')
    clf.fit(train_data, y_train)
    # y_predict = clf.predict(test_data)
    print('Naive Bayes Result:', clf.score(test_data, y_test))


# 训练并测试模型-logisticRegression
def train_model_LR():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    lr = LogisticRegression(C=1000)

    joblib.dump(lr, '../model/logistic.regression.pkl')
    lr = joblib.load('../model/logistic.regression.pkl')
    lr.fit(train_data, y_train)
    print('Logistic Regression Result:', lr.score(test_data, y_test))


# 训练并测试模型-svm
def train_model_SVM():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = SVC(C=1000.0)
    joblib.dump(clf, '../model/svm.pkl')
    clf = joblib.load('../model/svm.pkl')
    clf.fit(train_data, y_train)

    print('SVM Result:', clf.score(test_data, y_test))


# 训练并测试模型-xgboost
# 参考：https://www.cnblogs.com/wanglei5205/p/8579244.html
def train_model_xgboost():
    """
    learning_rate=0.1,
    n_estimators=1000,  # 树的个数--1000棵树建立xgboost
    max_depth=6,  # 树的深度
    min_child_weight = 1,  # 叶子节点最小权重
    gamma=0.,  # 惩罚项中叶子结点个数前的参数
    subsample=0.8,  # 随机选择80%样本建立决策树
    colsample_btree=0.8,  # 随机选择80%特征建立决策树
    objective='multi:softmax',  # 指定损失函数
    scale_pos_weight=1,  # 解决样本个数不平衡的问题
    random_state=27  # 随机数
    """
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    clf = XGBClassifier()  # 随机数
    clf.fit(train_data, y_train)
    joblib.dump(clf, '../model/xgboost.pkl')
    clf = joblib.load('../model/xgboost.pkl')

    # print(clf.predict(test_data))  # 输出预测结果
    print('Xgboost Result:', clf.score(test_data, y_test))


# 训练并测试模型-LightGBM
# 参考：https://blog.csdn.net/luanpeng825485697/article/details/80236759
def train_model_lightGBM():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = lgb.LGBMClassifier()
    joblib.dump(clf, '../model/lightgbm.pkl')
    clf = joblib.load('../model/lightgbm.pkl')
    clf.fit(train_data, y_train)
    print('LightGBM Result:', clf.score(test_data, y_test))


# 训练并测试模型-K-nearest-neighbour
def train_model_knn():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = KNeighborsClassifier()
    clf.fit(train_data, y_train)
    joblib.dump(clf, '../model/knn.pkl')
    clf = joblib.load('../model/knn.pkl')
    print('knn Result:', clf.score(test_data, y_test))


# 训练并测试模型-Random-Forest-Classifier
def train_model_rfc():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = RandomForestClassifier()
    clf.fit(train_data, y_train)
    joblib.dump(clf, '../model/rfc.pkl')
    clf = joblib.load('../model/rfc.pkl')
    print('RFC Result:', clf.score(test_data, y_test))


#  训练并测试模型-Decision-Tree
def train_model_dt():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, y_train)
    joblib.dump(clf, '../model/dt.pkl')
    clf = joblib.load('../model/dt.pkl')
    print('DT Result:', clf.score(test_data, y_test))


#  训练并测试模型GBDT(Gradient Boosting Decision Tree)
def train_model_gbdt():
    X_train, X_test, y_train, y_test = get_train_test_data('../data/merge.txt')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = GradientBoostingClassifier()
    clf.fit(train_data, y_train)
    joblib.dump(clf, '../model/gbdt.pkl')
    clf = joblib.load('../model/gbdt.pkl')
    print('GBDT Result:', clf.score(test_data, y_test))


if __name__ == "__main__":
    train_model_NB()
    train_model_LR()
    train_model_SVM()
    train_model_xgboost()
    train_model_lightGBM()
    train_model_knn()
    train_model_rfc()
    train_model_dt()
    train_model_gbdt()

