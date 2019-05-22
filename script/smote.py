#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:WWF
# datetime:2019/5/22 14:26
# 参考：https://www.cnblogs.com/paiandlu/p/8081763.html

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self, N=1, k=5):
        self.__shape = None
        self.__N = N
        self.__k = k

    def fit(self, samples):
        self.__shape = samples.shape  # 源样本的shape
        # 塑形为两位度才可以用KNN
        self.__samples = samples.reshape((self.__shape[0], -1))
        self.__tmp_shape = self.__samples.shape
        # 返回值的维度
        self.__ret_shape = (self.__shape[0] * self.__N,) + self.__shape[1:]

    def transform(self):
        # 如果没有喂给数据，则直接返回None
        if self.__shape == None:
            return None
        self.__index = 0  # 清零新增数据的索引
        self.__X = np.zeros((self.__tmp_shape[0] * self.__N, self.__tmp_shape[1]))  # 构造返回的数据，具体数据待填充
        neighbors = NearestNeighbors(n_neighbors=self.__k).fit(self.__samples)
        for i in range(self.__shape[0]):  # 根据每一个样本产生一个新样本
            # nnarray当前样本最近k个的样本的索引
            nnarray = neighbors.kneighbors(self.__samples[i].reshape(1, -1), return_distance=False)[0]
            # 根据当前样本索引和，最近k和样本生成一个新样本
            self.__new_one_sample(i, nnarray)
        return self.__X.reshape(self.__ret_shape)  # 重新塑形并返回

    def fit_transform(self, samples):
        self.fit(samples)
        return self.transform()

    # 根据当前样本索引和，最近k和样本生成一个新样本
    def __new_one_sample(self, i, nnarray):
        for _ in range(self.__N):
            # 从K个最近的样本随机挑选不同于当前样本的一个样本
            nn_idx = random.choice(nnarray)
            while (nn_idx == i):
                nn_idx = random.choice(nnarray)
            gap = self.__samples[nn_idx] - self.__samples[i]
            prob = random.random()
            # 根据公式生成新样本
            self.__X[self.__index] = self.__samples[i] + prob * gap
            self.__index += 1


if __name__ == '__main__':
    a = np.array([[1, 3, 4], [2, 5, 6], [4, 1, 2], [5, 1, 4], [3, 2, 4], [5, 3, 5]])
    print("\n" * 2, "测试维度为", a.shape)
    print("*" * 100)
    s = Smote()
    s.fit(a)
    print(s.transform())

    # 测试多维度支持
    b = np.zeros((10,) + a.shape)
    print("\n" * 2, "测试维度为", b.shape)
    print("*" * 100)
    for i in range(10):
        b[i, :] = s.fit_transform(a)
    print(s.fit_transform(b))