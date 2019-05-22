# -*- coding:utf-8 -*-
from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
import copy
import time

# 读取.csv文件
data_full = pd.read_csv("Iris.csv")
# 得到表格的列名
columns = list(data_full.columns)
# 前四个列名是鸢尾花特征（最后一列是鸢尾花种类）
features = columns[0:len(columns) - 1]
# 提取需要聚类的数据（根据列名提取前四列）
data = data_full[features]
# 分类数
c = 3
# 最大迭代数
MAX_ITER = 100
# 阈值
Epsilon = 0.00000001
# 样本数，行数
n = len(data)
# 模糊参数
m = 2.00


# 初始化模糊矩阵（隶属度矩阵 U）
# 用值在0，1间的随机数初始化隶属矩阵，得到c列的U，使其满足隶属度之和为1
def initialize():
    # 返回一个模糊矩阵的列表
    U = list()
    # 标准化
    for i in range(n):
        # 初始化，给与随机的隶属度
        random_list = [random.random() for i in range(c)]
        # print(random_list)
        # 标准化：值/每列的和
        summation = sum(random_list)
        # print(summation)
        temp_list = [x / summation for x in random_list]
        # print(temp_list)
        U.append(temp_list)
    return U


# 计算中心矩阵 V
def calculateCenter(U):
    # zip函数实现矩阵的转置 https://blog.csdn.net/qinze5857/article/details/80447835
    U_zhuanzhi = list(zip(*U))
    # 中心矩阵，列表
    V = list()
    for j in range(c):
        # 取出转置矩阵每列的150个元素
        x = U_zhuanzhi[j]
        # uij的m次方，m为模糊参数
        xraised = [e ** m for e in x]
        # 分母
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            # 得到分子中的 xj
            data_point = list(data.iloc[i])
            # uij的m次方 乘以 xj
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        # 分子：上面的结果求和
        numerator = map(sum, zip(*temp_num))
        # 求聚类中心
        center = [z / denominator for z in numerator]
        print(center)
        V.append(center)
    return V


# 更新隶属度矩阵 U
def U_update(U, V):
    # 2/(m-1)
    p = float(2 / (m - 1))
    for i in range(n):
        # 取出文件中的每一行数据
        x = list(data.iloc[i])

        # 求dij
        distances = [np.linalg.norm(list(map(operator.sub, x, V[j]))) for j in range(c)]
        for j in range(c):
            # 分母
            den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(c)])
            U[i][j] = float(1 / den)
    return U


# 迭代，最多迭代MAX_ITER次
# 计算中心矩阵V——》更新隶属度矩阵U——》计算更新后的中心矩阵V_update，V_update和V的距离若小于阈值则停止
def iteration(U):
    # 最大迭代次数：MAX_ITER=100
    iter = 0
    while iter <= MAX_ITER:
        iter += 1
        # 计算聚类中心矩阵 V
        V = calculateCenter(U)
        # 更新模糊矩阵 U
        U = U_update(U, V)
        # 得到更新后的中心矩阵
        V_update = calculateCenter(U)
        # 如果V_update和V的距离小于阈值，迭代停止
        juli = 0
        for i in range(c):
            for j in range(len(columns) - 1):
                juli = (V_update[i][j] - V[i][j]) ** 2 + juli
        if (sqrt(juli) < Epsilon):
            break
    return V, U


# 获得聚类结果（判断样本属于哪个类）
def getResult(U):
    results = list()
    # for循环取出U矩阵的150行数据
    for i in range(n):
        # 此时每条数据有3个隶属度，取最大的那个，并返回index值即0或1或2
        max_value, index = max((value, index) for (index, value) in enumerate(U[i]))
        # 以此值（0、1、2）作为结果
        results.append(index)
    return results


# 主函数
def FCM():
    start = time.time()
    # 初始化模糊矩阵 U
    U = initialize()
    # 迭代，最大迭代次数：100
    V, U = iteration(U)
    # 获得聚类结果
    results = getResult(U)
    # 打印聚类所用时长
    print("用时：{0} s".format(time.time() - start))
    return results, V, U


# Xie-Beni聚类有效性
def xie_beni(membership_mat, center, data):
    sum_cluster_distance = 0
    min_cluster_center_distance = inf
    for i in range(c):
        for j in range(n):
            sum_cluster_distance = sum_cluster_distance + membership_mat[j][i] ** 2 * sum(
                power(data[j, :] - center[i, :], 2))  # 计算类一致性
    for i in range(c - 1):
        for j in range(i + 1, c):
            cluster_center_distance = sum(power(center[i, :] - center[j, :], 2))  # 计算类间距离
            if cluster_center_distance < min_cluster_center_distance:
                min_cluster_center_distance = cluster_center_distance
    return sum_cluster_distance / (n * min_cluster_center_distance)


# matplotlib需要array类型的数据
results, V, U = FCM()
V_array = np.array(V)
DATA = np.array(data)
results = np.array(results)

# Xie-Beni聚类有效性
print("聚类有效性：", xie_beni(U, V_array, DATA))

# 将DATA的第一列和第二列作为x、y轴绘图
xlim(4, 8)
ylim(1, 5)
# 创建一个绘图窗口
plt.figure(1)
# 画散点图
# 样本点   其中nonzero(results==0)为取出0这一类的下标
plt.scatter(DATA[nonzero(results == 0), 0], DATA[nonzero(results == 0), 1], marker='o', color='r', label='0', s=30)
plt.scatter(DATA[nonzero(results == 1), 0], DATA[nonzero(results == 1), 1], marker='+', color='b', label='1', s=30)
plt.scatter(DATA[nonzero(results == 2), 0], DATA[nonzero(results == 2), 1], marker='*', color='g', label='2', s=30)
# 中心点
plt.scatter(V_array[:, 0], V_array[:, 1], marker='x', color='m', s=50)
plt.show()