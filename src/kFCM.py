# kFCM算法
import copy
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import decimal

# 用于初始化隶属度矩阵U
global MAX
MAX = 10000.0
# 用于结束条件
global Epsilon
Epsilon = 0.00000001
sigma = 150 # 高斯核函数的参数

def import_data_format_iris(file):
	""" 
	格式化数据，前四列为data，最后一列为cluster_location
	数据地址 http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
	"""
	data = []
	cluster_location =[]
	with open(str(file), 'r') as f:
		for line in f:
			current = line.strip().split(",")
			current_dummy = []
			for j in range(0, len(current)-1):
				current_dummy.append(float(current[j]))
			j += 1 
			if  current[j] == "Iris-setosa":
				cluster_location.append(0)
			elif current[j] == "Iris-versicolor":
				cluster_location.append(1)
			else:
				cluster_location.append(2)
			data.append(current_dummy)
	# print ("加载数据完毕")
	# print_matrix(data)
	# print_matrix(cluster_location)
	return data , cluster_location

def randomise_data(data):
	"""
	该功能将数据随机化，并保持随机化顺序的记录
	"""
	order = list(range(0, len(data)))
	random.shuffle(order)
	new_data = [[] for i in range(0, len(data))]
	for index in range(0, len(order)):
		new_data[index] = data[order[index]]
	# print_matrix(new_data)
	return new_data, order

def de_randomise_data(data, order):
	"""
	此函数将返回数据的原始顺序，将randomise_data()返回的order列表作为参数
	"""
	new_data = [[]for i in range(0, len(data))]
	for index in range(len(order)):
		new_data[order[index]] = data[index]
	return new_data
 
def print_matrix(list):
	""" 
	以可重复的方式打印矩阵
	"""
	for i in range(0, len(list)):
		print (list[i])

def initialise_U(data, cluster_number):
	"""
	这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
	"""
	global MAX
	U = []
	for i in range(0, len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):
			dummy = random.randint(1,int(MAX))
			# random.randint(a,b)：用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n：a<=n<=b
			current.append(dummy)
			rand_sum += dummy
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U
 
def distance(point, center):
	"""
	该函数计算2点之间的距离（作为列表）。我们指欧几里德距离，闵可夫斯基距离
	"""
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0, len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	dummy = math.exp((-dummy)/(2*sigma*sigma))
	return dummy
 
def end_conditon(U, U_old):
	"""
	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
	"""
	global Epsilon
	for i in range(0, len(U)):
		for j in range(0, len(U[0])):
			if abs(U[i][j] - U_old[i][j]) > Epsilon :
				return False
	return True
 
def normalise_U(U):
	"""
	在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
	"""
	for i in range(0, len(U)):
		maximum = max(U[i])
		for j in range(0, len(U[0])):
			if U[i][j] != maximum:
				U[i][j] = 0
			else:
				U[i][j] = 1
	return U
 
# m的最佳取值范围为[1.5，2.5]
def kfuzzy(data, cluster_number, m):
	"""
	这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
	参数是：簇数(cluster_number)和隶属度的因子(m)
	"""
	# 迭代次数
	ilteration_num = 0
	# 初始化隶属度矩阵U
	U = initialise_U(data, cluster_number)
	# print_matrix(U)
    # 使用fcm初始化聚类中心
	C = []
	for j in range(0, cluster_number):
		current_cluster_center = []
		for i in range(0, len(data[0])):
			dummy_sum_num = 0.0
			dummy_sum_dum = 0.0
			for k in range(0, len(data)):
				# 分子
				dummy_sum_num += (U[k][j]) * data[k][i]
				# 分母
				dummy_sum_dum += (U[k][j] ** m)
			# 第i列的聚类中心
			current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
		# 第j簇的所有聚类中心
		C.append(current_cluster_center)
	# 循环更新U
	while (True):
		# 迭代次数
		ilteration_num += 1
		# 创建它的副本，以检查结束条件
		U_old = copy.deepcopy(U)
        # 距离函数
		distance_matrix =[]
		for i in range(0, len(data)):
			current = []
			for j in range(0, cluster_number):
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)
 
		# 更新U
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				dummy = 0.0
				for k in range(0, cluster_number):
					# 分母
					dummy += (1-distance_matrix[i][k]) ** (-1/(m-1))
				U[i][j] = ((1-distance_matrix[i][j])**(-1/(m-1))) / dummy			
		# 计算聚类中心
		C = []
		for j in range(0, cluster_number):
			current_cluster_center = []
			for i in range(0, len(data[0])):
				dummy_sum_num = 0.0
				dummy_sum_dum = 0.0
				for k in range(0, len(data)):
					# 分子
					dummy_sum_num += (U[k][j]) * data[k][i] * distance_matrix[k][j]
					# 分母
					dummy_sum_dum += (U[k][j] * distance_matrix[k][j])
				# 第i列的聚类中心
				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
			# 第j簇的所有聚类中心
			C.append(current_cluster_center)
		
 
		if end_conditon(U, U_old):
			#print ("结束聚类")
			break
	# print ("迭代次数：" + str(ilteration_num))
	# print ("标准化 U")
	U = normalise_U(U)
	return U

def checker_iris(final_location):
	"""
	和真实的聚类结果进行校验比对
	"""
	right = 0.0
	for k in range(0, 3):
		checker =[0,0,0]
		for i in range(0, 50):
			for j in range(0, len(final_location[0])):
				if final_location[i + (50*k)][j] == 1:
					checker[j] += 1
		right += max(checker)
		# print (right)
	#print("正确聚类的数量：" + str(right))
	answer =  right / 150 * 100
	return "准确度：" + str(answer) +  "%"
 
if __name__ == '__main__':
	
	# 加载数据
	data, cluster_location = import_data_format_iris("iris.txt")
	# print_matrix(data)
 
	# 随机化数据
	data , order = randomise_data(data)
	# print_matrix(data)
 
	start = time.time()
	# 现在我们有一个名为data的列表，它只是数字
	# 我们还有另一个名为cluster_location的列表，它给出了正确的聚类结果位置
	# 调用模糊C均值函数
	final_location = kfuzzy(data , 3 , 2)
	# print_matrix(final_location)
 
	# 还原数据
	final_location = de_randomise_data(final_location, order)
	# print_matrix(final_location)
 
	# 准确度分析
	print (checker_iris(final_location))
	print ("用时：{0}".format(time.time() - start))