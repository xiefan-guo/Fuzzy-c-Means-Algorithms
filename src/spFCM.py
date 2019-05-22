# spFCM算法
import copy
import math
import random
import time
 
# 数据地址 http://archive.ics.uci.edu/ml/machine-learning-databases/iris/

# 用于初始化隶属度矩阵U
MAX = 10000.0
# 用于结束条件
Epsilon = 0.00000001

def import_data_first_iris(file, num):
	""" 
	采样第一份数据，前四列为data，最后一列为cluster_location
	"""
	with open(str(file), 'r') as f:
		ff = list(f)
		# print(ff)
		print("总长度为："+str(len(ff)))
		nf = random.sample(ff, num)
		print("nf长度为："+str(len(nf)))
	# 剩余的数据，因为列表里有重复的元素，所以要这样做
	for i in nf:
		ff.remove(i)
	print("剩余："+str(len(ff)))
	print("抽样数据加载完毕")
	print("在150个数据中随机抽样：" + str(num) + "个数据")
	return nf, ff

def acquire_data(f, num):
	"""
	采样后续的ns-1份数据
	"""
	print("f长度为："+str(len(f)))
	nf = random.sample(f, num)
	print("nf长度为："+str(len(nf)))
	for i in nf:
		f.remove(i)
	print("ret长度为："+str(len(f)))
	return nf, f

def import_data_full_iris(file):
	""" 
	全部数据，前四列为data，最后一列为cluster_location
	"""
	datafull = []
	cluster_locationfull =[]
	with open(str(file), 'r') as f:
		for line in f:
			current = line.strip().split(",")
			current_dummy = []
			for j in range(0, len(current)-1):
				current_dummy.append(float(current[j]))

			j += 1
			if  current[j] == "Iris-setosa":
				cluster_locationfull.append(0)
			elif current[j] == "Iris-versicolor":
				cluster_locationfull.append(1)
			else:
				cluster_locationfull.append(2)
			datafull.append(current_dummy)

		# print_matrix(datafull)
		# print_matrix(cluster_locationfull)
	# print ("加载数据完毕")
	return datafull , cluster_locationfull

def pre_deal(nf):
	"""
	预处理数据，前四列为data，最后一列为cluster_location
	"""
	data = []
	cluster_location =[]
	for line in nf:
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
	print ("抽样数据预处理完毕")
	return data , cluster_location
		
def randomise_data(data):
	"""
	该功能将数据随机化，并保持随机化顺序的记录
	"""
	order = list(range(0, len(data)))
	random.shuffle(order)# 用于将一个列表中的元素打乱
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
	该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。        闵可夫斯基距离
	"""
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0, len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	return math.sqrt(dummy)
 
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
def wfcm(data, cluster_number, m, w, C):
	"""
	这是循环里的wfcm，可以使用上次的聚类中心C加速收敛，并返回最终的归一化隶属矩阵U
	"""
	# 初始化隶属度矩阵U
	U = initialise_U(data, cluster_number)
	# print_matrix(U)
	# 迭代次数
	iteration_num = 0
	# 循环更新U
	while (True):
		# 迭代次数
		iteration_num += 1
		# 创建它的副本，以检查结束条件
		U_old = copy.deepcopy(U)
		# 计算聚类中心
		if C == []:
			for j in range(0, cluster_number):
				current_cluster_center = []
				for i in range(0, len(data[0])):
					dummy_sum_num = 0.0
					dummy_sum_dum = 0.0
					for k in range(0, len(data)):
						# 分子
						dummy_sum_num += (U[k][j] ** m) * data[k][i] * w[k]
						# 分母
						dummy_sum_dum += (U[k][j] ** m) * w[k]
					# 第i列的聚类中心
					current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
				# 第j簇的所有聚类中心
				C.append(current_cluster_center)
		# print_matrix(C)
		# 创建一个距离向量, 用于计算U矩阵。
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
					# 如果除数为0,则不处理
					if distance_matrix[i][k] == 0:
						U[i][j] = 0
						break
					dummy += (distance_matrix[i][j ] / distance_matrix[i][k]) ** (2/(m-1))
				if U[i][j] != 0:
					U[i][j] = 1 / dummy
 
		if end_conditon(U, U_old):
			print("结束抽样的聚类")
			break
		# 每次聚类中心要清零
		#C = []
	print("迭代次数：" + str(iteration_num))
	return U, C

def extension(data, cluster_number, m, C):
	"""
	用抽样样本得到的U扩展到整个数据集上，参数包括隶属度矩阵U和聚类中心C
	"""
	# 创建一个距离向量, 用于计算U矩阵。
	distance_matrix =[]
	for i in range(0, len(data)):
		current = []
		for j in range(0, cluster_number):
			current.append(distance(data[i], C[j]))
		distance_matrix.append(current)
	# 初始化U
	U = [[0]*cluster_number for i in range(len(data))]
	# 更新U
	for j in range(0, cluster_number):	
		for i in range(0, len(data)):
			dummy = 0.0
			for k in range(0, cluster_number):
				# 分母
				dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))
			U[i][j] = 1 / dummy
	
	print ("拓展到整个数据集上")
	print ("标准化 U")
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
	print("正确聚类的数量：" + str(right))
	answer =  right / 150 * 100
	return "准确度：" + str(answer) +  "%"

def initialise_w(ns):
	"""
	初始化w
	"""
	w = []
	for i in range(0, ns):
		w.append(1)
	return w

def ilteral_w(U, w, cluster_number, data):
	"""
	迭代更新w
	"""
	new_w = []
	for i in range(0, cluster_number):
		dummy_sum_num = 0.0
		for j in range(0, len(data)):
			dummy_sum_num += U[j][i] * w[j]
		new_w.append(dummy_sum_num)
	for i in range(0, len(data)):
		new_w.append(1)
	return new_w

def ilteral_all(ret, ns, first_U, V, w, cluster_number):
	"""
	除了第一份之后的每份的迭代，每次都使用上次的聚类中心加这份的数据作为wfcm的输入数据，
	每次都用上次的聚类中心初始化聚类中心，权重w每次也要迭代
	"""
	for i in range(1, int(150/15)):
		print("第"+str(i)+"份")	
		# nf是每次采样出来的数据，ret是第一次采样剩余的数据
		nf ,ret = acquire_data(ret, ns)
		# 预处理数据
		nf, cluster_location = pre_deal(nf)
		# 迭代更新w
		new_w = ilteral_w(first_U, w, 3, nf)
		print("w更新完成")
		# 深拷贝
		new_data = copy.deepcopy(V)
		# 数据组合
		for i in range(0, len(nf)):
			new_data.append(nf[i])
		# print_matrix(V)
		# print_matrix(new_data)
		U, V = wfcm(new_data, 3, 2, new_w, V)
		print("----------------------------------------")
	return V

if __name__ == '__main__':
	print("第0份")
	# 抽样的数据量
	ns = 15
	# 加载抽样的数据
	nf, ret = import_data_first_iris("iris.txt", ns)
	# 预处理数据
	data, cluster_location = pre_deal(nf)
	# 初始化w
	w = initialise_w(ns)
	start = time.time()
	# 调用模糊C均值函数
	first_U, V = wfcm(data, 3, 2, w, [])
	print("-----------------------------------------")
	# 剩余数据集的迭代
	final_V = ilteral_all(ret, ns, first_U, V, w, 3)
	# 加载完整数据
	datafull, cluster_locationfull = import_data_full_iris("iris.txt")
	final_location = extension(datafull, 3, 2, final_V)
	# 准确度分析
	print (checker_iris(final_location))
	print ("用时：{0}".format(time.time() - start))