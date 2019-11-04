import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')												# 过滤各种警告，也可以打开。看看有什么警告
import time
start = time.clock()
#-------------------------------------------------------------------------------
file_to_read = open('D_origin.pickle', 'rb')									# 读取原先的186个理论数据
tmp = pickle.load(file_to_read)
data = tmp.values

file_to_read2 = open('prep_theo_data.pickle', 'rb')								# 读取后来新处理的理论数据
tmp2 = pickle.load(file_to_read2)
data2 = tmp2.values

# original_dat = np.copy(data)   #for plot use

file_to_read1 = open('T_origin.pickle', 'rb')									# 读取原先的19个测试数据
tmp1 = pickle.load(file_to_read1)
test_data = tmp1.values

# x1 = np.linspace(0, 2252,num=2251)  #for plot use
# y1 = data[0,:2251]
#-------------------------------------------------------------------------------# 请修改又进来了多少新的label，【add_data_size】！！！！！！！
add_data_size   = 299
input_data_size = 186 + add_data_size

#-------------------------------------------------------------------------------
''' 
	选取 a ['ZIF-67' 'ZIF-67' 'ZIF-67'], b ['ZIF-71' 'ZIF-71' 
	'ZIF-71' 'ZIF-8' 'ZIF-8' 'ZIF-8' 'ZIF-8' 'ZIF-90' 'ZIF-90' 'ZIF-90']
	作为添加的噪声，test_data 的 label应该如上所示
'''
a = test_data[0:2, :]															# 扔掉一个zif67的噪声，因为幅度太大，有干扰
b = test_data[3:, :]
test_data = np.concatenate((a, b), axis=0)
#--------------------------------------------------------------------------------
'''
只取不太噪声的噪声往内添加，就是从第7个-13个，共7个数（7,8,9,10,11,12,13）
并合并新的数据的数值和label
'''
test_data = test_data[6:,:]
data = data[:,:-1]																# 去除掉186个理论数据的lable
data = np.concatenate((data, data2), axis=0)									# 合并186个理论数据(data)和新读取的理论数据的值(data2)

file_to_read3 = open('prep_theo_label.pickle', 'rb')							# 读取新的理论数据的lable
tmp3 = pickle.load(file_to_read3)
data3 = tmp3

label186 = tmp.values[:,-1]
label86 = data3
label = np.concatenate((label186, label86), axis=0)
# pd.to_pickle(label, "labelinput_data_size")

print('the previous lable total number is:  {}'.format(label186.shape))
print('the new imported lable total number is:  {}'.format(len(label86)))
print('the total combined lable number is:  {}'.format(label.shape))

# print(label) 																	# 打印查看全部的label，看是佛有数字或不正确的的lable存在其中
#-------------------------------------------------------------------------------
#归一和转置
test_data = test_data[:,:-1]
test_data = test_data.T
scaler = MinMaxScaler()
test_data = scaler.fit_transform(test_data)
test_data=test_data.T
data = data.T
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = data.T
#-------------------------------------------------------------------------------
'''
	添加噪声合成用于训练的数据
'''
lc_num = 401
lngth1 = int(test_data.shape[1] - lc_num)
new_training_data = np.zeros((input_data_size * 6 * 12, 2251)) 					# 创建空的容器
label_lists = []

np.random.seed(1)

for i in range(input_data_size):

	label_lists.append([label[i]] * 6 * 12)

	if i == 175:                              #zif-90
		j_lists = [0, 1, 2, 3, 0, 1]
	elif i == 155:                            #zif-8
		j_lists = [4, 5, 6, 4, 5, 6]
	else:
		j_lists = np.arange(6)	

	theoretical = data[i]
	h_the = np.partition(theoretical, -lc_num)[-lc_num]

	assert(len(j_lists) == 6)

	lc_id1 = i * 6 * 12

	for j in range(len(j_lists)):
		
		experiment = test_data[ j_lists[j] ]
		h_exp = np.partition(experiment, -lc_num)[-lc_num]

		part_data = experiment[experiment <= h_exp][:lngth1]

		lc_id2 = j * 12

		for k in range(12): 													#单个理论数据生成12x7个值,要调整调整这里！！
			
			copy_theo = np.copy(theoretical)
			copy_theo[theoretical < h_the] = np.random.permutation(part_data)

			lc_id = lc_id1 + lc_id2 + k

			new_training_data[lc_id] = copy_theo

new_train = new_training_data.tolist()

flat_list = [item for sublist in label_lists for item in sublist]				# 因为产生的是list包含list的一个数据，此处展开


print('double check the total combined lable number is:  {}'.format(len(label_lists)))
print('after filtering the repeated labels, the lable number is')
print(len(set(flat_list)))

print('the total train data number is:  {}'.format(len(new_train)))
print('the total train label number is:  {}'.format(len(flat_list)))

#-------------------------------------------------------------------------------
'''
	将新合成的label添加到新的数据后面
'''
for i in range(input_data_size * 6 * 12):
	new_train[i].append(flat_list[i])

end = time.clock()
print('time cost')
print(str(end-start))

pickle.dump(new_train, open( "new_data_filter0.pickle", "wb" ))

