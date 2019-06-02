import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import time
start = time.clock()

file_to_read = open('D_origin.pickle', 'rb')
tmp = pickle.load(file_to_read)
data = tmp.values


original_dat = np.copy(data)

file_to_read1 = open('T_origin.pickle', 'rb')
tmp1 = pickle.load(file_to_read1)
test_data = tmp1.values

x1 = np.linspace(0, 2252,num=2251)
y1 = data[0,:2251]


a = test_data[0:2, :]
b = test_data[3:, :]


test_data = np.concatenate((a, b), axis=0)
print(test_data[:,-1])

test_data = test_data[6:,:]
print(test_data.shape)


#只取不太噪声的噪声往内添加，就是从第7个-13个，共7个数（7,8,9,10,11,12,13）
data = data[:,:-1]
test_data = test_data[:,:-1]
test_data = test_data.T


scaler = MinMaxScaler()
test_data = scaler.fit_transform(test_data)

test_data=test_data.T


data = data.T
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = data.T

lc_num = 401
lngth1 = int(test_data.shape[1] - lc_num)



print(tmp.values[175,-1])
print(tmp.values[155,-1])


new_training_data = np.zeros((186 * 6 * 12, 2251)) #!!!!!attention
label_lists = []

np.random.seed(1)

for i in range(186):

	label_lists.append([tmp.values[i,-1]] * 6 * 12)

	if i == 175:
		j_lists = [0, 1, 2, 3, 0, 1]
	elif i == 155:
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

		for k in range(12): #单个理论数据生成12x7个值,要调整调整这里！！
			
			copy_theo = np.copy(theoretical)
			copy_theo[theoretical < h_the] = np.random.permutation(part_data)

			lc_id = lc_id1 + lc_id2 + k

			new_training_data[lc_id] = copy_theo

new_train = new_training_data.tolist()

flat_list = [item for sublist in label_lists for item in sublist]


for i in range(186 * 6 * 12):
	new_train[i].append(flat_list[i])

end = time.clock()
print(str(end-start))

pickle.dump(new_train, open( "new_data_filter0.pickle", "wb" ))


'''
plot function for test new synthesize data
'''

# x1 = np.linspace(0, 2251,num=2251)
# #y1 = new_training_data[0][:-1]
# y1 = new_training_data[0][:]

# plt.figure(1)
# plt.subplot(211)
# plt.plot(x1, y1)


# plt.subplot(212)
# plt.plot(x1, original_dat[0, :-1]/np.max(np.abs(original_dat[0, :-1])))

# plt.show()

