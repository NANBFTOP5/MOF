import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

import random

import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = '0'

test_file="Test_ori_zif67_1.pickle"
tmp = np.load(test_file, allow_pickle = True)
data_14 = tmp.values

zif67_3 = np.zeros((3,2304))
list_test = [data_14[0]]

print(max(list_test[0]))


for k in range(1):

	test = list_test[k]
#test = data_14[0]
	num_large = 400


	data_zif90 = test[:2251]
	data_zif90 -= np.median(data_zif90)
	data_zif90[data_zif90<0] = 0
	########################替换peak为mean
	h_the = np.partition(data_zif90, -num_large)[-num_large]
	copy_no_peak = np.copy(data_zif90)

	copy_no_peak[data_zif90 > h_the] = np.median(data_zif90)

	########################替换noise为0
	copy_only_peak = np.copy(data_zif90)
	copy_only_peak[data_zif90 < h_the] = 0

	names= ['original', 'no_peak', 'only_peak']
	print(names)


	fft_len = 2304
	effect = int(fft_len / 2)

	data = np.zeros((len(names), fft_len))

	data[0, 20:2251+20] = data_zif90
	data[1, 20:2251+20] = copy_no_peak
	data[2, 20:2251+20] = copy_only_peak

	magnitd = np.zeros((len(names), effect+1))

	for i in range(len(names)):
		data[i] /= np.max(np.abs(data[i]))

		tmp_complex = (np.fft.fft(data[i]))[:int(fft_len/2+1)]
		magnitd[i] = np.log(np.abs(tmp_complex) + 0.000001)

		plt.clf()
		figname = str(i) + '.eps'
		plt.plot(magnitd[i],     'r', label='freq')
		# plt.plot(data[i],     'r', label='freq')

		np.savetxt('zif67_1magnitd.csv', magnitd[i], delimiter = ',')

		plt.title(str(i) + names[i])
		plt.legend(bbox_to_anchor=(0.2, 1.08, 1., .102), loc=1)
		plt.savefig(figname)
		plt.close(0)



# 		if names[i] == 'only_peak':
# 			recov_freq = np.zeros((fft_len), dtype=np.complex_)

# 			recov_sigl = np.zeros((2304), dtype=np.complex_)

# 			recov_freq[:200] = tmp_complex[:200]

# 			#duplicate range when doning fft
# 			duplicate_range = int(fft_len / 2 - 1)

# 			for freq_id in range(duplicate_range):
# 				recov_freq[fft_len - freq_id - 1] = np.conjugate(recov_freq[freq_id + 1])
			
# 			recov_sigl = np.fft.ifft(recov_freq)

# 			# plt.clf()
# 			# figname = str(i) + '_recover.eps'
# 			# plt.plot(recov_sigl.real,     'r', label='time')

# 			# plt.title(str(i) + names[i] + "_recover")
# 			# plt.legend(bbox_to_anchor=(0.2, 1.08, 1., .102), loc=1)
# 			# plt.savefig(figname)
# 			# plt.close(0)

# 			zif67_3[k,:] = recov_sigl.real
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# zif67_3 = zif67_3.T
# scaler = MinMaxScaler()
# zif67_3 = scaler.fit_transform(zif67_3)
# zif67_3 = zif67_3.T



# '''
# plot function for test new synthesize data
# '''

# x1 = np.linspace(0, 2304,num=2304)
# y1 = zif67_3[0]

# plt.figure(1)
# plt.plot(x1, y1)
# plt.show()

# np.save('3_zif67',zif67_3)


