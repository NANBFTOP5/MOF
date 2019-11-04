'''
combing each material data into one file
将单独的理论数据tsv文件整合进一个pickle文件
'''

import os
import numpy as np
import pandas as pd

all_data_path = r'/home/wanghong/work/MOF_project/2019_11_3MOF/20191002mofdata'

data   = pd.DataFrame()															# 创建空的容器
label  = []

# 'test data'
# test_file = r'/home/wanghong/work/MOF_project/2019_9MOF/20191002mofdata/149068 MOF HOCHEM.tsv'
# one_data         = pd.read_csv(test_file, sep='\t', header=None)
# one_data_column  = pd.DataFrame(one_data.iloc[:,-1])
# print(one_data_column.type)

for i, file_name in enumerate(os.listdir(all_data_path)):

	one_data_path    = os.path.join(all_data_path, file_name)

	if one_data_path[-3:] != 'tsv':        										# 只有tsv结尾的文件数据能被读取
		continue

	one_data         = pd.read_csv(one_data_path, sep='\t', header=None)		# 分离数据
	one_data_column  = pd.DataFrame(one_data.iloc[:,-1])						# 数据分为两列，第一列1-50间隔0.2，无用。取第二列对应的理论值

	one_data_row     = one_data_column.T 										# 列数据转置成行数据
	data = pd.concat([data, one_data_row])										# 将每次读取的单个数据和前面读取的数据合并

	# print((file_name.split()[-1])[:-4]) 										#ex: UCOQEK.tsv remove .tsv

	label.append(((file_name.split()[-1])[:-4]))

pd.to_pickle(data,  "prep_theo_data.pickle" )
pd.to_pickle(label, "prep_theo_label.pickle")
print('output lable type is:  {}'.format(type(label)))
print('output lable length is:  {}'.format(len(label)))

# print(label) 																	# 打印查看全部的label，看是佛有数字或不正确的的lable存在其中
