# Jinwook Huh
import numpy as np
import pickle
from scipy import io
import pdb

def get_lidar(file_name):
	data = io.loadmat(file_name+".mat")
	lidar = []
	angles = np.double(data['Hokuyo0']['angles'][0][0])
	ranges = np.array(data['Hokuyo0']['ranges'][0][0]).T
	ts_set = data['Hokuyo0']['ts'][0,0][0]

	idx = 0	
	for m in ranges:
		tmp = {}
		tmp['t'] = ts_set[idx]
		tmp['scan'] = m
		tmp['angle'] = angles
		lidar.append(tmp)
		idx = idx + 1
	return lidar



def get_encoder(file_name):

	data = io.loadmat(file_name+".mat")
#	pdb.set_trace()

	FR = np.double(data['Encoders']['counts'][0,0][0])
	FL = np.double(data['Encoders']['counts'][0,0][1])
	RR = np.double(data['Encoders']['counts'][0,0][2])
	RL = np.double(data['Encoders']['counts'][0,0][3])
	ts = np.double(data['Encoders']['ts'][0,0][0])
	return FR, FL, RR, RL, ts	

def get_imu(file_name):

	data = io.loadmat(file_name+".mat")

	acc_x = np.double(data['vals'])[0]
	acc_y = np.double(data['vals'])[1]
	acc_z = np.double(data['vals'])[2]
	gyro_x = np.double(data['vals'])[3]
	gyro_y = np.double(data['vals'])[4]
	gyro_z = np.double(data['vals'])[5]	
	ts = np.double(data['ts'][0])
#	pdb.set_trace()

	return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts	

