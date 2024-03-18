# Jinwook Huh

import load_data as ld
import numpy as np
import matplotlib.pyplot as plt

# helper function to visualize the lidar data (from the robot frame)
def replay_lidar(lidar_data):
	# lidar_data type: array where each array is a dictionary with a form of 't','pose','res','rpy','scan'
	#theta = np.arange(0,270.25,0.25)*np.pi/float(180)
	theta = lidar_data[0]['angle']

	for i in range(200,len(lidar_data),10):
		for (k,v) in enumerate(lidar_data[i]['scan']):
			if v > 30:
				lidar_data[i]['scan'][k] = 0.0

		# Jinwook's plot
		ax = plt.subplot(111, projection='polar')
		ax.plot(theta, lidar_data[i]['scan'])
		ax.set_rmax(10)
		ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
		ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
		ax.grid(True)
		ax.set_title("Lidar scan data", va='bottom')

		plt.draw()
		plt.pause(0.001)
		ax.clear()


# this is just a test to make sure that this loading function doesn't error
# the results aren't used anywhere
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_ts	 = ld.get_imu('data/imu20')
# this is just a test to make sure that this loading function doesn't error
# the results aren't used anywhere
FL, FR, RL, RR, enc_ts = ld.get_encoder('data/Encoders20')

# load and display plots of the lidar data (robot fixed in center, walls moving around robot)
lidar = ld.get_lidar('data/Hokuyo20')
replay_lidar(lidar)
