#Daniel D. Lee, Alex Kushleyev, Kelsey Saulnier, Nikolay Atanasov, Jinwook Huh
import numpy as np

#Bresenham's line algorithm
# Inputs:
# xrobot: the x index of the cell grid the robot is in
# yrobot: the y index of the cell grid the robot is in
# xends: the x indices of the cells at the endpoints of the lidar hits (a vector)
# yends: the y indices of the cells at the endpoints of the lidar hits (a vector)
# maxMap: max width/height of the grid map (only actually used in the cython version to allocate a vector of large enough size to hold theoutput points)
# Output:
# Indices of the pixels in the ray between start and end points (exclusive of end point)
# Output is a numpy array of shape (2, P) where P is the total number of pixels in all the rays
def getMapCellsFromRay(xrobot,yrobot,xends,yends, maxMap):
	xyio = np.array([[],[]])
	for x1,y1  in zip(xends,yends):
		x0 = xrobot
		y0 = yrobot
		steep = (np.abs(y1-y0) > np.abs(x1-x0))
		if steep:
			temp = x0
			x0 = y0
			y0 = temp
			temp = x1
			x1 = y1
			y1 = temp
		if x0 > x1:
			temp = x0
			x0 = x1
			x1 = temp
			temp = y0
			y0 = y1
			y1 = temp
		deltax = x1 - x0
		deltay = np.abs(y1 - y0)
		error = deltax / 2.
		y = y0
		ystep = 0
		if y0 < y1:
		  ystep = 1
		else:
		  ystep = -1
		if steep:
			for x in np.arange(x0,x1):
				xyio = np.concatenate((xyio,np.array([[y],[x]])),axis=1)
				error = error - deltay
				if error < 0:
					y += ystep
					error += deltax
		else:
			for x in np.arange(x0,x1):
				xyio = np.concatenate((xyio,np.array([[x],[y]])),axis=1)
				error = error - deltay
				if error < 0:
					y += ystep
					error += deltax
	return xyio


