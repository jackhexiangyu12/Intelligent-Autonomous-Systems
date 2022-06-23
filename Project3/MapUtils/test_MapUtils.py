# Jinwook Huh 

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import matplotlib.image as mpimg
import MapUtils as MU
from mpl_toolkits.mplot3d import Axes3D
import time
import pdb
# todo load mat
dataIn = io.loadmat("../data/Hokuyo20.mat")
ranges = np.array([dataIn['Hokuyo0']['ranges'][0][0][:,0]]).T
angles = np.double(dataIn['Hokuyo0']['angles'][0][0])

indValid = np.logical_and((ranges < 30),(ranges> 0.1))
ranges = ranges[indValid]
angles = angles[indValid]

# init MAP
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -20  #meters
MAP['ymin']  = -20
MAP['xmax']  =  20
MAP['ymax']  =  20 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

# xy position in the sensor frame
xs0 = np.array([ranges*np.cos(angles)]);
ys0 = np.array([ranges*np.sin(angles)]);

# convert position in the map frame here 
Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0)
### HERE
# convert from meters to cells
xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

# build an arbitrary map  
indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
##inds = sub2ind(size(MAP.map),xis(indGood),yis(indGood));
MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1    # Maybe this is a problem

x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

x_range = np.arange(-0.2,0.2+0.05,0.05)
y_range = np.arange(-0.2,0.2+0.05,0.05)

#plot original lidar points
fig1 = plt.figure(1);
plt.plot(xs0,ys0,'.k')

#plot map
fig2 = plt.figure(2);
plt.imshow(MAP['map'],cmap="hot");

print("Testing getMapCellsFromRay...")
r = MU.getMapCellsFromRay(0,1,[10, 9],[5, 6],1000)
r_ex = np.array([[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8],
						[1,1,2,2,3,3,3,4,4,5,1,2,2,3,3,4,4,5,5]])

if np.sum(r == r_ex) == np.size(r_ex):
	print("...Test passed.")
else:
	print("...Test failed.")

