import numpy as np1
from load_data import *
import matplotlib.pyplot as plt


def show_paths(width, X_area, Y_area, indicesPerInt, fileNum, numParticles, Noise_bound, avg):
    # average right side and left side encoding data and lidar data to match length
    f_r, f_l, r_r_1, r_l, t_s_1 = get_encoder("data/" + "Encoders" + str(fileNum))
    l, lidar_data, r = Pre_deal(avg, f_l, f_r, fileNum, r_l, r_r_1, t_s_1)

    phi = 0
    globalDeltaX = []
    globalDeltaY = []
    globalLoc = np.zeros((2, numParticles))
    Cycle1(Noise_bound, avg, globalDeltaX, globalDeltaY, globalLoc, l, lidar_data, numParticles, phi, r, width)
    fig, ax = plt.subplots()
    for x in range(numParticles):
        # Change line color to red
        ax.plot([a[x] for a in globalDeltaX], [a[x] for a in globalDeltaY], color='red')
        # Change axis colors
        ax.spines['bottom'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['left'].set_color('green')
        ax.spines['right'].set_color('green')
        # Change axis labels
        ax.set_xlabel('X Axis', fontsize=14, color='purple')
        ax.set_ylabel('Y Axis', fontsize=14, color='purple')
        # Change background color
        ax.set_facecolor('lightgray')
    plt.show()


def Pre_deal(avg, f_l, f_r, fileNum, r_l, r_r_1, t_s_1):
    lidar = get_lidar("data/" + "Hokuyo" + str(fileNum))
    closest_lidar = [lidar[k] for k in
                     np1.argmin(
                         np1.abs(np1.array(t_s_1).reshape(1, -1) - np1.array([l3['t'] for l3 in lidar]).reshape(-1, 1)),
                         axis=0)]
    result = f_r, f_l, r_r_1, r_l, t_s_1, closest_lidar
    r1, l1, r2, l2, ts, lidar_data = result
    r = (r1 + r2)
    r = r / 2
    l = (l1 + l2)
    l = l / 2
    r = np.sum(r[:(r.size // avg) * avg].reshape(-1, avg), axis=1)
    l = np.sum(l[:(l.size // avg) * avg].reshape(-1, avg), axis=1)
    return l, lidar_data, r


def Cycle1(Noise_bound, avg, globalDeltaX, globalDeltaY, globalLoc, l, lidar_data, numParticles, phi, r, width):
    for i, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::avg])):
        phi += np.random.uniform(-Noise_bound, Noise_bound, numParticles).reshape(1, numParticles)
        # calculate and update values based on the slides (theta, delta x and y, R, phi)
        current_theta = (rr - ll) / width
        delta = np.array(
            [(rr + ll) * np.cos(current_theta / 2) / 2, (rr + ll) * np.sin(current_theta / 2) / 2]).reshape(-1,
                                                                                                            1) * .254 * np.pi / 360
        R_matrix = np.array([[np.cos(phi), - np.sin(phi)], [np.sin(phi), np.cos(phi)]]).reshape(2, 2, numParticles)
        phi += current_theta / 2
        globa = (R_matrix * delta.reshape(1, 2, 1)).sum(axis=1).reshape(2, numParticles)
        globalLoc += globa
        loc_x = globalLoc[0, :]
        loc_y = globalLoc[1, :]
        globalDeltaX.append(loc_x.flatten())
        globalDeltaY.append(loc_y.flatten())


show_paths(160, 80, 80, 4, 20, 10, 0.015,10)
