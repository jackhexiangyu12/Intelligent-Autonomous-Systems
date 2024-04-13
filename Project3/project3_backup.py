import numpy as np1
import numpy as np2
import matplotlib.pyplot as plt
import seaborn as sns
from MapUtils.MapUtils import getMapCellsFromRay
from load_data import *

def load_data(num, data_path="../Data/ECE5242Proj3-train/"):
    fr, fl, rr, rl, ts = get_encoder(data_path + "Encoders" + str(num))
    lidar = get_lidar(data_path + "Hokuyo" + str(num))
    closest_lidar = [lidar[k] for k in
                     np.argmin(np.abs(np.array(ts).reshape(1, -1) - np.array([l['t'] for l in lidar]).reshape(-1, 1)),
                               axis=0)]
    return fr, fl, rr, rl, ts, closest_lidar


def SLAM(w, x_o, y_o, x_r, y_r, ind, f_n, d_p, p, m_n, u_t, s, min=1000, avg=1, t=False):
    r1, l1, r2, l2, ts, lidar_data = load_data(f_n, data_path=d_p)
    r = (r1 + r2) / 2
    l = (l1 + l2) / 2
    indices = np.zeros((x_r * ind, y_r * ind))
    r = np.sum(r[:(r.size // avg) * avg].reshape(-1, avg), axis=1)
    l = np.sum(l[:(l.size // avg) * avg].reshape(-1, avg), axis=1)
    phi = 0
    global_delta_x, global_delta_y = [], []
    global_loc = np.zeros((2, p))
    particle_track = np.zeros((p))
    for index, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::avg])):
        phi += np.random.uniform(-m_n / np.sqrt(index + 1), m_n / np.sqrt(index + 1),
                                 p).reshape(1, p)
        current_theta = (rr - ll) / w
        delta = np.array(
            [(rr + ll) * np.cos(current_theta / 2) / 2, (rr + ll) * np.sin(current_theta / 2) / 2]).reshape(-1,
                                                                                                            1) * .254 * np.pi / 360
        R_matrix = np.array([[np.cos(phi), - np.sin(phi)], [np.sin(phi), np.cos(phi)]]).reshape(2, 2, p)
        phi += current_theta / 2
        globa = (R_matrix * delta.reshape(1, 2, 1)).sum(axis=1).reshape(2, p)
        global_loc += globa
        loc_x = global_loc[0, :]
        loc_y = global_loc[1, :]
        global_delta_x.append(loc_x.flatten())
        global_delta_y.append(loc_y.flatten())
        global_delta_x.append(globa[0])
        global_delta_y.append(globa[1])
        angles = scans['angle'].reshape(-1, 1)
        scan_x_list = np.rint(
            ((loc_x + np.cos(phi + angles) * scans['scan'].reshape(-1, 1)) + x_o) * ind)
        scan_y_list = np.rint(
            ((loc_y + np.sin(phi + angles) * scans['scan'].reshape(-1, 1)) + y_o) * ind)
        loc_x_index = np.rint((loc_x + x_o) * ind)
        loc_y_index = np.rint((loc_y + y_o) * ind)

        max_walls = -np.inf
        for i in range(p):
            walls = np.array([scan_x_list[:, i].astype(int), scan_y_list[:, i].astype(int)]).T.reshape(-1, 2)
            current_walls = np.sum(indices[walls[:, 0], walls[:, 1]])
            temp_indices = np.zeros_like(indices)
            temp_indices[walls[:, 0], walls[:, 1]] += 0.9
            particle_track[i] += current_walls
            if max_walls < current_walls:
                max_walls = current_walls
                best_walls = walls
                best_robot = i
        best_gaps = getMapCellsFromRay(loc_x_index[best_robot], loc_y_index[best_robot], scan_x_list[:, best_robot],
                                       scan_y_list[:, best_robot], 0).T.astype(int).reshape(-1, 2)
        temp_indices = np.zeros_like(indices)
        temp_indices[best_walls[:, 0], best_walls[:, 1]] += 0.9
        temp_indices[best_gaps[:, 0], best_gaps[:, 1]] -= s
        if (max_walls > u_t) or index < 10:
            indices[best_walls[:, 0], best_walls[:, 1]] += 0.9
            indices[best_gaps[:, 0], best_gaps[:, 1]] -= s
        indices = np.clip(indices, -min, min)
    if t:
        filename = "res/o" + str(f_n)
        print(filename)
        plt.figure()
        xmin, xmax, ymin, ymax = 0, -1, 0, -1
        while np1.sum(indices[xmin, :]) == 0:
            xmin += 1
        while np1.sum(indices[xmax, :]) == 0:
            xmax -= 1
        while np1.sum(indices[:, ymin]) == 0:
            ymin += 1
        while np1.sum(indices[:, ymax]) == 0:
            ymax -= 1
        sns.heatmap(indices[xmin:xmax, ymin:ymax])
        if filename:
            plt.savefig(filename)
        plt.show()
    else:
        filename1 = "res/SLAM_" + str(f_n) + "_" + str(int(10 * w)) + "_" + str(
            min) + "_" + str(int(10 * s)) + "_" + str(np.abs(u_t)) + "_" + str(
            int(1000 * m_n))
        print(filename1)
        plt.figure()
        xmin1, xmax1, ymin1, ymax1 = 0, -1, 0, -1
        while np2.sum(indices[xmin1, :]) == 0:
            xmin1 += 1
        while np2.sum(indices[xmax1, :]) == 0:
            xmax1 -= 1
        while np2.sum(indices[:, ymin1]) == 0:
            ymin1 += 1
        while np2.sum(indices[:, ymax1]) == 0:
            ymax1 -= 1
        sns.heatmap(indices[xmin1:xmax1, ymin1:ymax1])
        if filename1:
            plt.savefig(filename1)
        plt.show()


def show_paths(width, X_area, Y_area, indicesPerInt, fileNum, numParticles, Noise_bound, avg):
    f_r, f_l, r_r_1, r_l, t_s_1 = get_encoder("data/" + "Encoders" + str(fileNum))
    l, lidar_data, r = Pre_deal(avg, f_l, f_r, fileNum, r_l, r_r_1, t_s_1)

    phi = 0
    globalDeltaX = []
    globalDeltaY = []
    globalLoc = np.zeros((2, numParticles))
    Cycle1(Noise_bound, avg, globalDeltaX, globalDeltaY, globalLoc, l, lidar_data, numParticles, phi, r, width)

    for x in range(numParticles):
        fig, ax = plt.subplots()
        ax.plot([a[x] for a in globalDeltaX], [a[x] for a in globalDeltaY], color='red')
        ax.spines['bottom'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['left'].set_color('green')
        ax.spines['right'].set_color('green')
        ax.set_xlabel('X Axis', fontsize=14, color='purple')
        ax.set_ylabel('Y Axis', fontsize=14, color='purple')
        ax.set_facecolor('lightgray')
        plt.show()


def Pre_deal(avg, f_l, f_r, fileNum, r_l, r_r_1, t_s_1):
    lidar = get_lidar("data/" + "Hokuyo" + str(fileNum))
    closest_lidar = [lidar[k] for k in
                     np.argmin(
                         np.abs(np.array(t_s_1).reshape(1, -1) - np.array([l3['t'] for l3 in lidar]).reshape(-1, 1)),
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

data_path = "data/"
width = 165.5
for filenum in [20, 21, 23]:
    for angle in [0.001]:
        for thresh in [0]:
            for sub in [0.4, 0.5, 0.6, 0.7]:
                for min_max in [15,20,25]:
                    SLAM(width, 40, 40, 85, 85, 4, filenum, data_path, 45, angle, thresh, sub,
                         min=min_max, avg=4, t=False)
