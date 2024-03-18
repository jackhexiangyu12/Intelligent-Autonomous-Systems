from load_data import *
import matplotlib.pyplot as plt


# initial visualization indicats that motion of right wheels are approximately the same, same goes for left wheels
def load_data(num, data_path="data/"):
    fr, fl, rr, rl, ts = get_encoder(data_path + "Encoders" + str(num))
    lidar = get_lidar(data_path + "Hokuyo" + str(num))
    closest_lidar = [lidar[k] for k in
                     np.argmin(np.abs(np.array(ts).reshape(1, -1) - np.array([l['t'] for l in lidar]).reshape(-1, 1)),
                               axis=0)]
    return fr, fl, rr, rl, ts, closest_lidar


def show_paths(w, x_offset, y_offset, x_range, y_range, indices_per_int, file_num, num_particles, max_noise,
               min_max_cap=1000, averaging=1):
    # average right side and left side encoding data and lidar data to match length
    r1, l1, r2, l2, ts, lidar_data = load_data(file_num)
    r = (r1 + r2) / 2
    l = (l1 + l2) / 2
    # initialize grid based on hyperparameters
    indices = np.zeros((x_range * indices_per_int, y_range * indices_per_int))
    # track frequency of grid location (either wall or gap)
    freqs = np.zeros((x_range * indices_per_int, y_range * indices_per_int))
    # condense encodings to speed up operation if indicated in hyperparameters
    r = np.sum(r[:(r.size // averaging) * averaging].reshape(-1, averaging), axis=1)
    l = np.sum(l[:(l.size // averaging) * averaging].reshape(-1, averaging), axis=1)

    phi = 0
    global_delta_x, global_delta_y = [], []
    global_loc = np.zeros((2, num_particles))
    for index, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::averaging])):
        phi += np.random.uniform(-max_noise, max_noise, num_particles).reshape(1, num_particles)
        # calculate and update values based on the slides (theta, delta x and y, R, phi)
        current_theta = (rr - ll) / w
        delta = np.array(
            [(rr + ll) * np.cos(current_theta / 2) / 2, (rr + ll) * np.sin(current_theta / 2) / 2]).reshape(-1,
                                                                                                            1) * .254 * np.pi / 360
        R_matrix = np.array([[np.cos(phi), - np.sin(phi)], [np.sin(phi), np.cos(phi)]]).reshape(2, 2, num_particles)
        phi += current_theta / 2
        globa = (R_matrix * delta.reshape(1, 2, 1)).sum(axis=1).reshape(2, num_particles)
        global_loc += globa
        loc_x = global_loc[0, :]
        loc_y = global_loc[1, :]
        global_delta_x.append(loc_x.flatten())
        global_delta_y.append(loc_y.flatten())
    # plt.plot([a[0] for a in global_delta_x], [a[0] for a in global_delta_y])
    for x in range(num_particles):
        plt.plot([a[x] for a in global_delta_x], [a[x] for a in global_delta_y])
    plt.show()


show_paths(165.5, 40, 40, 80, 80, 4, 20, 10, 0.015, min_max_cap=100, averaging=20)

# for width in [160, 165.5, 170]:
#     data_path = "data/"
#     # width = 165.5
#     show_paths(width, 45, 45, 90, 90, 4, 20, 50, 0.001)
#
# for filenum in [20, 21, 23]:
#     for thresh in [0]:
#         for angle in [0.001]:
#             for min_max in [25]:  # check smaller?
#                 for sub in [0.4]:  # [0.3,0.4,0.5]: # check smaller?
#                     show_paths(width, 45, 45, 90, 90, 4, filenum, 50, 0.001, min_max_cap=100, averaging=20)
#
