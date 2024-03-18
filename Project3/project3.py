from load_data import *
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns
from MapUtils.MapUtils import getMapCellsFromRay


# initial visualization indicats that motion of right wheels are approximately the same, same goes for left wheels
def load_data(num, data_path="../Data/ECE5242Proj3-train/"):
    fr, fl, rr, rl, ts = get_encoder(data_path + "Encoders" + str(num))
    lidar = get_lidar(data_path + "Hokuyo" + str(num))
    closest_lidar = [lidar[k] for k in
                     np.argmin(np.abs(np.array(ts).reshape(1, -1) - np.array([l['t'] for l in lidar]).reshape(-1, 1)),
                               axis=0)]
    return fr, fl, rr, rl, ts, closest_lidar


def track_movement(r, l, w, averaging=1):
    r = np.mean(r[:(r.size // averaging) * averaging].reshape(-1, averaging), axis=1)
    l = np.mean(l[:(l.size // averaging) * averaging].reshape(-1, averaging), axis=1)

    theta = (r - l) / w
    delta_y = (r + l) * np.cos(theta / 2) / 2
    delta_x = (r + l) * np.sin(theta / 2) / 2
    plt.plot(np.cumsum(delta_x), np.cumsum(delta_y))
    plt.show()


def track_global_movement(r, l, lidar_data, w, averaging=1):
    r = np.mean(r[:(r.size // averaging) * averaging].reshape(-1, averaging), axis=1)
    l = np.mean(l[:(l.size // averaging) * averaging].reshape(-1, averaging), axis=1)

    phi = 0
    global_delta_x, global_delta_y = [], []
    for index, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::averaging])):
        current_theta = (rr - ll) / w
        delta = np.array(
            [(rr + ll) * np.cos(current_theta / 2) / 2, (rr + ll) * np.sin(current_theta / 2) / 2]).reshape(-1,
                                                                                                            1) * .254 * np.pi / 360
        R_matrix = np.array([[np.cos(phi), - np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        phi += current_theta / 2
        globa = (R_matrix @ delta).flatten()
        global_delta_x.append(globa[0])
        global_delta_y.append(globa[1])
        # xs = []
        # ys = []
        # for rad, dist in zip(scans['angle'].flatten(), scans['scan'].flatten()):
        #     xs.append(globa[0] + np.cos(phi + rad) * dist)
        #     ys.append(globa[1] + np.sin(phi + rad) * dist)
        # if index % 200 == 0:
        angles = scans['angle'].flatten()
        loc_x = np.sum(np.array(global_delta_x))
        loc_y = np.sum(np.array(global_delta_y))
        plt.scatter(loc_x + np.cos(phi + angles) * scans['scan'], loc_y + np.sin(phi + angles) * scans['scan'], s=5)
    plt.show()

    plt.plot(np.cumsum(np.array(global_delta_x)), np.cumsum(np.array(global_delta_y)))
    plt.title(str(w))
    plt.show()


# check results with trying phi += current_theta an dupdate width
# try hyperparam to cap the max/min value
# many robots by adding small amount of noise to phi

def track_global_indices(w, x_offset, y_offset, x_range, y_range, indices_per_int, file_num, averaging=1):
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
    for index, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::averaging])):
        # calculate and update values based on the slides (theta, delta x and y, R, phi)
        current_theta = (rr - ll) / w
        delta = np.array(
            [(rr + ll) * np.cos(current_theta / 2) / 2, (rr + ll) * np.sin(current_theta / 2) / 2]).reshape(-1,
                                                                                                            1) * .254 * np.pi / 360
        R_matrix = np.array([[np.cos(phi), - np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        phi += current_theta / 2
        globa = (R_matrix @ delta).flatten()
        global_delta_x.append(globa[0])
        global_delta_y.append(globa[1])
        angles = scans['angle'].flatten()
        # update global location of x based on sum of previous deltas (could optimize further easily)
        loc_x = np.sum(np.array(global_delta_x))
        loc_y = np.sum(np.array(global_delta_y))
        # map scanned (wall) values to grid
        scan_x_list = np.rint(((loc_x + np.cos(phi + angles) * scans['scan']) + x_offset) * indices_per_int)
        scan_y_list = np.rint(((loc_y + np.sin(phi + angles) * scans['scan']) + y_offset) * indices_per_int)
        # map current location to grid
        loc_x_index = np.rint((loc_x + x_offset) * indices_per_int)
        loc_y_index = np.rint((loc_y + y_offset) * indices_per_int)
        # insert grid locations of walls into Wx2 array
        walls = np.array([scan_x_list.astype(int), scan_y_list.astype(int)]).T.reshape(-1, 2)
        # get grid location of all gaps for all walls
        gaps = getMapCellsFromRay(loc_x_index, loc_y_index, scan_x_list, scan_y_list, 0).T.astype(int).reshape(-1, 2)
        # update indices values with (hyperparameters) based on gap or wall presence
        indices[walls[:, 0], walls[:, 1]] += 2.2
        indices[gaps[:, 0], gaps[:, 1]] -= 0.7

        # update freqs array (trying this out) - if low freq then random wall scans avoided
        freqs[walls[:, 0], walls[:, 1]] += 1
        freqs[gaps[:, 0], gaps[:, 1]] += 1
        # sns.heatmap(np.log(indices - np.min(indices) + 1))
        # plt.pause(0.01)

        # indices[int(loc_x_index), int(loc_y_index)] -= 500
        if index % 500 == 0:
            print(index)
            #     sns.heatmap(np.log(indices - np.min(indices) + 1))
        #     plt.show()
        # plt.scatter(scan_x_list, scan_y_list, s=5)
    sns.heatmap(indices)
    plt.show()
    # plt.show()
    # print(indices)
    # #plt.imshow(indices, cmap='hot')
    # ax = sns.heatmap(np.log(np.maximum(0.5, freqs * indices)))
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # plt.savefig("results/img" + str(file_num))
    # plt.show()
    # ax = sns.heatmap(indices > 0)
    # plt.show()
    # ax = sns.heatmap(indices < -10)
    # plt.show()

    plt.plot(np.cumsum(np.array(global_delta_x)), np.cumsum(np.array(global_delta_y)))
    plt.show()


# try hyperparam to cap the max/min value

def track_global_indices_SLAM(w, x_offset, y_offset, x_range, y_range, indices_per_int, file_num, data_path,
                              num_particles, max_noise, update_threshold, subtract, min_max_cap=1000, averaging=1,
                              testing=False):
    # average right side and left side encoding data and lidar data to match length
    r1, l1, r2, l2, ts, lidar_data = load_data(file_num, data_path=data_path)
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
    particle_track = np.zeros((num_particles))
    for index, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::averaging])):
        phi += np.random.uniform(-max_noise / np.sqrt(index + 1), max_noise / np.sqrt(index + 1),
                                 num_particles).reshape(1, num_particles)
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
        # print(global_delta_x)
        global_delta_x.append(globa[0])
        global_delta_y.append(globa[1])
        angles = scans['angle'].reshape(-1, 1)
        # map scanned (wall) values to grid
        scan_x_list = np.rint(
            ((loc_x + np.cos(phi + angles) * scans['scan'].reshape(-1, 1)) + x_offset) * indices_per_int)
        scan_y_list = np.rint(
            ((loc_y + np.sin(phi + angles) * scans['scan'].reshape(-1, 1)) + y_offset) * indices_per_int)
        # map current location to grid
        loc_x_index = np.rint((loc_x + x_offset) * indices_per_int)
        loc_y_index = np.rint((loc_y + y_offset) * indices_per_int)

        max_walls = -np.inf
        for i in range(num_particles):
            # insert grid locations of walls into Wx2 array
            walls = np.array([scan_x_list[:, i].astype(int), scan_y_list[:, i].astype(int)]).T.reshape(-1, 2)
            # get grid location of all gaps for all walls
            # gaps = getMapCellsFromRay(loc_x_index[i], loc_y_index[i], scan_x_list[:, i], scan_y_list[:, i], 0).T.astype(int).reshape(-1, 2)
            current_walls = np.sum(indices[walls[:, 0], walls[:, 1]])
            temp_indices = np.zeros_like(indices)
            temp_indices[walls[:, 0], walls[:, 1]] += 0.9
            particle_track[i] += current_walls
            # temp_indices[gaps[:, 0], gaps[:, 1]] -= 0.7
            # sns.heatmap(temp_indices)
            # plt.title(str(i))
            # plt.show()
            if max_walls < current_walls:
                max_walls = current_walls
                best_walls = walls
                # best_gaps = gaps
                best_robot = i
        # print(np.where(np.argsort(particle_track) == best_robot))
        best_gaps = getMapCellsFromRay(loc_x_index[best_robot], loc_y_index[best_robot], scan_x_list[:, best_robot],
                                       scan_y_list[:, best_robot], 0).T.astype(int).reshape(-1, 2)
        # temp_indices = np.zeros_like(indices)
        # temp_indices[walls[:, 0], walls[:, 1]] += 0.9
        # temp_indices[gaps[:, 0], gaps[:, 1]] -= 0.7
        # current_norm = np.linalg.norm(indices - temp_indices)
        # if min_norm > current_norm:
        #     min_norm = current_norm
        #     best_index = i
        #     best_walls = walls
        #     best_gaps = gaps
        temp_indices = np.zeros_like(indices)
        temp_indices[best_walls[:, 0], best_walls[:, 1]] += 0.9
        temp_indices[best_gaps[:, 0], best_gaps[:, 1]] -= subtract
        # sns.heatmap(temp_indices)
        # plt.show()
        # if index > 10 and np.where(np.argsort(particle_track) == best_robot)[0] < 10:
        #     display_indices(indices)
        #     display_indices(temp_indices)

        if (max_walls > update_threshold) or index < 10:
            indices[best_walls[:, 0], best_walls[:, 1]] += 0.9
            indices[best_gaps[:, 0], best_gaps[:, 1]] -= subtract
            # print(max_walls)
        # print(best_robot)
        # sns.heatmap(indices)
        # plt.title('current')
        # plt.show()

        # update indices values with (hyperparameters) based on gap or wall presence
        # indices[walls[:, 0], walls[:, 1]] += 2.2
        # indices[gaps[:, 0], gaps[:, 1]] -= 0.7
        indices = np.clip(indices, -min_max_cap, min_max_cap)

        # #indices[int(loc_x_index), int(loc_y_index)] -= 500
        # if index % 50 == 0:
        #    display_indices(indices)
        # plt.scatter(scan_x_list, scan_y_list, s=5)
    if testing:
        display_indices(indices, filename="results/output" + str(file_num))
    else:
        display_indices(indices, filename="results/SLAM_img" + str(file_num) + "_" + str(int(10 * w)) + "_" + str(
            min_max_cap) + "_" + str(int(10 * subtract)) + "_" + str(np.abs(update_threshold)) + "_" + str(
            int(1000 * max_noise)))
    # plt.show()
    # print(indices)
    # #plt.imshow(indices, cmap='hot')
    # ax = sns.heatmap(np.log(np.maximum(0.5, freqs * indices)))
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # plt.savefig("results/img" + str(file_num))
    # plt.show()
    # ax = sns.heatmap(indices > 0)
    # plt.show()
    # ax = sns.heatmap(indices < -10)
    # plt.show()


def track_global_indices_SLAMSLAM(w, x_offset, y_offset, x_range, y_range, indices_per_int, file_num, data_path,
                                  num_particles, max_noise, update_threshold, subtract, min_max_cap=1000, averaging=1,
                                  testing=False):
    # average right side and left side encoding data and lidar data to match length
    r1, l1, r2, l2, ts, lidar_data = load_data(file_num, data_path=data_path)
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
    particle_track = np.zeros((num_particles))
    particle_weights = np.zeros((num_particles)) + (1 / num_particles)
    particle_corrs = np.zeros((num_particles))
    for index, (rr, ll, scans) in enumerate(zip(r, l, lidar_data[::averaging])):
        phi += np.random.uniform(-max_noise / np.sqrt(index + 1), max_noise / np.sqrt(index + 1),
                                 num_particles).reshape(1, num_particles)
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
        # print(global_delta_x)
        global_delta_x.append(globa[0])
        global_delta_y.append(globa[1])
        angles = scans['angle'].reshape(-1, 1)
        # map scanned (wall) values to grid
        scan_x_list = np.rint(
            ((loc_x + np.cos(phi + angles) * scans['scan'].reshape(-1, 1)) + x_offset) * indices_per_int)
        scan_y_list = np.rint(
            ((loc_y + np.sin(phi + angles) * scans['scan'].reshape(-1, 1)) + y_offset) * indices_per_int)
        # map current location to grid
        loc_x_index = np.rint((loc_x + x_offset) * indices_per_int)
        loc_y_index = np.rint((loc_y + y_offset) * indices_per_int)

        max_walls = -np.inf
        for i in range(num_particles):
            # insert grid locations of walls into Wx2 array
            walls = np.array([scan_x_list[:, i].astype(int), scan_y_list[:, i].astype(int)]).T.reshape(-1, 2)
            # get grid location of all gaps for all walls
            # gaps = getMapCellsFromRay(loc_x_index[i], loc_y_index[i], scan_x_list[:, i], scan_y_list[:, i], 0).T.astype(int).reshape(-1, 2)
            current_walls = np.sum(indices[walls[:, 0], walls[:, 1]])
            particle_corrs[i] = np.log(current_walls + 0.02)
            temp_indices = np.zeros_like(indices)
            temp_indices[walls[:, 0], walls[:, 1]] += 0.9
            particle_track[i] += current_walls
            # temp_indices[gaps[:, 0], gaps[:, 1]] -= 0.7
            # sns.heatmap(temp_indices)
            # plt.title(str(i))
            # plt.show()
            if max_walls < current_walls:
                max_walls = current_walls
                best_walls = walls
                # best_gaps = gaps
                best_robot = i
        print(np.min(np.abs(particle_corrs)) / 2, particle_corrs)
        particle_corrs = np.maximum(np.min(np.abs(particle_corrs)) / 2, particle_corrs)
        print(particle_corrs)
        particle_weights = particle_weights * particle_corrs
        particle_weights = particle_weights / np.sum(particle_weights)
        print(particle_weights)
        best_robot = np.random.choice(num_particles, p=particle_weights)
        print(np.where(np.argsort(particle_track) == best_robot))
        best_gaps = getMapCellsFromRay(loc_x_index[best_robot], loc_y_index[best_robot], scan_x_list[:, best_robot],
                                       scan_y_list[:, best_robot], 0).T.astype(int).reshape(-1, 2)
        # temp_indices = np.zeros_like(indices)
        # temp_indices[walls[:, 0], walls[:, 1]] += 0.9
        # temp_indices[gaps[:, 0], gaps[:, 1]] -= 0.7
        # current_norm = np.linalg.norm(indices - temp_indices)
        # if min_norm > current_norm:
        #     min_norm = current_norm
        #     best_index = i
        #     best_walls = walls
        #     best_gaps = gaps
        temp_indices = np.zeros_like(indices)
        temp_indices[best_walls[:, 0], best_walls[:, 1]] += 0.9
        temp_indices[best_gaps[:, 0], best_gaps[:, 1]] -= subtract
        # sns.heatmap(temp_indices)
        # plt.show()
        # if index > 10 and np.where(np.argsort(particle_track) == best_robot)[0] < 10:
        #     display_indices(indices)
        #     display_indices(temp_indices)

        if (max_walls > update_threshold) or index < 10:
            indices[best_walls[:, 0], best_walls[:, 1]] += 0.9
            indices[best_gaps[:, 0], best_gaps[:, 1]] -= subtract
            # print(max_walls)
        # print(best_robot)
        # sns.heatmap(indices)
        # plt.title('current')
        # plt.show()

        # update indices values with (hyperparameters) based on gap or wall presence
        # indices[walls[:, 0], walls[:, 1]] += 2.2
        # indices[gaps[:, 0], gaps[:, 1]] -= 0.7
        indices = np.clip(indices, -min_max_cap, min_max_cap)

        # #indices[int(loc_x_index), int(loc_y_index)] -= 500
        # if index % 50 == 0:
        #    display_indices(indices)
        # plt.scatter(scan_x_list, scan_y_list, s=5)
    if testing:
        display_indices(indices, filename="results/final" + str(file_num))
    else:
        display_indices(indices, filename="results/SLAM_img" + str(file_num) + "_" + str(int(10 * w)) + "_" + str(
            min_max_cap) + "_" + str(int(10 * subtract)) + "_" + str(np.abs(update_threshold)) + "_" + str(
            int(1000 * max_noise)))
    # plt.show()
    # print(indices)
    # #plt.imshow(indices, cmap='hot')
    # ax = sns.heatmap(np.log(np.maximum(0.5, freqs * indices)))
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # plt.savefig("results/img" + str(file_num))
    # plt.show()
    # ax = sns.heatmap(indices > 0)
    # plt.show()
    # ax = sns.heatmap(indices < -10)
    # plt.show()


def display_indices(indices, filename=False):
    print(filename)
    plt.figure()
    xmin, xmax, ymin, ymax = 0, -1, 0, -1
    while np.sum(indices[xmin, :]) == 0:
        xmin += 1
    while np.sum(indices[xmax, :]) == 0:
        xmax -= 1
    while np.sum(indices[:, ymin]) == 0:
        ymin += 1
    while np.sum(indices[:, ymax]) == 0:
        ymax -= 1
    sns.heatmap(indices[xmin:xmax, ymin:ymax])
    if filename:
        plt.savefig(filename)
    plt.show()


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


# for abc in range(155, 165):
# for x in range(162, 168):
# track_global_movement(fr, fl, lidar[:len(ts)], 165.5)
# track_global_indices(165.5, 10, 10, 40, 40, 6, 20)
# track_global_indices(165.5, 15, 15, 45, 50, 6, 21)
# track_global_indices(165.5, 10, 35, 40, 55, 6, 23)

# track_global_indices_SLAM(165.5, 40, 40, 80, 80, 4, 20, 100, 0.01, min_max_cap=10, averaging=10)
# show_paths(165.5, 40, 40, 80, 80, 4, 20, 10, 0.015, min_max_cap=100, averaging=20)
# for width in [160, 165.5, 170]:
data_path = "data/"
width = 165.5
# show_paths(width, 45, 45, 90, 90, 4, 20, 50, 0.001)
for filenum in [20, 21, 23]:
    for thresh in [0]:
        for angle in [0.001]:
            for min_max in [25]:  # check smaller?
                for sub in [0.4]:  # [0.3,0.4,0.5]: # check smaller?
                    track_global_indices_SLAM(width, 45, 45, 90, 90, 4, filenum, data_path, 50, angle, thresh, sub,
                                              min_max_cap=min_max, averaging=4, testing=True)

