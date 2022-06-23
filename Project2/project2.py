import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


def kmeans_stuff(path, k=50):
    #combine text files into array and drop timestamps
    train = np.concatenate([np.loadtxt(path + file)[:, 1:] for file in os.listdir(path)], axis=0)

    #create dataframe from array with randomly assigned cluster
    train_df = pd.DataFrame(train, columns=["ax", "ay", "az", "gx", "gy", "gz"])
    centroids = train[np.random.randint(train.shape[0], size=k)]
    prev_centroids = centroids - 1

    #get mean,std and normalize
    train_mean, train_std = np.mean(train, axis=0), np.std(train, axis=0)
    train_normal = (train.reshape(-1, 1, 6) - train_mean) / train_std

    while np.linalg.norm(centroids - prev_centroids):
        #update centroids
        prev_centroids = centroids
        train_df['class'] = np.argmin(np.linalg.norm(train_normal - (centroids - train_mean) / train_std, axis=2), axis=1)
        centroids = np.array([np.mean(train[train_df.index[train_df['class'] == i].tolist()], axis=0) for i in range(k)])

    return centroids, train_mean, train_std


def plot_sample(path, centroids, train_mean, train_std):
    norm_input = (np.loadtxt(path)[:, 1:] - train_mean) / train_std
    norm_centroids = (centroids - train_mean) / train_std
    assignments = np.argmin(np.linalg.norm(norm_input.reshape(-1, 1, 6) - norm_centroids, axis=2), axis=1)

    #plt.plot(assignments)
    #plt.show()
    return assignments


def train(clusters, k_states=20, k_clusters=50, lmax=10):
    #initialize stuff
    T = clusters.size

    # pi - random initializing by getting random values and dividing by sum to add to 1
    pi = np.abs(np.random.rand(k_states))
    pi = pi / np.sum(pi)

    # B - random initilizing and normalizing
    B = np.abs(np.random.rand(k_states, k_clusters))
    B = B / B.sum(axis=1).reshape(-1, 1)

    # A - random initializing?
    A = np.abs(np.random.rand(k_states, k_states)) + np.eye(k_states)
    A = A / A.sum(axis=1).reshape(-1, 1)

    # set correct sizes for alpha and beta, and epsilon
    alpha = np.zeros((k_states, T))
    beta = np.zeros((k_states, T))
    epsilon = np.zeros((k_states, k_states, T - 1))

    alpha_factor, beta_factor = 0, 0
    afs, bfs = [], []

    for l in range(lmax):
        alpha[:, 0] = (pi * B[:, clusters[0]]).flatten()
        beta[:, -1] = 1
        alpha_factor, beta_factor = 0, 0
        for t in range(1, T):
            alpha[:, t] = (alpha[:, t - 1].reshape(1, -1) @ A).flatten() * B[:, clusters[t]]
            beta[:, -1 - t] = (A @ (B[:, clusters[-t]] * beta[:, -t]).reshape(-1, 1)).flatten()

            alpha_sum = np.sum(alpha[:, t])
            beta_sum = np.sum(beta[:, - 1 - t])
            alpha[:, t] /= alpha_sum
            beta[:, - 1 - t] /= beta_sum
            alpha_factor += np.log(alpha_sum)
            beta_factor += np.log(beta_sum)
        afs.append(alpha_factor)
        bfs.append(beta_factor)

        # E step
        epsilon = alpha[:, :-1].reshape(k_states, 1, T - 1) * beta[:, 1:].reshape(1, k_states, T - 1) * A.reshape(k_states, k_states, 1) * B[:, clusters[1:]].reshape(1, k_states, T - 1)
        epsilon /= np.sum(np.sum(epsilon, axis=0), axis=0)

        gamma = (alpha * beta) / np.sum(alpha * beta, axis=0)

        # M step
        pi = (gamma[:, 0] + 0.01) / (np.sum(0.01 + gamma[:, 0])).flatten()
        A = np.sum(epsilon, axis=-1) / np.sum(gamma, axis=-1).reshape(-1, 1) + 1e-8
        for vk in range(k_clusters):
            B[:, vk] = np.sum(gamma[:, np.where(clusters == vk)], axis=-1).flatten() / np.sum(gamma, axis=-1).flatten() + 1e-8
        assert alpha.shape == (k_states, T)
        assert beta.shape == (k_states, T)
        assert gamma.shape == (k_states, T)
        assert pi.shape == (k_states, )
        assert A.shape == (k_states, k_states)
        assert epsilon.shape == (k_states, k_states, T - 1)
        assert B.shape == (k_states, k_clusters)
    # plt.figure()
    # plt.plot([l1 + 1 for l1 in range(lmax)], afs, label="alpha log sum")
    # plt.plot([l1 + 1 for l1 in range(lmax)], bfs, label="beta log sum")
    # plt.legend()
    #plt.show()

    return A, B, pi


def test(clusters, A, B, pi, k_states=20):
    alpha_factor, beta_factor = 0,0
    T = clusters.size
    alpha = np.zeros((k_states, T))
    beta = np.zeros((k_states, T))
    alpha[:, 0] = (pi * B[:, clusters[0]]).flatten()
    beta[:, -1] = 1
    for t in range(1, T):
        alpha[:, t] = (alpha[:, t - 1].reshape(1, -1) @ A).flatten() * B[:, clusters[t]]
        beta[:, -1 - t] = (A @ (B[:, clusters[-t]] * beta[:, -t]).reshape(-1, 1)).flatten()

        alpha_sum = np.sum(alpha[:, t])
        beta_sum = np.sum(beta[:, - 1 - t])
        alpha[:, t] /= alpha_sum
        beta[:, - 1 - t] /= beta_sum
        alpha_factor += np.log(alpha_sum)
        beta_factor += np.log(beta_sum)

    return alpha_factor, beta_factor 

    



# Questions
# how long do we loop

# eventually use time stamp instead of sample index

train_path = "../Data/ECE5242Proj2-train/"
test_path = "../Data/ECE5242Proj2-test/"
class_labels = ["beat3", "beat4", "circle", "eight", "inf", "wave"]
k_clusters = 50
k_states = 15

# c, m, s = kmeans_stuff("../Data/ECE5242Proj2-train/", k=k_clusters)
# with open('model/c.npy', 'wb') as f:
#     np.save(f, c)
# with open('model/m.npy', 'wb') as f:
#     np.save(f, m)
# with open('model/s.npy', 'wb') as f:
#     np.save(f, s)


# proj_dict = {}
# for current_label in class_labels:
#     print("Training " + current_label)
#     proj_dict[current_label] = {"pi": np.zeros((k_states)), "A": np.zeros((k_states, k_states)), "B": np.zeros((k_states, k_clusters)), "num_training_files": 0}
#     for current_file in os.listdir(train_path):
#         if current_label in current_file:
#             a, b, pi = train(plot_sample(train_path + current_file, c, m, s), k_states=k_states, k_clusters=k_clusters)
#             proj_dict[current_label]["pi"] += pi
#             proj_dict[current_label]["A"] += a
#             proj_dict[current_label]["B"] += b
#             proj_dict[current_label]["num_training_files"] += 1

#     if proj_dict[current_label]["num_training_files"]:
#         proj_dict[current_label]["pi"] /= proj_dict[current_label]["num_training_files"]
#         proj_dict[current_label]["A"] /= proj_dict[current_label]["num_training_files"]
#         proj_dict[current_label]["B"] /= proj_dict[current_label]["num_training_files"]
#         proj_dict[current_label]["pi"] = (proj_dict[current_label]["pi"] + 0.01) / (np.sum(proj_dict[current_label]["pi"] + 0.01))
#     else:
#         print("No training files for " + current_label)
# with open('model/model.pkl', 'wb') as handle:
#     pickle.dump(proj_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model/c.npy', 'rb') as f:
    c = np.load(f)
with open('model/m.npy', 'rb') as f:
    m = np.load(f)
with open('model/s.npy', 'rb') as f:
    s = np.load(f)

with open('model/model.pkl', 'rb') as handle:
    proj_dict = pickle.load(handle)



for test_file in os.listdir(test_path):
    current_test_clusters = plot_sample(test_path + test_file, c, m, s)
    alpha_dict = {}
    for current_class in proj_dict.keys():
        try:
            test_results = test(current_test_clusters, proj_dict[current_class]["A"], proj_dict[current_class]["B"], proj_dict[current_class]["pi"], k_states=k_states)
        except:
            test_results = (-np.inf, -np.inf)
        alpha_dict[current_class] = test_results[0]
    results = sorted(alpha_dict.items(), key=lambda item: np.abs(item[1]))
    #print(results)
    
    print("Prediction for " + test_file + ": " + results[0][0] + "  (second choice: " + results[1][0] + ", third choice: " + results[2][0] + ")")
