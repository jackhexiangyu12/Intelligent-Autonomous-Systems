from cmath import inf
from roipoly import RoiPoly
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from skimage.measure import regionprops, label
from pprint import pprint
import sys
import functools


# This code was used to gather dimensions of the height and width of barrel (in pixels) and saved in a training set with the actual distance (dist.npy)
def get_distance_train_set(path):
    distances = []
    for file in os.listdir(path):
        img2 = cv2.cvtColor(cv2.imread(path + file), cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img2)
        my_roi = RoiPoly(color='r') # draw new ROI in red color
        # plt.figure()
        # my_roi.display_roi()
        print(my_roi.x, my_roi.y)
        current = [int(file.split('.')[0]), np.abs(my_roi.y[0] - my_roi.y[1]), np.abs(my_roi.x[1] - my_roi.x[2])]
        print(current)
        distances.append(current)
    return distances
# dists = get_distance_train_set('./ECE5242Proj1-train/')
# with open('dist.npy', 'wb') as f:
#     np.save(f, np.array(dists))

#This code involved using roipoly to select pixels representing each barrel and save them to a training set (reds.npy)
def define_pixels(path):
    image_reds = []
    for file in os.listdir(path):
        img2 = cv2.cvtColor(cv2.imread(path + file), cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img2)
        my_roi = RoiPoly(color='r') # draw new ROI in red color
        # plt.figure()
        # my_roi.display_roi()
        mask = my_roi.get_mask(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))
        # plt.imshow(mask)
        # plt.show()
        # print(np.sum(mask))
        reds = mask.reshape(900,1200,1) * img2
        image_reds.append(reds[np.all(reds != [0, 0, 0], axis=-1)])
    return np.concatenate(image_reds, axis=0)

#This function compiles every single pixel from every single image in the training set and saves to all.npy
def get_all_pixels(path):
    image_pixels = []
    for file in os.listdir(path):
        img2 = cv2.cvtColor(cv2.imread(path + file), cv2.COLOR_BGR2RGB)
        image_pixels.append(img2)
    return np.concatenate(image_pixels, axis=0)

# extended_red = define_pixels('./ECE5242Proj1-train/').reshape(-1, 3)
# print(extended_red.shape)
# with open('extend_red.npy', 'wb') as f:
#     np.save(f, extended_red)

#Load all red pixels and all overall pixels
with open('extend_red.npy', 'rb') as f:
    all_reds = np.load(f)
with open('all.npy', 'rb') as f:
    all_pixs = np.load(f)

#convert red pixels to hsv
all_reds_hsv = cv2.cvtColor(all_reds.reshape(1, all_reds.shape[0], all_reds.shape[1]), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float64)
# plt.hist(all_reds_hsv[:, 0], bins=50)
# plt.show()
# plt.hist((all_reds_hsv[:, 0] + 90) % 180, bins=50)
# plt.show()
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
 
# # Creating plot
# ax.scatter3D(all_reds_hsv[:,0], all_reds_hsv[:,1], all_reds_hsv[:,2], color = "green")
# plt.xlabel('x')
# plt.title("simple 3D scatter plot")
 
# # show plot
# plt.show()

mu = all_reds.mean(axis=0)
mu_hsv = all_reds_hsv.mean(axis=0)
# print(mu, mu_hsv)

cov = np.cov(all_reds.T)
det = np.linalg.det(cov)
inv = np.linalg.inv(cov)
denom = np.sqrt(det * (2 * np.pi) ** 3)
cov_hsv = np.cov(all_reds_hsv.T)
det_hsv = np.linalg.det(cov_hsv)
inv_hsv = np.linalg.inv(cov_hsv)
denom_hsv = np.sqrt(det_hsv * (2 * np.pi) ** 3)
# print(cov, cov_hsv)
    
pr_y = all_reds.shape[0] / all_pixs.shape[0]

def get_pr_xy(pixel):
    pr_x = np.count_nonzero((all_pixs == pixel).all(axis = -1)) / all_pixs.shape[0]
    pr_yx = (1 / denom) * np.exp(-0.5 * (pixel - mu).T @ inv @ (pixel - mu))
    return pr_yx * pr_x / pr_y

def process_image(folder, filename, number, display=False, hsv=False, dilate=True, hybrid=False):
    if hsv:
        img2 = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2HSV)
        img2[:, 0] = (img2[:, 0] + 90) % 180
        sub = img2 - mu_hsv
        temp = sub.reshape(-1, 3) @ inv_hsv
        prod = temp.reshape(-1, 1, 3) @ sub.reshape(-1, 3, 1)
        prod = prod.reshape(sub.shape[0], sub.shape[1])
        if hybrid:
            img2 = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2RGB)
            sub2 = img2 - mu
            temp2 = sub2.reshape(-1, 3) @ inv
            prod2 = temp2.reshape(-1, 1, 3) @ sub2.reshape(-1, 3, 1)
            prod2 = prod2.reshape(sub2.shape[0], sub2.shape[1])
            prod = prod + prod2
    else:
        img2 = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2RGB)
        sub = img2 - mu
        temp = sub.reshape(-1, 3) @ inv
        prod = temp.reshape(-1, 1, 3) @ sub.reshape(-1, 3, 1)
        prod = prod.reshape(sub.shape[0], sub.shape[1])

    #show original image
    if display:
        plt.imshow(img2)
        plt.show()
    #TODO: figure out a better way to get this constant
    bw = 255 * (prod < 7)

    if dilate:
        kernel = np.ones((5, 5), 'uint8')
        bw = cv2.dilate(np.uint8(bw), kernel)

    if display:
        plt.imshow(bw, cmap='gray')
        plt.show()

    for r in regionprops(label(bw)):
        a, b, c, d = r.bbox
        if r.area > bw.size / 500:
            cv2.rectangle(img2, (d, c), (b, a), (255,0,0), 2)
            print('ImageNo = [' + str(number) + '], CentroidX = ' + str((b + d) / 2) + ', CentroidY = ' + str((a + c) / 2) + ', Distance = ' + str(445 / np.abs(b - d)) + ' or ' + str(649 / np.abs(a - c)) + ', but actually ' + str(filename.split('.')[0]))

    if display:
        plt.imshow(img2)
        plt.show()



    # TODO 2/9 pm:
    # gmm...
    # ensure dilation size is good
    # implement hsv/rgb hybrid
    # calculate constant based on image? (to handle shadows?) - ensure at least 1 barrel
    # how to filter down box sizes? How to use height/width? (ratio of dimensions?)

@functools.cache
def get_gmm_params(k, hsv=False):
    gmm_means = np.random.rand(k, 3) * 255
    gmm_covs = np.array([cov + np.random.rand() * 1 for i in range(k)])
    
    with open('extend_red.npy', 'rb') as f:
        gmm_X = np.load(f)
        if hsv:
            gmm_X = cv2.cvtColor(gmm_X.reshape(1, gmm_X.shape[0], gmm_X.shape[1]), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float64)
            gmm_X[:, 0] = (gmm_X[:, 0] + 90) % 180


    gmm_N = gmm_X.shape[0]

    current = -inf
    while True:
        gmm_sub = gmm_X.reshape(-1, 1, 3) - gmm_means.reshape(1, -1, 3)
        #print("sub test", gmm_sub.shape == (gmm_N, k, 3))
        #print((gmm_sub.reshape(gmm_N, k, 1, 3) @ gmm_covs).shape)
        gmm_g = np.exp(-0.5 * ((gmm_sub.reshape(gmm_N, k, 1, 3) @ np.linalg.inv(gmm_covs)).reshape(gmm_N, k, 3) * gmm_sub).sum(axis=2)) / ((2 * np.pi) ** 1.5 * np.sqrt(np.array([np.linalg.det(gmm_covs[i, :, :].reshape(3, 3)) for i in range(k)]))).reshape(1, k)
        #print("g test", gmm_g.shape == (gmm_N, k))
        gmm_z = gmm_g / gmm_g.sum(axis=1, keepdims=True)
        #print("z test", gmm_z.shape == (gmm_N, k))

        gmm_means = gmm_z.T @ gmm_X / gmm_z.sum(axis=0, keepdims=True).T
        #print("means test", gmm_means.shape == (k, 3))
        gmm_sub = gmm_X.reshape(-1, 1, 3) - gmm_means.reshape(1, -1, 3)
        gmm_covs = (gmm_sub.reshape(gmm_N, k, 3, 1) * gmm_sub.reshape(gmm_N, k, 1, 3) * gmm_z.reshape(gmm_N, k, 1, 1)).sum(axis=0) / gmm_z.sum(axis=0).reshape(k, 1, 1)
        #print("covs test", gmm_covs.shape == (k, 3, 3))
        next = np.sum(np.log((gmm_g / k).sum(axis=1)))
        if current / next < 1.00001:
            break
        current = next
        print(current)
    print(gmm_means, gmm_covs)
    print(gmm_g)
    w = 1 / np.max(gmm_g, axis=0)
    # plt.hist(np.sum(gmm_g * w, axis=1))
    # plt.show()

    with open(str(k) + str(hsv) + '_mu.npy', 'wb') as f:
        np.save(f, np.array(gmm_means))
    with open(str(k) + str(hsv) + '_cov.npy', 'wb') as f:
        np.save(f, np.array(gmm_covs))
    with open(str(k) + str(hsv) + '_w.npy', 'wb') as f:
        np.save(f, np.array(w))
    
    return gmm_means, gmm_covs, w

def gmm_process_image(folder, filename, k, number, display=False, hsv=False, dilate=True, hybrid=False):
    # if hsv:
    #     img2 = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2HSV)
    #     img2[:, 0] = (img2[:, 0] + 90) % 180
    #     sub = img2 - mu_hsv
    #     temp = sub.reshape(-1, 3) @ inv_hsv
    #     prod = temp.reshape(-1, 1, 3) @ sub.reshape(-1, 3, 1)
    #     prod = prod.reshape(sub.shape[0], sub.shape[1])
    #     if hybrid:
    #         img2 = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2RGB)
    #         sub2 = img2 - mu
    #         temp2 = sub2.reshape(-1, 3) @ inv
    #         prod2 = temp2.reshape(-1, 1, 3) @ sub2.reshape(-1, 3, 1)
    #         prod2 = prod2.reshape(sub2.shape[0], sub2.shape[1])
    #         prod = prod + prod2
    # else:
    img2 = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2RGB)
    img2_hsv = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2HSV)
    img2_hsv[:, 0] = (img2_hsv[:, 0] + 90) % 180

    #mus, covs, ws = get_gmm_params(k)
    with open(str(k) + str(False) + '_mu.npy', 'rb') as f:
        mus = np.load(f)
    with open(str(k) + str(False) + '_cov.npy', 'rb') as f:
        covs = np.load(f)
    with open(str(k) + str(False) + '_w.npy', 'rb') as f:
        ws = np.load(f)
    subs = img2.reshape(img2.shape[0], img2.shape[1], 1, 3) - mus.reshape(1, 1, k, 3)
    temp = (subs.reshape(subs.shape[0], subs.shape[1], k, 1, 3) @ np.array([np.linalg.inv(cov) for cov in covs])).reshape(subs.shape[0], subs.shape[1], k, 3)
    prod = (temp * subs).sum(axis=3)
    g = np.exp(-0.5 * prod) / ((2 * np.pi) ** 1.5 * np.sqrt(np.array([np.linalg.det(covs[i, :, :].reshape(3, 3)) for i in range(k)])))
    p = np.sum(g * ws, axis=2)
    bw = 255 * (p > 0.25)
    if hsv:
        #mus_hsv, covs_hsv, ws_hsv = get_gmm_params(k, hsv=True)
        with open(str(k) + str(hsv) + '_mu.npy', 'rb') as f:
            mus_hsv = np.load(f)
        with open(str(k) + str(hsv) + '_cov.npy', 'rb') as f:
            covs_hsv = np.load(f)
        with open(str(k) + str(hsv) + '_w.npy', 'rb') as f:
            ws_hsv = np.load(f)
        subs_hsv = img2_hsv.reshape(img2_hsv.shape[0], img2_hsv.shape[1], 1, 3) - mus_hsv.reshape(1, 1, k, 3)
        temp_hsv = (subs_hsv.reshape(subs_hsv.shape[0], subs_hsv.shape[1], k, 1, 3) @ np.array([np.linalg.inv(cov) for cov in covs_hsv])).reshape(subs_hsv.shape[0], subs_hsv.shape[1], k, 3)
        prod_hsv = (temp_hsv * subs_hsv).sum(axis=3)
        g_hsv = np.exp(-0.5 * prod_hsv) / ((2 * np.pi) ** 1.5 * np.sqrt(np.array([np.linalg.det(covs_hsv[i, :, :].reshape(3, 3)) for i in range(k)])))
        p += np.sum(g_hsv * ws_hsv, axis=2)
        bw = 255 * (0.5 * p)# (p > 0.3)
        bw = (bw - np.mean(bw)) / np.std(bw)
        bw = bw > 2.5


    #show original image
    if display:
        plt.imshow(img2)
        plt.show()
    #TODO: figure out a better way to get this constant
    #bw = 255 * (p > 0.25)

    if dilate:
        kernel = np.ones((5, 5), 'uint8')
        bw = cv2.dilate(np.uint8(bw), kernel)

    if display:
        plt.imshow(bw, cmap='gray')
        plt.show()

    for r in regionprops(label(bw)):
        a, b, c, d = r.bbox
        if r.area > bw.size / 450 and (r.extent > 0.9 or ((c - a) / (d - b) > 1.2 and (c - a) / (d - b) < 2.1)):# or ((c - a) / (d - b) > 0.7):
            cv2.rectangle(img2, (d, c), (b, a), (255,0,0), 2)
            pred = max(445 / np.abs(b - d), 649 / np.abs(a - c))
            #if pred / float(filename.split('.')[0].split('_')[0]) > 1.5 or pred / float(filename.split('.')[0].split('_')[0]) < 0.6:
            print(filename)
            print('ImageNo = [' + str(number) + '], CentroidX = ' + str((b + d) / 2) + ', CentroidY = ' + str((a + c) / 2) + ', Distance = ' + str(pred))# + ', but actually ' + str(filename.split('.')[0]))

    if display or number in [2,3,11,25,40,43,47]:
        plt.imshow(img2)
        plt.show()

folder = sys.argv[1]

for no, file in enumerate(os.listdir(folder)):#['2.2.png', '2.6.png', '2.8.png', '3.11.png', '4.5.png', '4.8.png', '5.9.png' '7.6.png', '8.5.png', '14.png']):
    gmm_process_image(folder, file, 3, no, display=True, hsv=True)
#2,3,11,25,40,43,47



# def find_barrels(path, display=False):
#     for no, file in enumerate(['2.2.png', '2.6.png', '2.8.png', '3.11.png', '4.5.png', '4.8.png', '5.9.png' '7.6.png', '8.5.png', '14.png']): #os.listdir(path)):
#         process_image(path + file, no, display)


#find_barrels('./ECE5242Proj1-train/', display=True)

# with open('dist.npy', 'rb') as f:
#     dists = np.load(f)

# xs, ys = [], []
# for d in dists:
#     xs.append(d[0] * d[1])
#     ys.append(d[0] * d[2])

# xs = np.array(xs)
# ys = np.array(ys)
# print(xs.mean(), np.median(xs), xs.std(), ys.mean(), np.median(ys), ys.std())
# height: 649, width: 445

# folder = sys.argv[1]
# for number, filename in enumerate(os.listdir(folder)):#['2.2.png', '2.6.png', '4.8.png', '7.6.png', '8.5.png']):
#     print(filename) # record the filename, since listdir is unordered
#     # read one test image
#     # Your computations here!
#     process_image(folder, filename, number, display=True, hsv=True, hybrid=True)
#     # Display results:
#     # (1) Segmented image
#     # (2) Barrel bounding box
#     # (3) Distance of barrel
#     # You may also want to plot and display other diagnostic information
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


