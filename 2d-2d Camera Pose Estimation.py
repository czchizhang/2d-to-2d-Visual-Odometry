import cv2
import numpy as np
from cv2 import DMatch
from matplotlib import pyplot as plt
from python_orb_slam3 import ORBExtractor
from mayavi import mlab
fx = 573.1821
fy = 570.1708
cx = 274.9323
cy = 196.7571
k1 = -0.5364
k2 = 0.4164
k3 = -0.2957
p1 = 0
p2 = 0
# 读取相机内参
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

print(camera_matrix)

# # 视频转图片
#
# vc = cv2.VideoCapture(r'')  # 读入视频文件，命名cv
# n = 1  # 计数
#
# if vc.isOpened():  # 判断是否正常打开
#     rval, frame = vc.read()
# else:
#     rval = False
#
# timeF = 10  # 视频帧计数间隔频率
#
# i = 0
# while rval:  # 循环读取视频帧
#     rval, frame = vc.read()
#     if (n % timeF == 0):  # 每隔timeF帧进行存储操作
#         i += 1
#         print(i)
#         cv2.imwrite(r'C:\Users\TomZC\Desktop\thesis\master project\python_orbslam\example_book_1/{}.jpg'.format(i),
#                     frame)  # 存储为图像
#     n = n + 1
#     cv2.waitKey(1)
# vc.release()

# 读取图像
image1 = cv2.imread('p1.png')
image2 = cv2.imread('p2.png')

# 去畸变

image1 = cv2.undistort(image1, camera_matrix, dist_coeffs)
image2 = cv2.undistort(image2, camera_matrix, dist_coeffs)

# 灰度
image1gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 角点筛选
pts1 = cv2.goodFeaturesToTrack(image1gray, 1000, qualityLevel=0.01, minDistance=3)
pts2 = cv2.goodFeaturesToTrack(image2gray, 1000, qualityLevel=0.01, minDistance=3)

kps1 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in pts1]
kps2 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in pts2]

# ORB 特征点手动匹配
orb = cv2.ORB.create()
keypoints1, descriptors1 = orb.compute(image1gray, kps1)
keypoints2, descriptors2 = orb.compute(image2gray, kps2)
print(len(keypoints1))
print(len(kps1))
print(len(keypoints2))
# #ORB 自动提取特征点
# orb_extractor = ORBExtractor()
# keypoints1, descriptors1 = orb_extractor.detectAndCompute(image1,)
# keypoints2, descriptors2 = orb_extractor.detectAndCompute(image2,)
# 特征点匹配
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.match(descriptors1, descriptors2)

min_distance = 10000
max_distance = 0


# for x in matches:
#     if x.distance < min_distance:
#         min_distance = x.distance
#     if x.distance > min_distance:
#         max_distance = x.distance
# print('min_distance: %f' % min_distance)
# print('max_distance: %f' % max_distance)
g_matches = []
for x in matches:
    # if x.distance <= min(max(min_dist, 30), 100):
    if x.distance <= 30:
        g_matches.append(x)
print('# of matches: %d' % len(g_matches))
# g_matches  = matches

g_matches_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, g_matches, None)

# Show matches
plt.imshow(g_matches_image[:, :, ::-1])
plt.show()

points1 = []
points2 = []

for i in g_matches:
    points1.append(list(keypoints1[i.queryIdx].pt))
    points2.append(list(keypoints2[i.trainIdx].pt))

points1 = np.array(points1)
points2 = np.array(points2)

em, mask = cv2.findEssentialMat(points1, points2, camera_matrix, cv2.RANSAC)

num, R, t, mask = cv2.recoverPose(em, points1, points2, camera_matrix)
print("Essential matrix:")
print(em)
print("R_M:")
print(R)
print("t_M:")
print(t)

# p1 = np.array([[0],[0],[0]])
#
# p2 = p1 + t
#
#
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# x, y, z = [p1[0][0], p2[0][0]], [p1[1][0], p2[1][0]], [p1[2][0], p2[2][0]]
# ax.scatter(x, y, z, c='red', s=100)
# ax.plot(x, y, z, color='black')
# plt.show()


# triangulate Points get depth

projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机参数
projMatr2 = np.concatenate((R, t), axis=1)  # 第二个相机参数
projMatr1 = np.matmul(camera_matrix, projMatr1)  # 相机内参 相机外参
projMatr2 = np.matmul(camera_matrix, projMatr2)  #
points4D = cv2.triangulatePoints(projMatr1, projMatr2, points1.T, points2.T)
print(points4D)
points4D /= points4D[3]  # 归一化
points4D = points4D.T[:, 0:3]  # 取坐标点
print("Feature points depth example (The first five):")
print(points4D[0:5])
print("Current camera pose:")
print(projMatr2)


def get_color(depth):
    up_th = 50
    low_th = 10
    th_range = up_th - low_th
    if (depth > up_th):
        depth = up_th
    if (depth < low_th):
        depth = low_th
    return (255 * depth / th_range, 0, 255 * (1 - depth / th_range));


## 深度可视化
for i in range(points1.shape[0]):
    # 第一幅图
    im1 = cv2.circle(image1, (points1[i][0].astype(int), points1[i][1].astype(int)), 5, get_color(points4D[i, 2]), -1)
    # 第二幅图
    tmp_point = np.dot(R, points4D[i, :].reshape(3, 1)) + t
    tmp_point = tmp_point.reshape(-1)
    im2 = cv2.circle(image2, (points2[i][0].astype(int), points2[i][1].astype(int)), 5, get_color(tmp_point[2]), -1)
print(f"{max(points4D[:, 2])=}")
plt.subplot(121)
plt.imshow(im1[:, :, ::-1])
plt.subplot(122)
plt.imshow(im2[:, :, ::-1])
plt.show()
mlab.points3d(points4D[:, 0], points4D[:, 1], points4D[:, 2], scale_factor=0.05)
mlab.show()
# points1[:,0] = (points1[:,0] - cx) / fx
# points1[:,1] = (points1[:,1] - cy) / fy
#
# points2[:,0] = (points2[:,0] - cx) / fx
# points2[:,1] = (points2[:,1] - cy) / fy
#
# b1 = np.ones(points1.shape[0])
# b2 = np.ones(points2.shape[0])
# #
# #
# y1 = np.c_[points1,b1]
# y2 = np.c_[points2,b2]

# y1 = np.array([[points1[0]], [points1[1]], [1]])
# y2 = np.array([[points2[0]], [points2[1]], [1]])

def pixel2cam(pt):
    x = (pt[0] - cx) / fx
    y = (pt[1] - cy) / fy
    return np.array([x, y, 1])


# 定义 t_x 矩阵
t_x = np.array([[0, -t[2, 0], t[1, 0]],
                [t[2, 0], 0, -t[0, 0]],
                [-t[1, 0], t[0, 0], 0]])

print("t_x =", t_x)
tR = np.dot(t_x, R)

print("t^R =", tR)
print('points2 %d' % len(points2))

ds = []
total_d = 0
num = 0
for m in matches:
    # 计算 y1
    pt1 = pixel2cam(keypoints1[m.queryIdx].pt)
    y1 = np.array([[pt1[0]], [pt1[1]], [1]])

    # 计算 y2
    pt2 = pixel2cam(keypoints2[m.trainIdx].pt)
    y2 = np.array([[pt2[0]], [pt2[1]], [1]])

    # 计算 d
    d = np.dot(np.dot(np.dot(y2.T, t_x), R), y1)
    ds.append(d[0, 0])
    total_d = total_d + d
    num = num + 1
    print("epipolar constraint =", d)
mean_d = total_d/num
print("mean_d =", mean_d)
plt.hist(ds, bins=50)
plt.show()

# image1 = cv2.imread('e2.jpg')
# image2 = cv2.imread('e3.jpg')
"""
image_files = []
mapp = np.empty((0, 3))
prev_file = read image
read imu
original transformation [[]]
for file in image_files[1:]:
    b = read_prev file
    a = read file
    feature_a = get_feature_pts(a)
    feaature_b
    3dpoints = get_3d_points(a, b)
    3dpoints@trasnformation
    current transformation
    mapp = np.row_stack((mapp, new_observation))
    
"""
