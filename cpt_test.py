import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

fx = 400
fy = 400
cx = 400
cy = 300
k1 = 0
k2 = 0
k3 = 0
p1 = 0
p2 = 0
# 读取相机内参
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])


class ORBSlamVideo:

    def __init__(self):
        self.orb = cv2.ORB.create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # get keypoints attribute
        self.maxCorners = 1000
        self.qualityLevel = 0.01
        self.minDistance = 3
        self.keypoint_size = 20

        self.min_dist = 30

        fx = 400
        fy = 400
        cx = 400
        cy = 300
        k1 = 0
        k2 = 0
        k3 = 0
        p1 = 0
        p2 = 0
        # 读取相机内参
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])

    def preprocess_image(self, image):
        image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return imagegray

    def get_key_points(self, frame_processed):
        pts_curr = cv2.goodFeaturesToTrack(frame_processed, self.maxCorners, qualityLevel=self.qualityLevel,
                                           minDistance=self.minDistance)

        kps_curr = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=self.keypoint_size) for pt in pts_curr]
        keypoints_curr, descriptors_curr = self.orb.compute(frame_processed, kps_curr)
        return keypoints_curr, descriptors_curr

    def get_g_matches(self, matches, min_dist_abs):
        g_matches = []
        dist_all = [x.distance for x in matches]
        min_dist = min(dist_all)
        for x in matches:
            if x.distance <= max(2 * min_dist, min_dist_abs):
                g_matches.append(x)
        return g_matches

    def get_points(self, g_matches, keypoints_prev, keypoints_curr):
        points1 = []
        points2 = []
        for i in g_matches:
            points1.append(list(keypoints_prev[i.queryIdx].pt))
            points2.append(list(keypoints_curr[i.trainIdx].pt))
        points1 = np.array(points1)
        points2 = np.array(points2)
        return points1, points2

    def plot_trajectory(self, trajectory, trajectory_gt=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if trajectory_gt:
            trajectory_gt = [[i[0] - trajectory_gt[0][0], i[1] - trajectory_gt[0][1], i[2] - trajectory_gt[0][2]]
                             for i in trajectory_gt]
            trajectory_gt = np.array(trajectory_gt)
            ax.plot(trajectory_gt[:, 0], trajectory_gt[:, 1], marker='x')

        trajectory = np.array(trajectory)
        ax.plot(-trajectory[:, 0], -trajectory[:, 2], marker='o')
        ax.scatter(-trajectory[0, 0], -trajectory[0, 2], c='g', marker='o', label='Start')
        ax.scatter(-trajectory[-1, 0], -trajectory[-1, 2], c='r', marker='o', label='End')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        plt.title('Camera Trajectory')
        plt.show()

    @staticmethod
    def get_color(depth, up_th, low_th):
        th_range = up_th - low_th
        if depth > up_th:
            depth = up_th
        if depth < low_th:
            depth = low_th
        return (255 * depth / th_range, 0, 255 * (1 - depth / th_range))

    def get_3dpts(self, R, t, points1, points2):
        projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机参数
        projMatr2 = np.concatenate((R, t), axis=1)  # 第二个相机参数
        projMatr1 = np.matmul(self.camera_matrix, projMatr1)  # 相机内参 相机外参
        projMatr2 = np.matmul(self.camera_matrix, projMatr2)  #
        points4D = cv2.triangulatePoints(projMatr1, projMatr2, points1.T, points2.T)
        points4D /= points4D[3]  # 归一化
        points3D = points4D.T[:, 0:3]  # 取坐标点
        return points3D

    def pixel2cam(self, pt):
        x = (pt[0] - cx) / fx
        y = (pt[1] - cy) / fy
        return np.array([x, y, 1])

    def read_carla(self, filedir):
        files = os.listdir(filedir)
        i = 0
        mean_ds = []
        frame_datas = []
        trajectory = []
        cur_R = np.eye(3)
        cur_t = np.zeros((3, 1))
        trajectory_gt = []
        for file in files[:30]:
            if os.path.isfile(f'{filedir}/{file}'):
                continue
            frame = cv2.imread(f'{filedir}/{file}/rgb.png')
            depth = cv2.imread(f'{filedir}/{file}/depth.png')

            imu = np.loadtxt(f'{filedir}/{file}/imu.txt', delimiter=',')
            gnss = np.loadtxt(f'{filedir}/{file}/gnss.txt', delimiter=',')
            trajectory_gt.append(gnss)

            if not isinstance(frame, np.ndarray):
                break
            self.orb = cv2.ORB.create()
            frame_processed = self.preprocess_image(frame)
            keypoints_curr, descriptors_curr = self.get_key_points(frame_processed)
            if i == 0:
                frame_prev = frame
                keypoints_prev, descriptors_prev = keypoints_curr, descriptors_curr
                i += 1
                continue

            matches = self.matcher.match(descriptors_prev, descriptors_curr)
            g_matches = self.get_g_matches(matches, self.min_dist)
            g_matches_image = cv2.drawMatches(frame_prev, keypoints_prev, frame, keypoints_curr, g_matches, None)
            points_prev, points_curr = self.get_points(g_matches, keypoints_prev, keypoints_curr)
            em, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, cv2.RANSAC)
            num, R, t, mask = cv2.recoverPose(em, points_prev, points_curr, self.camera_matrix)
            points3d_local_curr = self.get_3dpts(R, t, points_prev, points_curr)
            cur_R = R @ cur_R
            cur_t = cur_t + cur_R @ t
            trajectory.append((cur_t[0, 0], cur_t[1, 0], cur_t[2, 0]))

            frame_prev = frame
            keypoints_prev, descriptors_prev,points3d_local_prev = keypoints_curr, descriptors_curr,points3d_local_curr
            i += 1
            frame_datas.append([R, t, points3d_local_prev])

            cv2.imshow('keypoints', g_matches_image)
            cv2.imshow('depth', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.bundle_adjustment(frame_datas, points_prev, points_curr, points3d_local_prev, points3d_local_curr)
        self.plot_trajectory(trajectory, trajectory_gt)


    def bundle_adjustment(self, frame_datas, points_prev, points_curr, points3d_local_prev, points3d_local_curr):
        n_cameras = len(frame_datas)
        n_points = points_prev.shape[0]
        camera_params = np.zeros((n_cameras, 6))
        points_3d = np.zeros((n_points, 3))

        for i, (R, t, points3d) in enumerate(frame_datas):
            rvec, _ = cv2.Rodrigues(R)
            camera_params[i, :3] = rvec.flatten()
            camera_params[i, 3:] = t.flatten()
            points_3d = points3d  # 更新三维点

        camera_indices = np.repeat(np.arange(n_cameras), n_points)
        point_indices = np.tile(np.arange(n_points), n_cameras)
        points_2d = np.vstack([points_prev for _ in range(n_cameras)])

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

        result = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                               args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

        optimized_camera_params = result.x[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = result.x[n_cameras * 6:].reshape((n_points, 3))

        print("优化后的相机参数：\n", optimized_camera_params)
        print("优化后的三维点：\n", optimized_points_3d)





    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def project(self, points, camera_params):
        points_proj = np.zeros((points.shape[0], 2))
        for i, point in enumerate(points):
            rvec = camera_params[i][:3]
            tvec = camera_params[i][3:]
            R, _ = cv2.Rodrigues(rvec)
            point_proj = R @ point + tvec
            point_proj = self.camera_matrix @ point_proj
            points_proj[i] = point_proj[:2] / point_proj[2]
        return points_proj




if __name__ == '__main__':
    slam = ORBSlamVideo()
    slam.read_carla('2024-07-17_20-37-36')