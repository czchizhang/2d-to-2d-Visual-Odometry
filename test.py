import cv2
import numpy as np
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
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])

    def preprocess_image(self, image):
        image = cv2.undistort(image, self.camera_matrix, dist_coeffs)
        imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return imagegray

    def get_key_points(self, frame_processed):
        pts_curr = cv2.goodFeaturesToTrack(frame_processed, self.maxCorners, qualityLevel=self.qualityLevel,
                                           minDistance=self.minDistance)

        kps_curr = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=self.keypoint_size) for pt in pts_curr]
        keypoints_curr, descriptors_curr = self.orb.compute(frame_processed, kps_curr)
        print(len(kps_curr))
        return keypoints_curr, descriptors_curr

    def get_g_matches(self, matches, min_dist_abs):
        g_matches = []
        dist_all = [x.distance for x in matches]
        min_dist = min(dist_all)
        for x in matches:
            if x.distance <= max(2*min_dist, min_dist_abs):
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

    @staticmethod
    def get_color(depth, up_th, low_th):
        th_range = up_th - low_th
        if (depth > up_th):
            depth = up_th
        if (depth < low_th):
            depth = low_th
        return (255 * depth / th_range, 0, 255 * (1 - depth / th_range));

    def get_3dpts(self, R, t, points1, points2):
        projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机参数
        projMatr2 = np.concatenate((R, t), axis=1)  # 第二个相机参数
        projMatr1 = np.matmul(camera_matrix, projMatr1)  # 相机内参 相机外参
        projMatr2 = np.matmul(camera_matrix, projMatr2)  #
        points4D = cv2.triangulatePoints(projMatr1, projMatr2, points1.T, points2.T)
        points4D /= points4D[3]  # 归一化
        points3D = points4D.T[:, 0:3]  # 取坐标点
        return points3D

    def read_video(self, filename):
        cap = cv2.VideoCapture(filename)
        i = 0

        # mlab.figure(size=(800, 600), bgcolor=(0, 0, 0))
        while (cap.isOpened()):
            ret, frame = cap.read()
            # self.orb = cv2.ORB.create()
            # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

            if not isinstance(frame, np.ndarray):
                break
            self.orb = cv2.ORB.create()
            frame_processed = self.preprocess_image(frame)
            keypoints_curr, descriptors_curr = self.get_key_points(frame_processed)
            print(len(keypoints_curr))
            if i == 0:
                frame_prev = frame

                keypoints_prev, descriptors_prev = keypoints_curr, descriptors_curr
                i += 1
                continue
            cv2.imwrite('p1.png', frame_prev)
            cv2.imwrite('p2.png', frame)
            # if i%2:
            #     i+=1
            #     continue
            # matches = self.matcher.match(descriptors_prev, descriptors_curr)
            matches = self.matcher.match(descriptors_prev, descriptors_curr)
            g_matches = self.get_g_matches(matches, self.min_dist)
            g_matches_image = cv2.drawMatches(frame_prev, keypoints_prev, frame, keypoints_curr, g_matches, None)

            points_prev, points_curr= self.get_points(g_matches, keypoints_prev, keypoints_curr)
            em, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, cv2.RANSAC)
            num, R, t, mask = cv2.recoverPose(em, points_prev, points_curr, self.camera_matrix)
            points3d_local = self.get_3dpts(R, t, points_prev, points_curr)

            for i in range(points_curr.shape[0]):
                # 第一幅图
                im1 = cv2.circle(frame_prev, (points_prev[i][0].astype(int), points_prev[i][1].astype(int)), 5,
                                 self.get_color(points3d_local[i, 2], 50, 10), -1)

            cv2.imshow('keypoints', g_matches_image)
            cv2.imshow('depth', im1)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # print(points3d_local)
            # mlab.points3d(points3d_local[:, 0], points3d_local[:, 1], points3d_local[:, 2], scale_factor=1, color=(1, 0, 0))
            frame_prev = frame
            keypoints_prev, descriptors_prev = keypoints_curr, descriptors_curr
            i += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    slam = ORBSlamVideo()
    slam.read_video('video_e2.mp4')
