import random
import math
import cv2
import numpy as np


def read_image():
    simA = cv2.imread(
        fr'C:\Users\IMOE001\Desktop\shahaf\coursses\ps4\simA.jpg',
        cv2.IMREAD_GRAYSCALE)
    simB = cv2.imread(
        fr'C:\Users\IMOE001\Desktop\shahaf\coursses\ps4\simB.jpg',
        cv2.IMREAD_GRAYSCALE)
    transA = cv2.imread(
        fr'C:\Users\IMOE001\Desktop\shahaf\coursses\ps4\transA.jpg',
        cv2.IMREAD_GRAYSCALE)
    transB = cv2.imread(
        fr'C:\Users\IMOE001\Desktop\shahaf\coursses\ps4\transB.jpg',
        cv2.IMREAD_GRAYSCALE)

    return transA, transB


def compute_gradients(img=None):
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    return grad_x, grad_y


def compute_R_matrix(img, alpha=0.02):
    R_matrix = np.empty(img.shape)
    I_x, I_y = compute_gradients(img)
    I_x_x = cv2.GaussianBlur(I_x ** 2, (7, 7), 0)
    I_y_y = cv2.GaussianBlur(I_y ** 2, (7, 7), 0)
    I_x_y = cv2.GaussianBlur(I_x * I_y, (7, 7), 0)
    for (i, j), t in np.ndenumerate(R_matrix):
        M_matrix = [[I_x_x[i, j], I_x_y[i, j]], [I_x_y[i, j], I_y_y[i, j]]]
        M_matrix = np.array(M_matrix).reshape(2, 2)
        R_matrix[i, j] = cv2.determinant(M_matrix) - alpha * cv2.trace(M_matrix)[0] ** 2
    return R_matrix


def harris_corners_detctor(R_matrix, thresh=0.1):
    size = (10, 10)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    max_values_img = cv2.dilate(R_matrix, kernel)
    threshold = np.amax(max_values_img) * thresh
    max_values_img[max_values_img < threshold] = 0
    local_max_index = np.where(max_values_img)

    return local_max_index


def drew_harris_corners(img):
    R_matrix = compute_R_matrix(img)
    img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    harris_corners_0, harris_corners_1 = harris_corners_detctor(R_matrix)
    for i, j in zip(harris_corners_0, harris_corners_1):
        harris_detector = cv2.circle(img, (j, i), 1, (0, 255, 0), -1)
    return harris_detector


def NMS(harris_corners_0, harris_corners_1, mag):
    current_i = 0
    current_i_array = []
    for i in range(len(harris_corners_0) - 1):
        if harris_corners_0[i] == harris_corners_0[i + 1] and \
                mag[harris_corners_0[i], harris_corners_1[i]] >= mag[harris_corners_0[i], harris_corners_1[i]]:
            current_i = i
        elif harris_corners_0[i] != harris_corners_0[i + 1]:
            current_i_array.append(current_i)
        elif i == len(harris_corners_0):
            current_i_array.append(current_i)
    return current_i_array


def compute_key_points(img):
    I_x, I_y = compute_gradients(img)
    mag, angle = cv2.cartToPolar(I_x, I_y, angleInDegrees=True)
    angle = np.arctan2(I_x, I_y)
    R_matrix = compute_R_matrix(img)
    harris_corners_0, harris_corners_1 = harris_corners_detctor(R_matrix)
    key_points = []
    pick_i = NMS(harris_corners_0, harris_corners_1, mag)
    mag = 1
    for x in pick_i:
        key_points.append(cv2.KeyPoint(float(harris_corners_1[x]), float(harris_corners_0[x]),
                                       mag,
                                       180 + np.rad2deg(angle[harris_corners_0[x], harris_corners_1[x]])))
    drew_key_points = cv2.drawKeypoints(img, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return key_points


def find_SIFT_matches(img_1, img_2):
    sift = cv2.SIFT_create()
    bfm = cv2.BFMatcher()
    img1_k_points = compute_key_points(img_1)
    _, des_points = sift.compute(img_1, img1_k_points)
    img2_k_points = compute_key_points(img_2)
    _, des_trans_points = sift.compute(img_2, img2_k_points)
    matches = bfm.match(des_points, des_trans_points)
    return img1_k_points, img2_k_points, matches

def drew_matches(img_1, img_2, matches=None):
    connect_img = cv2.cvtColor(np.concatenate((img_1, img_2), axis=1), cv2.COLOR_GRAY2RGB)
    if matches == None:
        img_1_keypoints, img_2_keypoints, matches = find_SIFT_matches(img_1, img_2)
    else:
        img_1_keypoints, img_2_keypoints, _ = find_SIFT_matches(img_1, img_2)

    for m in matches:
        src_points = img_1_keypoints[m.queryIdx].pt
        dst_points = img_2_keypoints[m.trainIdx].pt
        cv2.line(connect_img, (int(src_points[0]), int(src_points[1])), (int(img_1.shape[1] + dst_points[0]),
                                                                         int(dst_points[1])),
                 color=np.random.randint(0, 255, 3).tolist(), thickness=1)
    return connect_img

def get_points_and_distance(img_1_keypoints, img_2_keypoints, random_sample):
    first_point = img_1_keypoints[random_sample.queryIdx].pt
    second_point = img_2_keypoints[random_sample.trainIdx].pt
    first_point = (int(first_point[0]), int(first_point[1]))
    second_point = (int(second_point[0]), int(second_point[1]))
    d_x = first_point[0] - second_point[0]
    d_y = first_point[1] - second_point[1]
    return first_point, second_point, d_x, d_y

def RANSAC_sample_solve_translation(img_1, img_2, threshold = 5):
    inliers = []
    inliers_temp = []
    img_1_keypoints, img_2_keypoints, matches = find_SIFT_matches(img_1, img_2)
    for i in matches:
        _, _, d_x, d_y = get_points_and_distance(img_1_keypoints, img_2_keypoints, random.choice(matches))
        for m in matches:
            first_point, second_point, _, _ = get_points_and_distance(img_1_keypoints, img_2_keypoints, m)
            remove_first_point = (first_point[0] - d_x, first_point[1] - d_y)
            err = math.hypot(remove_first_point[0] - second_point[0], remove_first_point[1] - second_point[1])
            if err <= threshold:
                inliers_temp.append(m)
        if len(inliers_temp) > len(inliers):
            inliers = inliers_temp
            best_match = (d_x, d_y)
        inliers_temp = []
    RANSAC_match = drew_matches(img_1, img_2, inliers)
    return best_match, inliers, RANSAC_match

def RANSAC_sample_solve_similarity(img_1, img_2, threshold = 5, matrix='translation'):
    inliers = []
    inliers_temp = []
    img_1_keypoints, img_2_keypoints, matches = find_SIFT_matches(img_1, img_2)
    for i in matches:
        x, y, d_x, d_y = get_points_and_distance(img_1_keypoints, img_2_keypoints, random.choice(matches))
        x_w, y_w, _, _ = get_points_and_distance(img_1_keypoints, img_2_keypoints, random.choice(matches))
        h_matrix = compute_h_matrix(x, y, x_w, y_w)
        for m in matches:
            first_point, second_point, _, _ = get_points_and_distance(img_1_keypoints, img_2_keypoints, m)
            first_point = np.array([first_point[0], first_point[1], 1]).reshape(3, 1)
            if matrix == 'translation':
                translation_matrix = np.array([[1, 0, h_matrix[0][2]], [0, 1, h_matrix[0][3]], [0, 0, 1]]).reshape(3, 3)
                new_point = np.matmul(translation_matrix, first_point)
            if matrix == 'similarity':
                similarity_matrix = np.array([[h_matrix[0][0], -h_matrix[0][1], h_matrix[0][2]],
                                             [h_matrix[0][1], h_matrix[0][0], h_matrix[0][3]]]).reshape(2, 3)
                new_point = np.matmul(similarity_matrix, first_point)
            err = math.hypot(new_point[0] - second_point[0], new_point[1] - second_point[1])
            if err <= threshold:
                inliers_temp.append(m)
        if len(inliers_temp) > len(inliers):
            inliers = inliers_temp
        inliers_temp = []
    RANSAC_match = drew_matches(img_1, img_2, inliers)
    return RANSAC_match

def compute_h_matrix(first_point, second_point, first_point_w, second_point_w):
    p_vector = np.array([[first_point[0], -first_point[1], 1, 0], [first_point[1], -first_point[0], 0, 1],
                         [second_point[0], -second_point[1], 1, 0],
                         [second_point[1], -second_point[0], 0, 1]]).reshape(4, 4)
    pw_vector = np.array([first_point_w[0], first_point_w[1], second_point_w[0], second_point_w[1]]).reshape(4, 1)
    h_martix = np.linalg.solve(p_vector, pw_vector)
    return h_martix.reshape(1, 4)

def translation_transform(d_x, d_y, matches, img_1_keypoints, img_2_keypoints, img_1, img_2):
    inliers = []
    inliers_temp = []
    threshold = 5
    for m in matches:
        first_point, second_point, _, _ = get_points_and_distance(img_1_keypoints, img_2_keypoints, m)
        remove_first_point = (first_point[0] - d_x, first_point[1] - d_y)
        err = math.hypot(remove_first_point[0] - second_point[0], remove_first_point[1] - second_point[1])
        if err <= threshold:
            inliers_temp.append(m)
    if len(inliers_temp) > len(inliers):
        inliers = inliers_temp
        best_match = (d_x, d_y)
    inliers_temp = []
    return inliers

def RANSAC_sample_solve(img_1, img_2, matrix='translation', threshold=5):
    inliers = []
    inliers_temp = []
    img_1_keypoints, img_2_keypoints, matches = find_SIFT_matches(img_1, img_2)
    for i in matches:
        if matrix == 'translation':
            err, m = similarity_transform(matches, img_1_keypoints, img_2_keypoints)
        if matrix == 'similarity':
            inliers.append(similarity_transform(x, y, x_w, y_w, matches, img_1_keypoints, img_2_keypoints))
        if err <= threshold:
            inliers_temp.append(m)
    if len(inliers_temp) > len(inliers):
        inliers = inliers_temp
    inliers_temp = []
    RANSAC_match = drew_matches(img_1, img_2, inliers)
    return RANSAC_match

def similarity_transform(matches, img_1_keypoints, img_2_keypoints, threshold=5):
    inliers_temp = []
    _, _, d_x, d_y = get_points_and_distance(img_1_keypoints, img_2_keypoints, random.choice(matches))
    for m in matches:
        first_point, second_point, _, _ = get_points_and_distance(img_1_keypoints, img_2_keypoints, m)
        remove_first_point = (first_point[0] - d_x, first_point[1] - d_y)
        err = math.hypot(remove_first_point[0] - second_point[0], remove_first_point[1] - second_point[1])
        if err <= threshold:
            inliers_temp.append(m)
    return inliers_temp
