import numpy as np
import cv2
import os
from scipy.ndimage import maximum_filter


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

    return simA

def scaled_img(img):
    scaled_img = 255*(img - np.min(img)) / np.ptp(img).astype(int)
    return scaled_img

def compute_gradients(img = None):
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    return grad_x, grad_y

def compute_R_matrix(img, alpha = 0.01):
    R_matrix = np.empty(img.shape)
    I_x, I_y = compute_gradients(img)
    I_x_x = cv2.GaussianBlur(I_x**2, (7, 7), 0)
    I_y_y = cv2.GaussianBlur(I_y**2, (7, 7), 0)
    I_x_y = cv2.GaussianBlur(I_x*I_y, (7, 7), 0)
    for (i, j), t in np.ndenumerate(R_matrix):
        M_matrix = [[I_x_x[i, j], I_x_y[i, j]], [I_x_y[i, j], I_y_y[i, j]]]
        M_matrix = np.array(M_matrix).reshape(2, 2)
        R_matrix[i, j] = cv2.determinant(M_matrix) - alpha*cv2.trace(M_matrix)[0]**2
    return R_matrix

def harris_corners_detctor(R_matrix):
    max_values = maximum_filter(R_matrix, size=4, mode='constant')
    print(max_values)

