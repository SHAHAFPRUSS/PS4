
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

    return transA

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

def harris_corners_detctor(R_matrix, threshold=0.25):
    size = (3, 3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    max_values_img = cv2.dilate(R_matrix, kernel)
    threshold = np.amax(max_values_img) * threshold
    max_values_img[max_values_img < threshold] = 0
    local_max_index = np.where(max_values_img)
    return local_max_index

def drew_harris_corners(img):
    R_matrix = compute_R_matrix(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    harris_corners_0, harris_corners_1 = harris_corners_detctor(R_matrix)
    for i, j in zip(harris_corners_0, harris_corners_1):
        harris_detector = cv2.circle(img, (j, i), 1, (0, 255, 0), -1)
    return harris_detector




