import Utils
import cv2
import os

def main():

    save_dir = r'C:\Users\IMOE001\Desktop\resultsps4'
    os.makedirs(save_dir, exist_ok=True)
    img = Utils.read_image()
    R_matrix = Utils.compute_R_matrix(img)
    Utils.harris_corners_detctor(R_matrix)
    harris_corners_detctor = Utils.drew_harris_corners(img)
    #scaled_R_matrix = Utils.scaled_img(R_matrix)

    cv2.imwrite(os.path.join(save_dir, f'harris_corners_detctor2.jpg'), harris_corners_detctor)
    #cv2.imwrite(os.path.join(save_dir, f'grad_x.jpg'), check_img_x)

if __name__ == '__main__':
    main()

