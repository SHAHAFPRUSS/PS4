import Utils
import os
import cv2

def main():

    save_dir = r'C:\Users\IMOE001\Desktop\resultsps4'
    os.makedirs(save_dir, exist_ok=True)
    img, trans_img = Utils.read_image()
    print('hello')
    #Utils.compute_R_matrix(img)
    connect_img_sift = Utils.drew_SIFT_matches(img, trans_img)
    #key_points, drew_key_points = Utils.compute_key_points(img)
    #img_keypoints, img_trans_keypoints, matches = Utils.SIFT_matches(img, trans_img)
    #des, des_trans, matches = Utils.SIFT_matches(img, trans_img)

    print('try')
    cv2.imwrite(os.path.join(save_dir, f'img.jpg'), connect_img_sift)
    #cv2.imwrite(os.path.join(save_dir, f'grad_x.jpg'), check_img_x)

if __name__ == '__main__':
    main()

