import Utils
import os
import cv2

def main():

    save_dir = r'C:\Users\IMOE001\Desktop\resultsps4'
    os.makedirs(save_dir, exist_ok=True)
    img_1, img_2 = Utils.read_image()
    _, _, RANSAC_MATCH = Utils.RANSAC_sample_solve_translation(img_1, img_2)
    RANSAC_MATCH = Utils.RANSAC_sample_solve_similarity(img_1, img_2)
    cv2.imwrite(os.path.join(save_dir, f'RANSAC_new.jpg'), RANSAC_MATCH)

if __name__ == '__main__':
    main()

