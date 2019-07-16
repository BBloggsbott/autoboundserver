import cv2

def get_corners(img, original_image_size=(256,256)):
    dst = cv2.cornerHarris(img,2, 3, 0.04)
    corners = []
    m = dst.max()
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] > 0.01*m:
                corners.extend([(i,j)])
    return corners