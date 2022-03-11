import cv2
import numpy as np


if __name__ == "__main__":
    imge = cv2.imread("./hough/hough2.png")
    # convert the read image to greyscale
    imag = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    # doing gaussian blur to remove noise
    img = cv2.GaussianBlur(imag,(3,3),0)

    # get X and Y gradient

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    #Ix = cv2.convertScaleAbs(sobelX)
    #Iy = cv2.convertScaleAbs(sobelY)
    (height, width) = img.shape[:2]
    Ix = np.zeros((height, width))
    Iy = np.zeros((height, width))
    cv2.normalize(np.absolute(sobelX), Ix, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(np.absolute(sobelY), Iy, 0, 255, cv2.NORM_MINMAX)
    Ix = np.uint8(Ix)
    Iy = np.uint8(Iy)

    # find double derivative
    dxx = sobelX*sobelX
    dyy = sobelY*sobelY
    dxy = sobelY*sobelX

    Ixy = np.zeros((height,width))
    cv2.normalize(np.absolute(dxy), Ixy, 0, 255, cv2.NORM_MINMAX)
    Ixy = np.uint8(Ixy)

    response = np.zeros((height, width))
    har_response = np.zeros((height, width))
    win_size = 3
    offset = win_size//2
    k = 0.04
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            Mxx = np.sum(dxx[start_y : end_y, start_x : end_x])
            Myy = np.sum(dyy[start_y : end_y, start_x : end_x])
            Mxy = np.sum(dxy[start_y : end_y, start_x : end_x])

            det = (Mxx*Myy) - (Mxy**2)
            trace = Mxx + Myy
            r = det - k*(trace**2)
            response[y-offset][x-offset] = r

    print(dxx)
    print(response)

    cv2.normalize(response, har_response, 0, 255, cv2.NORM_MINMAX)
    #corners = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    print(har_response)
    #print(corners)

    cv2.imshow("Edge X", Ix)
    cv2.imshow("Edge Y", Iy)
    cv2.imshow("Ixy", Ixy)
    cv2.imshow("Response", har_response)
    cv2.waitKey(0)
    cv2.destroyAllWindows()