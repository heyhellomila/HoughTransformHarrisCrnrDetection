import cv2
import numpy as np

'''
    We use the formula p = x*cos(theta) + y*cos(theta) as representation of an edge line where p is the length of the 
    perpendicular line drawn form origin(0,0) to the edge line and theta is the angle that the perpendicular line makes 
    with the x-axis in counter-clockwise direction.
    We transform each edge point(x',y') in the edgeMap image to a cosine curve in the paramater space using the above 
    formula. As a result, we will get many cosine curves intersecting at various points. We will find the ...
    points (p, theta) through which many cosine curves pass greater than threshold pass and those will correspond to 
    detected lines in edge map   
'''
def houghTransform(edgeMap, p_max):
    '''
    :param edgeMap: image containing only edges obtained through canny edge detection
    :param p_max: maximum possible length of the perpendicular drawn from origin to any edge line in edge image
    :return:  the np array containing hough transgform
    '''

    (height, width) = edgeMap.shape[:2]
    # p_max represents the maximum length a perpendicular drawn from origin to an edge line in image can have
    # p can range from -p_max to p_max
    degrees = 180
    # hough image will store hough transform data
    houghImage = np.zeros((2*(p_max+1)+1, degrees+1), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            if edgeMap[y][x] != 0:
                #we have an edge point, find all (p, theta) pairs of lines passing through these points
                for theta in range(1,degrees):
                    # p = x*cos(theta) + y*sin(theta)
                    # numpy cos and sin functions take angle as radians so convert degrees to radians
                    p = x*np.cos(theta*np.pi/180) + y*np.sin(theta*np.pi/180)
                    p = int(p + p_max)
                    if(houghImage[p][theta] < 255):
                        houghImage[p][theta] += 1

    return houghImage


def findPointsForLine(p, theta, width, height):
    # p = x*cos(theta) + y*sin(theta)
    x_0 = 0
    y_0 = 0
    try:
        y_0 = int(p/np.sin(theta*np.pi/180))
    except ZeroDivisionError:
        x_0 = p
        y_0 = 0

    x_1 = width - 1
    y_1 = 0
    try:
        y_1 = int((p - x_1 * np.cos(theta * np.pi / 180)) / np.sin(theta * np.pi / 180))
    except ZeroDivisionError:
        x_1 = p
        y_1 = 1

    y_2 = 0
    x_2 = 0
    try:
        x_2 = int(p/np.cos(theta*np.pi/180))
    except ZeroDivisionError:
        y_2 = p
        x_2 = 0

    y_3 = height - 1
    x_3 = 0
    try:
        x_3 = int((p - y_3 * np.sin(theta*np.pi/180))/np.cos(theta * np.pi/180))
    except ZeroDivisionError:
        y_3 = p
        x_3 = 1

    # check which two of four pairs are within bounds
    points = np.zeros((4,2), dtype=np.uint8)
    k = 0
    if y_0 >= 0 and y_0 < height:
        points[k][0] = np.uint8(x_0)
        points[k][1] = np.uint8(y_0)
        k += 1

    if y_1 >= 0 and y_1 < height:
        points[k][0] = np.uint8(x_1)
        points[k][1] = np.uint8(y_1)
        k += 1

    if x_2 >= 0 and x_2 < width:
        points[k][0] = np.uint8(x_2)
        points[k][1] = np.uint8(y_2)
        k += 1

    if x_3 >= 0 and x_3 < width:
        points[k][0] = np.uint8(x_3)
        points[k][1] = np.uint8(y_3)
        k += 1
    return points


if __name__ == "__main__":
    img = cv2.imread("./hough/hough2.png")
    cannyEdgeImg = cv2.Canny(img,100,200)
    (height, width) = cannyEdgeImg.shape[:2]
    #print(cannyEdgeImg)
    p_max = int(np.sqrt(height**2 + width**2))
    houghImage = houghTransform(cannyEdgeImg, p_max)
    threshold = 40
    # now select the (p,theta) pairs such number of intersecting cosine curves in greater than threshold
    lineDict = {}
    (houghHeight, houghWidth) = houghImage.shape[:2]
    for p in range(houghHeight):
        for t in range(houghWidth):
            if houghImage[p][t] >= threshold:
                if p not in lineDict:
                    lineDict[p-p_max] = []
                lineDict[p-p_max].append(t)

    # now draw lines on original image
    print(lineDict)
    i = 0
    for p in lineDict:
        for theta in lineDict[p]:
            #get two points corresponding to this (p, theta) pair
            # p= x*cos(theta) + y*sin(theta)
            points = findPointsForLine(p, theta, width, height)
            #print("{}, {}".format(points[0][0], points[0][1]) )
            #print("{}, {}".format(points[1][0], points[1][1]))
            cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (255,255), 1)
    cv2.imshow("Edge Map", cannyEdgeImg)
    cv2.imshow("Hough Map", houghImage)
    cv2.imshow("image with lines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()