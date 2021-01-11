import numpy as np
import cv2 as cv

im_file = r"A:\repo\chess-sim\Chess Simulation\Images\board_0.png"
img = cv.imread(im_file)

# Changing the colour-space
LUV = cv.cvtColor(img, cv.COLOR_BGR2LUV)

# Find edges
edges = cv.Canny(LUV, 10, 100) # TODO params?
#cv.imshow("LUV", LUV)
#cv.waitKey(0)

# Find Contours
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#find biggest area, then get outmost pixels, refer to matlab script maybe
#any boundry rect type deal will give a straight rect, dont want that!
max_area = 400*400 # TODO
min_area = 9000
for c in contours:
    area = cv.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv.boundingRect(c)

        points = c.reshape(-1, 2)
        points_sum = points.sum(axis=1)
        points_diff = np.diff(points)
        top_left = np.argmin(points_sum)
        top_right = np.argmin(points_diff)
        bot_left = np.argmax(points_diff)
        bot_right = np.argmax(points_sum)

        pts = np.array([c[top_left], c[top_right], c[bot_right], c[bot_left]], np.int32)
        print(pts)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(img, [pts], True, (0, 255, 255), 2)

        #green
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#draw in blue the contours that were founded
cv.drawContours(img, contours, -1, 255, 1)
# find the biggest countour (c) by the area
c = max(contours, key=cv.contourArea)
print(cv.contourArea(c))
x,y,w,h = cv.boundingRect(c)

# draw the biggest contour (c) in red
#cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#cv.drawContours(img, contours, -1, (0,255,0), 3)
cv.imshow("drawContours", img)
cv.waitKey(0)
