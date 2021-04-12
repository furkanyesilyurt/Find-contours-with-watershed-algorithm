import cv2
import numpy as np
import matplotlib.pyplot as plt

#FIND CONTOURS
#import image
coins = cv2.imread("coins.jpg")
# plt.figure(), plt.imshow(coins), plt.axis("off")

#blurring
coins_blur = cv2.medianBlur(coins, ksize = 13)
# plt.figure(), plt.imshow(coins_blur), plt.axis("off")

#gray scale
coins_gray = cv2.cvtColor(coins_blur, cv2.COLOR_BGR2GRAY)
# plt.figure(), plt.imshow(coins_gray, cmap = "gray"), plt.axis("off")

#threshold
ret, coin_tresh = cv2.threshold(coins_gray,80, 255, cv2.THRESH_BINARY)
# plt.figure(), plt.imshow(coin_tresh, cmap = "gray"), plt.axis("off")

#contour
contours, hierarchy = cv2.findContours(coin_tresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# plt.figure(), plt.imshow(coin_tresh, cmap = "gray"), plt.axis("off")

for i in range(len(contours)):
    
    if hierarchy[0][i][3]:
        cv2.drawContours(coins, contours, i, (255,0,0),10)
        
plt.figure(), plt.imshow(coins), plt.axis("off")


#%%  WATERSHED

#import image
coins = cv2.imread("coins.jpg")
# plt.figure(), plt.imshow(coins), plt.axis("off")

#blurring
coins_blur = cv2.medianBlur(coins, ksize = 13)
# plt.figure(), plt.imshow(coins_blur), plt.axis("off")

#gray scale
coins_gray = cv2.cvtColor(coins_blur, cv2.COLOR_BGR2GRAY)
# plt.figure(), plt.imshow(coins_gray, cmap = "gray"), plt.axis("off")

#threshold
ret, coin_tresh = cv2.threshold(coins_gray,80, 255, cv2.THRESH_BINARY)
# plt.figure(), plt.imshow(coin_tresh, cmap = "gray"), plt.axis("off")

#morfolojik operasyon - açılma
kernel = np.ones((3,3),np.uint8)
openning = cv2.morphologyEx(coin_tresh, cv2.MORPH_OPEN, kernel, iterations = 2)
# plt.figure(), plt.imshow(openning, cmap = "gray"), plt.axis("off")

#distances
dist_transform = cv2.distanceTransform(openning, cv2.DIST_L2, 5)
# plt.figure(), plt.imshow(dist_transform, cmap = "gray"), plt.axis("off")

#make smaller the image
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform), 255, 0)
# plt.figure(), plt.imshow(sure_foreground, cmap = "gray"), plt.axis("off")

#make bigger the image
sure_background = cv2.dilate(openning, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background, sure_foreground)
# plt.figure(), plt.imshow(unknown, cmap = "gray"), plt.axis("off")

#connections
ret, marker = cv2.connectedComponents(sure_foreground)
marker += 1
marker[unknown == 255] = 0
# plt.figure(), plt.imshow(marker, cmap = "gray"), plt.axis("off")

#watershed
marker = cv2.watershed(coins, marker)
# plt.figure(), plt.imshow(marker, cmap = "gray"), plt.axis("off")

#contour
contours, hierarchy = cv2.findContours(marker, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3]:
        cv2.drawContours(coins, contours, i, (255,0,0),10)


#FINAL IMAGE
plt.figure(), plt.imshow(coins), plt.axis("off")