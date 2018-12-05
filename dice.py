import cv2
import numpy as np
import os
 
minThreshold = 100
maxThreshold = 200
minArea = 100
minCircularity = .7
minInertiaRatio = .6
minConvexity = .1

def diceDetect(filename):
  im = cv2.imread(filename)
  # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  params = cv2.SimpleBlobDetector_Params()
  params.minThreshold = minThreshold
  params.maxThreshold = maxThreshold
  params.filterByColor = True
  params.blobColor = 255
  params.filterByArea = True
  params.minArea = minArea
  params.filterByCircularity = True
  params.minCircularity = minCircularity
  params.filterByInertia = True
  params.minInertiaRatio = minInertiaRatio
  params.filterByConvexity = True
  params.minConvexity = minConvexity

  detector = cv2.SimpleBlobDetector_create(params)
  keypoints = detector.detect(im)

  for k in keypoints:
    (x,y) = k.pt
    x=int(round(x))
    y=int(round(y))
    size = int(round(k.size) / 2)
    cv2.circle(im, (x,y), size, (0, 0, 255), 5)

  # here we draw keypoints on the frame.
  # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
  #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  number = len(keypoints)
  text = "Number is: {0}".format(number)
  cv2.putText(im, text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
  print(text)
  cv2.imshow("Dice Reader", im)
  cv2.waitKey(1000)

imgSrc = 'sources'
for file in os.listdir(imgSrc):
  diceDetect("{0}/{1}".format(imgSrc, file))

cv2.destroyAllWindows()
 