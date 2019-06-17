import cv2
import numpy as np
img = cv2.imread('/home/grim/learning_image_classification/code/cracks/327.jpg',0)
cv2.imshow("normal",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
img = cv2.imread('/home/grim/learning_image_classification/code/cracks/327.jpg',0)
g_kernel = cv2.getGaborKernel((30, 30), 5.6, np.pi/4, 19 , -20, 0 , ktype=cv2.CV_32F)
filtered_img = cv2.filter2D(img, -1 , g_kernel)
cv2.imshow("filtered image",filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
