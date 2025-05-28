import cv2
import numpy as np

# Load image
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Apply Harris Corner Detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate corner points for better visibility
dst = cv2.dilate(dst, None)

# Threshold to mark corners in red
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# Show and save the result
cv2.imshow('Harris Corners', image)
cv2.imwrite('harris_output.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
