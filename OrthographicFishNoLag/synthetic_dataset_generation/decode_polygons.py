import matplotlib.pyplot as plt
import numpy as np
import cv2 

def polygons_to_binary_image(polygons, image_size):
    # Create a blank binary image (all zeros)
    binary_image = np.zeros(image_size, dtype=np.uint8)

    # Draw each polygon on the binary image
    for polygon in polygons:
        # Convert the flat list of coordinates to a NumPy array of shape (-1, 1, 2)
        points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        # Draw the polygon on the binary image (255 for white)
        cv2.fillPoly(binary_image, [points], color=255)

    return binary_image


contour = [[679, 422, 687, 423, 695, 430, 701, 450, 710, 462, 713, 462, 713, 454, 702, 438, 701, 432, 696, 427, 697, 426, 690, 420]]
#image_size = [602, 741]
image_size = [700, 1280]
mask = polygons_to_binary_image(contour, image_size)
plt.imshow(mask)
plt.show()

