from pycocotools import mask
import matplotlib.pyplot as plt

rle_counts = [353022, 1, 599, 4, 598, 4, 599, 4, 599, 4, 598, 4, 599, 4, 599, 5, 597, 5, 597, 6, 596, 6, 597, 6, 596, 6, 597, 5, 598, 4, 598, 5, 597, 5, 596, 7, 596, 7, 596, 6, 597, 6, 596, 6, 597, 7, 596, 6, 597, 5, 596, 7, 595, 7, 595, 7, 596, 7, 595, 8, 594, 8, 595, 8, 594, 8, 594, 9, 594, 9, 594, 9, 593, 11, 592, 11, 591, 11, 591, 12, 590, 12, 590, 13, 589, 13, 589, 13, 590, 12, 591, 11, 591, 11, 592, 10, 593, 8, 596, 5, 63537]
size = [602, 741]
rle = {'counts': rle_counts, 'size': size}
compressed_rle = mask.frPyObjects(rle, size[0], size[1])
decoded_mask = mask.decode(compressed_rle)
plt.imshow(decoded_mask, cmap='gray')
plt.show()
