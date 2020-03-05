import matplotlib.pyplot as plt
import matplotlib.rcsetup
import numpy as np
import cv2
import os

file_path = r"C:\Users\s167917\Documents\_School_\Jaar 4\3 CS Medical Image Analysis\project\results\threshold_vs_dice.txt"
data = np.loadtxt(file_path)
x = data[0]
thresholds = np.zeros(6)
matplotlib.rcParams.update({'font.size':16})
fig = plt.figure(figsize=(19.2, 10.8))
for i, y in enumerate(data[1:]):
    plt.plot(x, y)
    max_index = int((np.argmax(y) + len(x) - 1 - np.argmax(y[::-1])) / 2)
    thresholds[i] = x[max_index]
    print(f"{np.argmax(y)}_{len(x) - 1 - np.argmax(y[::-1])}_{max_index}_{x[max_index]}")
    plt.scatter(x[max_index], y[max_index])

plt.ylabel("Dice overlap")
plt.xlabel("Threshold")
plt.legend((f"Default ({thresholds[0]})",
            f"NMI norm ({thresholds[1]})",
            f"NMI norm squared ({thresholds[2]})",
            f"Pre ({thresholds[3]})",
            f"Pre norm ({thresholds[4]})",
            f"Pre norm squared ({thresholds[5]})"), loc="lower left")

img_path = f"{file_path[:-3]}png"
plt.savefig(img_path)

def remove_white_borders(input_img_path):
    img = cv2.imread(input_img_path)
    summed_img = np.sum(img, 2)
    negative = np.max(summed_img) - summed_img
    ones_x = np.where((np.sum(negative, 0) > 0) == 1)
    ones_y = np.where((np.sum(negative, 1) > 0) == 1)
    return img[np.min(ones_y):np.max(ones_y) + 1, np.min(ones_x):np.max(ones_x) + 1, :]

cv2.imwrite(img_path, remove_white_borders(img_path))
os.startfile(img_path)
