import matplotlib.pyplot as plt
import numpy as np

file_path = r"threshold_dice.txt"
data = np.loadtxt(file_path)
x = data[0]
thresholds = np.zeros(6)
for i, y in enumerate(data[1:]):
    plt.plot(x, y)
    max_index = int((np.argmax(y) + len(x) - 1 - np.argmax(y[::-1])) / 2)
    thresholds[i] = x[max_index]
    print(f"{np.argmax(y)}_{len(x) - 1 - np.argmax(y[::-1])}_{max_index}_{x[max_index]}")
    plt.scatter(x[max_index], y[max_index])

plt.ylabel("dice score")
plt.xlabel("threshold")
plt.legend((f"default ({thresholds[0]})",
            f"normalized normalized mutual information ({thresholds[1]})",
            f"normalized normalized mutual information squared ({thresholds[2]})",
            f"predetermined ({thresholds[3]})",
            f"normalized predetermined ({thresholds[4]})",
            f"normalized predetermined squared ({thresholds[5]})"))
plt.show()
