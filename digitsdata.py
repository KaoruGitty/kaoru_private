import sklearn.datasets
import matplotlib.pyplot as plt

digits = sklearn.datasets.load_digits()

for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.axis("off")
    plt.title(digits.target[i])
    plt.imshow(digits.images[i], cmap="Grays")
plt.show()

"""
print("データの個数=",len(digits.images))
print("画像データ=",digits.images[0])
print("何の数字か=",digits.target[0])
"""