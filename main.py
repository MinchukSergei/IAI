import cv2
import numpy as np


def main():
    m = 5
    n = 5
    image = cv2.imread('source_image.jpg')
    h, w, d = image.shape

    # sample = np.arange(100).reshape(10, 10)
    # sample = np.hstack((sample[:, :8], sample[:, -4:-2], sample[:, -2:]))
    # sample = blockshaped(sample, 4, 4)

    pixels = block_shaped(pixels, m, n)
    transformed_pixels = 2 * pixels[:, :] / np.max(pixels) - 1

    h = pixels.shape[0]
    w = pixels.shape[1]

    pass


def block_shaped(arr, m, n):
    h = arr.shape[0]
    w = arr.shape[1]
    pivot_c = w - w % n
    start_c = w - n

    pivot_r = h - h % m
    start_r = h - m

    arr = np.hstack((arr[:, :pivot_c], arr[:, start_c:pivot_c], arr[:, pivot_c:]))
    arr = np.vstack((arr[:pivot_r, :], arr[start_r:pivot_r, :], arr[pivot_r:, :]))
    h, w, d = arr.shape

    return arr.reshape(h // m, m, -1, n, d).swapaxes(1, 2).reshape(-1, m, n, d)


if __name__ == '__main__':
    main()
