import cv2
import numpy as np
import math
import time


def main():
    # sample = np.arange(100 * 3).reshape(10, 10, 3)
    # sample = block_shaped(sample, 4, 4)
    # p, t, y, u = sample.shape
    image = cv2.imread('source_image.jpg')
    image_height, image_width, S = image.shape
    m = 20
    n = 20

    transformed_pixels = 2 * image[:, :] / np.max(image) - 1
    pixels = block_shaped(transformed_pixels, m, n)
    pixel_blocks = np.split(pixels, pixels.shape[0], axis=0)
    pixel_blocks_shape = pixel_blocks[0].shape
    pixel_blocks_X = []
    for pb in pixel_blocks:
        pixel_blocks_X.append(pb.reshape(-1))

    N = m * n * S
    p = 20
    e = 0.005 * m * n * p
    e_ = e / p
    E = math.inf
    E_ = math.inf
    W = np.random.uniform(-1, 1, N * p).reshape(N, p)
    W_ = W.transpose()
    Wn = W
    W_n = W_
    dXs = np.empty((len(pixel_blocks_X), W.shape[0]))
    Eq = np.empty(len(pixel_blocks_X))
    Eq_ = np.empty(len(pixel_blocks_X))
    t = 0
    # alpha = 0.000001 * m * n * p
    # alpha_ = 0.001

    while E > e or E_ > e_:
        t += 1
        for i, X in enumerate(pixel_blocks_X):
            Y = np.apply_along_axis(lambda col: np.sum(col * X), axis=0, arr=W)
            X_ = np.apply_along_axis(lambda col: np.sum(col * Y), axis=0, arr=W_)
            Y_ = np.apply_along_axis(lambda col: np.sum(col * X_), axis=0, arr=W)

            alpha = 1 / np.sum(X_ * X_)
            alpha_ = 1 / np.sum(Y * Y)

            W = W - alpha * X_[:, np.newaxis] @ (Y_ - Y)[np.newaxis, :]
            W_ = W_ - alpha_ * Y[:, np.newaxis] @ (X_ - X)[np.newaxis, :]

            len_W = np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=1, arr=W)
            len_W_ = np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W_)

            W = W / len_W[:, np.newaxis]
            W_ = W_ / len_W_

            dX = X_ - X
            Eq[i] = np.sum(dX * dX, axis=0) / 2

            dY = Y_ - Y
            Eq_[i] = np.sum(dY * dY, axis=0) / 2
            # print("Error: {}. Error': {}".format(Eq[i], Eq_[i]))
            # time.sleep(0.5)
            # print("Error: {}. Error': {}".format(E, E_))
            # print(Eq[i])
            # print(i)
        E = np.sum(Eq, axis=0) / len(pixel_blocks_X)
        E_ = np.sum(Eq_, axis=0) / len(pixel_blocks_X)
        print("Error: {}. Error': {}".format(E, E_))


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


def block_shaped_test(arr, m, n):
    h = arr.shape[0]
    w = arr.shape[1]
    pivot_c = w - w % n
    start_c = w - n

    pivot_r = h - h % m
    start_r = h - m

    arr = np.hstack((arr[:, :pivot_c], arr[:, start_c:pivot_c], arr[:, pivot_c:]))
    arr = np.vstack((arr[:pivot_r, :], arr[start_r:pivot_r, :], arr[pivot_r:, :]))
    h, w = arr.shape

    return arr.reshape(h // m, m, -1, n).swapaxes(1, 2).reshape(-1, m, n)


if __name__ == '__main__':
    main()
