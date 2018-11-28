import cv2
import numpy as np
import math


def main():
    # sample = np.arange(100 * 3).reshape(10, 10, 3)
    # sample = block_shaped(sample, 4, 4)
    # p, t, y, u = sample.shape
    image = cv2.imread('source_image.jpg')
    image_height, image_width, S = image.shape
    m = 5
    n = 5
    alpha = 0.01

    transformed_pixels = 2 * image[:, :] / np.max(image) - 1
    pixels = block_shaped(transformed_pixels, m, n)
    pixel_blocks = np.split(pixels, pixels.shape[0], axis=0)
    pixel_blocks_shape = pixel_blocks[0].shape
    pixel_blocks_X = []
    for pb in pixel_blocks:
        pixel_blocks_X.append(pb.reshape(-1))

    N = m * n * S
    p = 20
    e = 0.1
    E = math.inf
    W = np.random.uniform(-1, 1, N * p).reshape(N, p)
    W_ = W.transpose()
    Wn = W
    W_n = W_
    dXs = np.empty((len(pixel_blocks_X), W.shape[0]))
    Eq = np.empty(len(pixel_blocks_X))
    t = 0

    while E > e:
        t += 1
        for i, X in enumerate(pixel_blocks_X):
            X = X[np.newaxis, :]
            Y = X @ W
            X_ = Y @ W_
            dX = X_ - X
            dXs[i] = dX

            W_ = W_ - alpha * Y.transpose() @ dX
            W = W - alpha * X.transpose() @ dX @ W_.transpose()

            W = W / np.apply_along_axis(lambda col: np.max(np.abs(col)), axis=1, arr=W)[:, np.newaxis]
            W_ = W_ / np.apply_along_axis(lambda col: np.max(np.abs(col)), axis=1, arr=W_)[:, np.newaxis]

            Eq[i] = np.sum(dX * dX, axis=1) / len(pixel_blocks_X)
            # print(Eq[i])
            # print(i)
        E = np.sum(Eq, axis=0)
        print('Error==================================: {}'.format(E))


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
