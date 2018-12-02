import cv2
import numpy as np
import math
from pathlib import Path


def main():
    image_name = Path('images') / '1lina32x32.jpg'
    P = [30, 40, 50]
    e = 5
    m = 5
    n = 5
    N = m * n * 3

    h, w, s, block_shape, source_shape, pixels = prepare_image(image_name, m, n)
    for p in P:
        E, t, Z, L, W, W_ = train(pixels, N, p, e, image_name)
        restore_image(W, W_, pixels, block_shape, source_shape, h, w, m, n, image_name, N, L, E, p)
        print('=====================')


def prepare_image(image_name, m, n):
    image = cv2.imread(str(image_name))
    h, w, s = image.shape
    image = image.astype('float32')
    transformed_pixels = 2 * image / 255 - 1
    source_shape, pixels = block_shaped(transformed_pixels, m, n)
    block_shape = pixels[0].shape
    pixels = pixels.reshape(pixels.shape[0], -1)
    return h, w, s, block_shape, source_shape, pixels


def train(pixels, N, p, e, image_name):
    W = np.random.uniform(-1, 1, N * p).reshape(N, p)
    W_ = W.copy().transpose()
    E = math.inf
    E_ = math.inf
    L = len(pixels)
    Eq = np.empty(L)
    Eq_ = np.empty(L)
    t = 0

    while E > e or E_ > e:
        t += 1
        for i, X in enumerate(pixels):
            Y = W.T @ X
            X_ = W_.T @ Y
            Y_ = W.T @ X_

            alpha = 1 / math.pow(np.sum(X_ * X_), 2)
            alpha_ = 1 / math.pow(np.sum(Y * Y), 2)

            W = W - alpha * (X_[:, np.newaxis] @ (Y_ - Y)[np.newaxis, :])
            W_ = W_ - alpha_ * (Y[:, np.newaxis] @ (X_ - X)[np.newaxis, :])

            W = W / np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W)
            W_ = W_ / np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W_)

            dX = X_ - X
            Eq[i] = np.sum(dX * dX, axis=0) / 2

            dY = Y_ - Y
            Eq_[i] = np.sum(dY * dY, axis=0) / 2

        E = np.sum(Eq, axis=0)
        E_ = np.sum(Eq_, axis=0)

    print('Image={}'.format(image_name))
    print('Iterations={}'.format(t))
    print('E={}'.format(E))
    Z = (N * L) / ((N + L) * p + 2)
    print('Z={}'.format(Z))
    model_folder = Path('models')
    readable_model_folder = model_folder / 'readable'
    name_model = 'IMAGE{}_N{}_L{}_E{}_P{}model'.format(image_name, N, L, E, p)
    name_model_ = 'IMAGE{}_N{}_L{}_E{}_P{}model_'.format(image_name, N, L, E, p)
    np.save(model_folder / name_model, W)
    np.save(model_folder / name_model_, W)
    with open(readable_model_folder / name_model, 'wt') as f:
        print('W', file=f)
        print([col for col in W], file=f)
    with open(readable_model_folder / name_model, 'wt') as f:
        print("W'", file=f)
        print([col for col in W_], file=f)
    return E, t, Z, L, W, W_


def restore_image(W, W_, pixels, block_shape, source_shape, h, w, m, n, image_name, N, L, E, p):
    mm_W_W_ = W_.T @ W.T
    Y = np.array([mm_W_W_ @ X for X in pixels])
    restored_pixels = Y.reshape(pixels.shape[0], block_shape[0], block_shape[1], block_shape[2])
    restored_merged_pixels = block_unshaped(restored_pixels, source_shape, h, w, m, n)
    restored_merged_pixels = 255 * (restored_merged_pixels + 1) / 2
    restored_merged_pixels = np.rint(restored_merged_pixels)
    cv2.imwrite('restored_images/IMAGE{}_N{}_L{}_E{}_P{}restored.jpg'.format(image_name, N, L, E, p), restored_merged_pixels)


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

    return arr.shape, arr.reshape(h // m, m, -1, n, d).swapaxes(1, 2).reshape(-1, m, n, d)


def block_unshaped(arr, source_shape, h, w, m, n):
    arr = arr.reshape(source_shape[0] // m, source_shape[1] // m, m, n, 3).swapaxes(1, 2).reshape(source_shape)

    pivot_c = w - w % n
    start_c = w - n

    pivot_r = h - h % m
    start_r = h - m

    arr = np.hstack((arr[:, :pivot_c], arr[:, pivot_c + pivot_c - start_c:]))
    arr = np.vstack((arr[:pivot_r, :], arr[pivot_r + pivot_r - start_r:, :]))

    return arr


if __name__ == '__main__':
    main()
