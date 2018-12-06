import cv2
import numpy as np
import math
from pathlib import Path
import json


def main():
    np.random.seed(1)

    plot_data = {}
    t_on_p = {'p': [], 't': []}
    t_on_im = {'im': [], 't': []}
    t_on_e = {'e': [], 't': []}
    plot_data['t_on_p'] = t_on_p
    plot_data['t_on_im'] = t_on_im
    plot_data['t_on_e'] = t_on_e

    with open('report_info.txt', 'w') as f:
        print('Different p', file=f)
        image_name = Path('images') / '1lina32x32.jpg'
        P = [50, 60, 70]
        e = 5
        m = 5
        n = 5
        N = m * n * 3

        h, w, s, block_shape, source_shape, pixels = prepare_image(image_name, m, n)
        for p in P:
            E, t, Z, L, W, W_ = train(pixels, N, p, e, image_name, f)
            t_on_p['p'].append(p)
            t_on_p['t'].append(t)
            restore_image(W, W_, pixels, block_shape, source_shape, h, w, m, n, image_name, N, L, e, p)
            print('=====================', file=f)

        print('\n\nDifferent images', file=f)
        image_names = sorted(Path('images').glob('*.jpg'))
        p = 50
        e = 15
        m = 5
        n = 5
        N = m * n * 3

        for image_name in image_names:
            h, w, s, block_shape, source_shape, pixels = prepare_image(image_name, m, n)
            E, t, Z, L, W, W_ = train(pixels, N, p, e, image_name, f)
            t_on_im['im'].append(image_name.name)
            t_on_im['t'].append(t)
            restore_image(W, W_, pixels, block_shape, source_shape, h, w, m, n, image_name, N, L, e, p)
            print('=====================', file=f)

        print('\n\nDifferent e', file=f)
        image_name = Path('images') / '1lina32x32.jpg'
        p = 50
        e = [10, 15, 20]
        m = 5
        n = 5
        N = m * n * 3

        h, w, s, block_shape, source_shape, pixels = prepare_image(image_name, m, n)
        for _e in e:
            E, t, Z, L, W, W_ = train(pixels, N, p, _e, image_name, f)
            t_on_e['e'].append(_e)
            t_on_e['t'].append(t)
            restore_image(W, W_, pixels, block_shape, source_shape, h, w, m, n, image_name, N, L, _e, p)
            print('=====================', file=f)

    with open('plots.json', 'w') as f:
        json.dump(plot_data, f)


def prepare_image(image_name, m, n):
    image = cv2.imread(str(image_name))
    h, w, s = image.shape
    image = image.astype('float32')
    transformed_pixels = 2 * image / 255 - 1
    source_shape, pixels = block_shaped(transformed_pixels, m, n)
    block_shape = pixels[0].shape
    pixels = pixels.reshape(pixels.shape[0], -1)
    return h, w, s, block_shape, source_shape, pixels


def train(pixels, N, p, e, image_name, file):
    W = np.random.uniform(-1, 1, N * p).reshape(N, p)
    W_ = W.copy().transpose()
    E = math.inf
    L = len(pixels)
    t = 0

    while E > e:
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

        E = np.sum(np.apply_along_axis(lambda row: np.sum((W_.T @ W.T @ row - row) ** 2) / 2, axis=1, arr=pixels))
        # print("E:{}, avgE:{}".format(E, E / L))

    Z = (N * L) / ((N + L) * p + 2)
    print('Image={}'.format(image_name), file=file)
    print('Iterations={}'.format(t), file=file)
    print('Edest={}'.format(e), file=file)
    print('E={}'.format(E), file=file)
    print('P={}'.format(p), file=file)
    print('L={}'.format(L), file=file)
    print('N={}'.format(N), file=file)
    print('Z={}'.format(Z), file=file)
    model_folder = Path('models')
    readable_model_folder = model_folder / 'readable'
    name_model = 'IMAGE{}_N{}_L{}_E{}_P{}model'.format(image_name.name, N, L, e, p)
    name_model_ = 'IMAGE{}_N{}_L{}_E{}_P{}model_'.format(image_name.name, N, L, e, p)
    np.save(model_folder / name_model, W)
    np.save(model_folder / name_model_, W)
    with open(readable_model_folder / name_model, 'wt') as f:
        print('W', file=f)
        for row in W:
            print(str(row).replace('\n', ' '), file=f)
    with open(readable_model_folder / name_model_, 'wt') as f:
        print("W'", file=f)
        for row in W_:
            print(str(row).replace('\n', ' '), file=f)
    return E, t, Z, L, W, W_


def restore_image(W, W_, pixels, block_shape, source_shape, h, w, m, n, image_name, N, L, e, p):
    mm_W_W_ = W_.T @ W.T
    Y = np.array([mm_W_W_ @ X for X in pixels])
    restored_pixels = Y.reshape(pixels.shape[0], block_shape[0], block_shape[1], block_shape[2])
    restored_merged_pixels = block_unshaped(restored_pixels, source_shape, h, w, m, n)
    restored_merged_pixels = 255 * (restored_merged_pixels + 1) / 2
    restored_merged_pixels = np.rint(restored_merged_pixels)
    cv2.imwrite('restored_images/IMAGE{}_N{}_L{}_E{}_P{}restored.jpg'.format(image_name.name, N, L, e, p), restored_merged_pixels)


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
    arr = arr.reshape(source_shape[0] // m, source_shape[1] // m, m, n, source_shape[2]).swapaxes(1, 2).reshape(source_shape)

    pivot_c = w - w % n
    start_c = w - n

    pivot_r = h - h % m
    start_r = h - m

    arr = np.hstack((arr[:, :pivot_c], arr[:, pivot_c + pivot_c - start_c:]))
    arr = np.vstack((arr[:pivot_r, :], arr[pivot_r + pivot_r - start_r:, :]))

    return arr


if __name__ == '__main__':
    main()
