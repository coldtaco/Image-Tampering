from PIL import Image
import numpy as np
# from math import erfc
import matplotlib.pyplot as plt
from scipy.special import erfc

from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a, type = 1), type=1, axis = 0)

def get_acquired(img):
    height, width = img.shape
    even_row_x = (np.arange(width) % 2) == 0
    odd_row_x = (np.arange(width) % 2) == 1

    acquired_selector = np.concatenate([even_row_x if x%2 == 0 else odd_row_x for x in range(height)], axis = 0, dtype=bool)

    return acquired_selector

def est_interpolation_kernel(img) -> np.ndarray:
    """
        Returns interpolation kernel with N = 2    
    """

    height, width, channel = img.shape

    greens = img[:,:,1]
    greens_flat = greens.reshape(-1)

    acquired_selector = get_acquired(greens)

    ys = np.reshape(greens, (-1))
    xs = np.arange(ys.shape[0]).reshape(-1,1)
    xs = np.repeat(xs, 12, axis = 1)

    # Training data uses interpolated pixels, inputs are acquired in relation to interpolated
    xs, ys = xs[~acquired_selector], ys[~acquired_selector]


    # Modifiers to flattened image for indexing
    row_n = width
    col_n = 1

    x_modifier = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            if (x + y) % 2 == 0:continue
            x_modifier.append(row_n * x + col_n * y)

    matrix_x_selector = np.tile(x_modifier, (xs.shape[0], 1))

    xs += matrix_x_selector
    # Mask for values < 0, meaning that values do not exist
    xs[xs >= (height * width)] = -1
    negative_mask = (xs < 0)

    # Mask values with negative index
    xs = greens_flat[xs]
    xs[negative_mask] = 0

    h_prime = np.linalg.inv(xs.T @ xs) @ xs.T @ ys
    
    return h_prime

def estimate_img(img):
    kernel = est_interpolation_kernel(img)
    height, width, channel = img.shape

    greens = img[:,:,1]
    greens_flat = greens.reshape(-1)

    acquired_selector = get_acquired(greens)
    # greens_flat[acquired_selector] = 0
    print(greens_flat)

    ys = np.reshape(greens, (-1))
    xs = np.arange(ys.shape[0]).reshape(-1,1)
    xs = np.repeat(xs, 12, axis = 1)

    # Training data uses interpolated pixels, inputs are acquired in relation to interpolated
    # xs, ys = xs[~acquired_selector], ys[~acquired_selector]

    # Modifiers to flattened image for indexing
    row_n = width
    col_n = 1

    x_modifier = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            if (x + y) % 2 == 0:continue
            x_modifier.append(row_n * x + col_n * y)

    matrix_x_selector = np.tile(x_modifier, (xs.shape[0], 1))

    xs += matrix_x_selector

    # Mask for values < 0, meaning that values do not exist
    xs[xs >= (height * width)] = -1
    negative_mask = (xs < 0)

    xs = greens_flat[xs]
    # Mask values with negative index
    xs[negative_mask] = 0

    new_val = xs @ kernel
    # greens_flat[~acquired_selector] = new_val
    new_val[acquired_selector] = 0

    return new_val.reshape((height, width))

def get_test_mod(img):
    """
        Returns ground truth indexes for 2-pixel neighbour
    """
    h,w,c = img.shape
    
    test_img = np.arange(h*w*1).reshape(h, w)
    test_mod = [test_img[0, 2],
                test_img[1, 1],
                test_img[1, 2],
                test_img[1, 3],
                test_img[2, 0],
                test_img[2, 1],
                test_img[2, 3],
                test_img[2, 4],
                test_img[3, 1],
                test_img[3, 2],
                test_img[3, 3],
                test_img[4, 2],
                ]
    test_mod = np.asarray(test_mod)
    test_mod -= test_img[2,2]
    print(test_mod)

def tamper_prob(x, std):
    return erfc(np.abs(x)/(std*np.sqrt(2)))
    # return erfc(x)

def get_Stats(greens):
    height, width, = greens.shape

    actual_greens = np.empty_like(greens, dtype = bool)

    row_mask = {
        # Key = x%2, 0 = Even, 1 = Odd
        0: ~((np.arange(width) % 2).astype(bool)),
        1: ((np.arange(width) % 2).astype(bool)),
    }

    # Filter out interpolated greens
    for x in range(actual_greens.shape[0]):
        is_even = (x % 2)
        actual_greens[x] = row_mask[is_even]

    std_a, std_i = greens[actual_greens].std(), greens[~actual_greens].std()
    mean_a, mean_i = greens[actual_greens].mean(), greens[~actual_greens].mean()

    print(f'Acquired: {std_a}, Interpolated: {std_i}')
    print(f'Acquired: {mean_a}, Interpolated: {mean_i}')

def check_tamper(greens):
    height, width, = greens.shape

    ys = np.reshape(greens, (-1))
    acquired_greens = get_acquired(img).reshape(height,width)
    
    result = np.ones_like(greens)

    std_a, std_i = greens[acquired_greens].std(), greens[~acquired_greens].std()
    mean_a, mean_i = greens[acquired_greens].mean(), greens[~acquired_greens].mean()
    print(std_a, std_i, mean_a, mean_i)

    result[acquired_greens] = tamper_prob(greens[acquired_greens], std_i)
    result[~acquired_greens] = tamper_prob(greens[~acquired_greens], std_a)
    print(result)
    B = 16
    for x in range(0, width, B):
        for y in range(0, height, B):
            pass
            result[y:min(height, y+B), x:min(width, x+B)] = (dct2(result[y:min(height, y+B), x:min(width, x+B)]))[-1,-1]
    result /= 2*(B-1)
    print(result)

    return result


if __name__ == '__main__':    
    img = Image.open('data\TRAINING_CG-1050\TRAINING\TAMPERED\Im1_f2.jpg')
    img = Image.open('data\TRAINING_CG-1050\TRAINING\ORIGINAL\Im1_2.jpg')
    img = Image.open('b.jpg')
    img = Image.open('image-012.jpg')
    # img = Image.open('xnkvgttxk01a1.jpg')
    img = np.asarray(img)
    greens = img[:,:,1]

    img = estimate_img(img)
    get_Stats(greens - img)
    # exit()
    # plt.imshow(greens - img, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.hist((img-greens).reshape(-1))
    # exit()

    # print(kernel)
    res = check_tamper(greens - img)
    cutoff = dict(zip(['vmin', 'vmax'], np.percentile(res, (5,95))))
    plt.imshow(res, cmap='gray',**cutoff)
    plt.colorbar()
    plt.show()

