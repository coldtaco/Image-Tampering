from PIL import Image
import numpy as np
# from math import erfc
import matplotlib.pyplot as plt
from scipy.special import erfc

from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def est_interpolation_kernel(img) -> np.ndarray:
    """
        Returns interpolation kernel with N = 2    
    """

    height, width, channel = img.shape

    greens = img[:,:,1]
    ys = np.reshape(greens, (-1))
    actual_greens = np.array(greens)

    row_mask = {
        # Key = x%2, 0 = Even, 1 = Odd
        0: ~(np.arange(width) % 2),
        1: (np.arange(width) % 2),
    }

    # Filter out interpolated greens
    for x in range(actual_greens.shape[0]):
        is_even = x % 2
        actual_greens[x, row_mask[is_even]] = 0

    xs = np.arange(ys.shape[0]).reshape(-1,1)
    print(xs.shape)
    xs = np.repeat(xs, 12, axis = 1)

    # Modifiers to flattened image for indexing
    row_n = width
    col_n = 1

    x_modifier = [-row_n * 2,
                  -row_n - col_n,
                  -row_n,
                  -row_n + col_n,
                  -2 * col_n,
                  -1 * col_n,
                  1 * col_n,
                  2 * col_n,
                  row_n - col_n,
                  row_n,
                  row_n + col_n,
                  2 * row_n,
                  ]
    # One row for every pixel
    x_modifier = np.asarray(x_modifier)
    # Shift columns to correct index for 2-neighbour pixel
    xs += x_modifier
    # Mask for values < 0, meaning that values do not exist
    xs[xs >= (height * width)] = -1
    negative_mask = (xs < 0)
    xs = greens.reshape(-1)[xs]

    # Mask values with negative index
    xs[negative_mask] = 0

    h_prime = np.linalg.inv(xs.T @ xs) @ xs.T @ ys
    
    return h_prime

def estimate_img(img):
    kernel = est_interpolation_kernel(img)
    height, width, channel = img.shape

    greens = img[:,:,1]
    ys = np.reshape(greens, (-1))
    g_hat = np.array(greens)

    row_mask = {
        # Key = x%2, 0 = Even, 1 = Odd
        0: ~(np.arange(width) % 2),
        1: (np.arange(width) % 2),
    }

    # Filter out acquired greens
    for x in range(g_hat.shape[0]):
        is_even = (x % 2) == 1
        g_hat[x, row_mask[is_even]] = 0

    xs = np.arange(ys.shape[0]).reshape(-1,1)
    print(xs.shape)
    xs = np.repeat(xs, 12, axis = 1)

    # Modifiers to flattened image for indexing
    row_n = width
    col_n = 1

    x_modifier = [-row_n * 2,
                  -row_n - col_n,
                  -row_n,
                  -row_n + col_n,
                  -2 * col_n,
                  -1 * col_n,
                  1 * col_n,
                  2 * col_n,
                  row_n - col_n,
                  row_n,
                  row_n + col_n,
                  2 * row_n,
                  ]

    x_modifier = np.asarray(x_modifier)
    # Shift columns to correct index for 2-neighbour pixel
    xs += x_modifier
    # Mask for values < 0, meaning that values do not exist
    xs[xs >= (height * width)] = -1
    negative_mask = (xs < 0)
    xs = ys.reshape(-1)[xs]

    # Mask values with negative index
    xs[negative_mask] = 0

    new_val = xs @ kernel
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
    # return erfc(np.abs(x)/(std*np.sqrt(2)))
    return erfc(x)

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

    print((actual_greens))
    print((~actual_greens))

    std_a, std_i = greens[actual_greens].std(), greens[~actual_greens].std()
    mean_a, mean_i = greens[actual_greens].mean(), greens[~actual_greens].mean()

    print(f'Acquired: {std_a}, Interpolated: {std_i}')
    print(f'Acquired: {mean_a}, Interpolated: {mean_i}')

def check_tamper(greens):
    height, width, = greens.shape

    ys = np.reshape(greens, (-1))
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
    print(actual_greens)
    
    result = np.empty_like(greens)

    std_a, std_i = greens[actual_greens].std(), greens[~actual_greens].std()
    mean_a, mean_i = greens[actual_greens].mean(), greens[~actual_greens].mean()
    print(std_a, std_i, mean_a, mean_i)

    result[actual_greens] = tamper_prob(greens[actual_greens] - mean_a, std_a)
    result[~actual_greens] = tamper_prob(greens[~actual_greens] - mean_i, std_i)
    print(width, height)
    B = 16
    for x in range(0, width, B):
        for y in range(0, height, B):
            pass
    #         result[y:min(height, y+B), x:min(width, x+B)] = dct2(result[y:min(height, y+B), x:min(width, x+B)])
    # result /= 2*(B-1)

    return result


if __name__ == '__main__':    
    img = Image.open('data\TRAINING_CG-1050\TRAINING\TAMPERED\Im1_col2.jpg')
    img = Image.open('image-012.jpg')
    img = np.asarray(img)/255
    greens = img[:,:,1]

    img = estimate_img(img)
    # get_Stats(greens - img)
    # exit()
    # plt.imshow(img - greens, cmap='gray', vmin=0, vmax=1)
    plt.hist((img-greens).reshape(-1))
    plt.show()
    exit()

    # print(kernel)
    res = check_tamper(greens - img)
    print(res)
    plt.imshow(res, cmap='gray', vmin=0, vmax=1)
    plt.show()

