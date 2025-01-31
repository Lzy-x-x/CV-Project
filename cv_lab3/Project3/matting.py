import sys
from pathlib import Path
import numpy as np
import cv2
from numba import jit
from argparse import ArgumentParser
import warnings
import scipy.ndimage

from orchard_bouman_clust import clustFunc

def matte_sad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_sad = np.sum(np.abs(pred_matte - gt_matte))    
    return error_sad

def matte_mad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mad = np.mean(np.abs(pred_matte - gt_matte))    
    return error_mad

def matte_mse(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mse = np.mean(np.power(pred_matte - gt_matte, 2))    
    return error_mse


def matte_grad(pred_matte, gt_matte):
    assert(len(pred_matte.shape) == len(gt_matte.shape))
    # alpha matte 的归一化梯度，标准差 =1.4，1 阶高斯导数的卷积
    predict_grad = scipy.ndimage.filters.gaussian_filter(pred_matte, 1.4, order=1) 
    gt_grad = scipy.ndimage.filters.gaussian_filter(gt_matte, 1.4, order=1)
    error_grad = np.sum(np.power(predict_grad - gt_grad, 2))
    return error_grad


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:  # normalize
        h /= sumh
    return h

# returns the surrounding N-rectangular neighborhood of matrix m, centered
# at pixel (x,y), (odd valued N)
@jit(nopython=True, cache=True)
def get_window(m, x, y, N):
    h, w, c = m.shape
    halfN = N // 2
    r = np.zeros((N, N, c))
    xmin = max(0, x - halfN)
    xmax = min(w, x + (halfN + 1))
    ymin = max(0, y - halfN)
    ymax = min(h, y + (halfN + 1))
    pxmin = halfN - (x - xmin)
    pxmax = halfN + (xmax - x)
    pymin = halfN - (y - ymin)
    pymax = halfN + (ymax - y)

    r[pymin:pymax, pxmin:pxmax] = m[ymin:ymax, xmin:xmax]
    return r

@jit(nopython=True, cache=True)
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, sigma_C, alpha_init, maxIter, minLike):
    I = np.eye(3)
    FMax = np.zeros(3)
    BMax = np.zeros(3)
    alphaMax = 0
    maxlike = - np.inf
    invsgma2 = 1 / sigma_C ** 2
    for i in range(mu_F.shape[0]):
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])
        for j in range(mu_B.shape[0]):
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            myiter = 1
            lastLike = -1.7977e+308
            while True:
                # solve for F,B
                A11 = invSigma_Fi + I * alpha ** 2 * invsgma2
                A12 = I * alpha * (1 - alpha) * invsgma2
                A22 = invSigma_Bj + I * (1 - alpha) ** 2 * invsgma2
                A = np.vstack((np.hstack((A11, A12)), np.hstack((A12, A22))))
                b1 = invSigma_Fi @ mu_Fi + C * (alpha) * invsgma2
                b2 = invSigma_Bj @ mu_Bj + C * (1 - alpha) * invsgma2
                b = np.atleast_2d(np.concatenate((b1, b2))).T

                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                # solve for alpha

                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T - B).T @ (F - B)) / np.sum((F - B) ** 2)))[0, 0]
                # calculate likelihood
                L_C = - np.sum((np.atleast_2d(C).T - alpha * F - (1 - alpha) * B) ** 2) * invsgma2
                L_F = (- ((F - np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F - np.atleast_2d(mu_Fi).T)) / 2)[0, 0]
                L_B = (- ((B - np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B - np.atleast_2d(mu_Bj).T)) / 2)[0, 0]
                like = (L_C + L_F + L_B)

                if like > maxlike:
                    alphaMax = alpha
                    maxLike = like
                    FMax = F.ravel()
                    BMax = B.ravel()

                if myiter >= maxIter or abs(like - lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1
    return FMax, BMax, alphaMax

def bayesian_matte(img, trimap, sigma=8, N=25, minN=9, minN_reduction=0):
    # check minN_reduction parameter
    if minN_reduction >= minN:
        raise ValueError("minN_reduction parameter must be less than minN")

    img = img / 255

    h, w, c = img.shape  # height, width, channel
    alpha = np.zeros((h, w))

    fg_mask = trimap == 255
    bg_mask = trimap == 0
    unknown_mask = True ^ np.logical_or(fg_mask, bg_mask)
    foreground = img * np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)
    background = img * np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)

    gaussian_weights = matlab_style_gauss2d((N, N), sigma)  # gaussian filter
    gaussian_weights = gaussian_weights / np.max(gaussian_weights)  # normalize

    alpha[fg_mask] = 1  # set alpha to 1 for foreground, means white
    F = np.zeros(img.shape)  # foreground
    B = np.zeros(img.shape)  # background
    alphaRes = np.zeros(trimap.shape)  # alpha result

    n = 1
    alpha[unknown_mask] = np.nan
    nUnknown = np.sum(unknown_mask)  # number of unknown pixels
    unkreg = unknown_mask

    kernel = np.ones((3, 3))  # 3*3 kernel
    while n < nUnknown:
        unkreg = cv2.erode(unkreg.astype(np.uint8), kernel, iterations=1)
        unkpixels = np.logical_and(np.logical_not(unkreg), unknown_mask)

        Y, X = np.nonzero(unkpixels)

        for i in range(Y.shape[0]):
            if n % 100 == 0:
                print(f"Processed {n} out of {nUnknown} unknown pixels")
            y, x = Y[i], X[i]
            p = img[y, x]
            # Try cluster Fg, Bg in p's known neighborhood

            # take surrounding alpha values
            a = get_window(alpha[:, :, np.newaxis], x, y, N)[:, :, 0]

            # Take surrounding foreground pixels
            f_pixels = get_window(foreground, x, y, N)
            f_weights = (a ** 2 * gaussian_weights).ravel()

            f_pixels = np.reshape(f_pixels, (N * N, 3))
            posInds = np.nan_to_num(f_weights) > 0
            f_pixels = f_pixels[posInds, :]
            f_weights = f_weights[posInds]

            # Take surrounding background pixels
            b_pixels = get_window(background, x, y, N)
            b_weights = ((1 - a) ** 2 * gaussian_weights).ravel()

            b_pixels = np.reshape(b_pixels, (N * N, 3))
            posInds = np.nan_to_num(b_weights) > 0
            b_pixels = b_pixels[posInds, :]
            b_weights = b_weights[posInds]

            # if not enough data, return to it later...
            if len(f_weights) < minN or len(b_weights) < minN:
                #print(f"Not enough data at pixel ({x}, {y}): f_weights={len(f_weights)}, b_weights={len(b_weights)}")
                # if end of loop has been reached and n is still < nUnknown, infinite loop will occur
                if i == Y.shape[0] and n < nUnknown:
                    print('i == Y.shape[0] and n < nUnknown')
                    # adjust minN, break loop, and retry. If that still fails, terminate the program
                    if minN > (minN - minN_reduction):
                        minN -= 1
                        n = 1
                        warnings.warn(message="Infinite loop encountered. Reducing minN by 1 and retrying.",
                                      category=RuntimeWarning)
                        break
                    else:
                        raise RuntimeError("Terminating infinite loop. Adjust input parameters and retry.")
                continue

            # Partition foreground and background pixels to clusters (in a weighted manner)
            mu_f, sigma_f = clustFunc(f_pixels, f_weights)
            mu_b, sigma_b = clustFunc(b_pixels, b_weights)

            alpha_init = np.nanmean(a.ravel())
            # Solve for F,B for all cluster pairs
            f, b, alphaT = solve(mu_f, sigma_f, mu_b, sigma_b, p, 0.01, alpha_init, 50, 1e-6)
            foreground[y, x] = f.ravel()
            background[y, x] = b.ravel()
            alpha[y, x] = alphaT
            unknown_mask[y, x] = 0
            n += 1

    print('Unknown pixels filled')
    return alpha

def main(img, trimap, mask, sigma, N, minN, minN_reduction):
    img = cv2.imread(str(Path(img)))[:, :, :3]
    trimap = cv2.imread(str(Path(trimap)), cv2.IMREAD_GRAYSCALE)
    alpha = bayesian_matte(img, trimap, sigma, N, minN, minN_reduction)
    print('Alpha matte generated')
    # save alpha matte
    cv2.imwrite('Project3/ans/Figure_1.bmp', alpha * 255)
    plt.title("Alpha matte")
    plt.imshow(alpha, cmap='gray')
    plt.show()   
    mask_img = cv2.imread(str(Path(mask)), cv2.IMREAD_GRAYSCALE)
    if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
        print("Alpha contains NaN or Inf values")
    if np.any(np.isnan(mask_img)) or np.any(np.isinf(mask_img)):
        print("Mask image contains NaN or Inf values")
    # calculate errors
    # SAD
    sad =  matte_sad(alpha,mask_img)
    print('SAD:', sad)
    # MAD
    mad = matte_mad(alpha,mask_img)
    print('MAD:', mad)
    # MSE
    mse = matte_mse(alpha,mask_img)
    print('MSE:', mse)
    # gradient error
    grad = matte_grad(alpha, mask_img)
    print('Gradient error:', grad)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # start parser
    parser = ArgumentParser()

    # add args
    parser.add_argument('image', help="path to image to be segmented")
    parser.add_argument('trimap', help="path to trimap of image")
    parser.add_argument('mask', help="path to given mask of image")
    parser.add_argument('-s', '--sigma', default=8, help="variance of gaussian for spatial weighting")
    parser.add_argument('-n', '--N', default=25, help="pixel neighborhood size")
    parser.add_argument('-mn', '--minN', default=10, help="minimum required foreground and background neighbors for optimization")
    parser.add_argument('-red', '--minN_reduction', default=0, help="number of times to reduce minN if an infinite loop is encountered")

    args = parser.parse_args()
    # call main with all args
    main(args.image, args.trimap, args.mask, args.sigma, args.N, args.minN, args.minN_reduction)