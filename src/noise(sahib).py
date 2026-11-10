import numpy as np 
import cv2 
import random as rd 

def brightness_contrast(img, rng, alpha_range=(0.6, 1.4), beta_range=(-40, 40)):
    alpha = rng.uniform(*alpha_range)
    beta = rng.uniform(*beta_range)

    img_f = img.astype(np.float32)
    out = alpha * img_f + beta

    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, rng, length_range=(5, 25), angle_range=(0, 180)):
    length = int(rng.integers(length_range[0], length_range[1] + 1))
    angle = rng.uniform(*angle_range)
    k=length
    if k%2 == 0:
        k=k+1
    kernel = np.zeros((k, k), dtype=np.float32)

    kernel[k // 2, :] = 1.0

    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k), flags=cv2.INTER_LINEAR)

    s = kernel.sum()
    if s == 0:
        return img.copy()
    kernel /= s

    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)