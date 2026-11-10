import cv2
import numpy as np


def gaussian_noise(img, rng, sigma, mean=0.0):
    img_f = img.astype(np.float32)
    noise = rng.normal(loc=mean, scale=sigma, size=img.shape).astype(np.float32)
    return np.clip(img_f + noise, 0, 255).astype(np.uint8)


def speckle_noise(img, rng, sigma):
    img_f = img.astype(np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    return np.clip(img_f + img_f * noise, 0, 255).astype(np.uint8)


def salt_and_pepper(img, rng, amount, salt_ratio=0.5):
    out = img.copy()
    u = rng.random(img.shape)
    salt_mask = u < amount * salt_ratio
    pepper_mask = (u >= amount * salt_ratio) & (u < amount)
    out[salt_mask] = 255
    out[pepper_mask] = 0
    return out


def rotate_skew(img, rng,
                theta_range=(-3, 3), scale_range=(0.95, 1.05),
                shear_x_range=(-0.3, 0.3), shear_y_range=(-0.03, 0.03)):
    theta = rng.uniform(*theta_range)
    scale = rng.uniform(*scale_range)
    shear_x = rng.uniform(*shear_x_range)
    shear_y = rng.uniform(*shear_y_range)

    h, w = img.shape[:2]
    rotation = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), theta, scale), [0, 0, 1]])
    shear = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float64)
    to_origin = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    back = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])

    M = rotation @ (back @ shear @ to_origin)
    return cv2.warpAffine(img, M[:2, :], (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=int(np.median(img)))   # grayscale images


def elastic_warp(img, rng, sigma_range=(3, 6), alpha_range=(1.0, 3.0), dy_scale=0.5):
    """alpha = typical displacement in pixels; sigma = smoothness of the distortion."""
    h, w = img.shape[:2]
    sigma = rng.uniform(*sigma_range)
    alpha = rng.uniform(*alpha_range)

    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=sigma)
    dx = dx / (dx.std() + 1e-8) * alpha
    dy = dy / (dy.std() + 1e-8) * alpha * dy_scale

    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, x + dx, y + dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)



def jpeg_compression(img, rng, quality_range=(30, 95)):
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(buf, flag)


def gaussian_blur(img, rng=None, ksize=5, sigma=0):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8     

    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()

    src = img.astype(np.float32)
    h, w = src.shape[:2]
    pad = [(r, r), (r, r)] + [(0, 0)] * (src.ndim - 2)
    p = np.pad(src, pad, mode="reflect")

    tmp = sum(k[i] * p[:, i:i + w] for i in range(ksize))     
    out = sum(k[i] * tmp[i:i + h] for i in range(ksize))     

    if img.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out.astype(img.dtype)



def _rand_color(img, rng):
    if img.ndim == 2:
        return int(rng.integers(0, 256))
    return [int(v) for v in rng.integers(0, 256, size=img.shape[2])]


def _fill_ellipse(img, cx, cy, rx, ry, color):
    h, w = img.shape[:2]
    rx, ry = max(int(rx), 1), max(int(ry), 1)
    x0, x1 = max(cx - rx, 0), min(cx + rx + 1, w)
    y0, y1 = max(cy - ry, 0), min(cy + ry + 1, h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    img[y0:y1, x0:x1][mask] = color


def _draw_line(img, x0, y0, x1, y1, color, thickness=1):
    h, w = img.shape[:2]
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    for ox in range(thickness):
        for oy in range(thickness):
            px, py = xs + ox, ys + oy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            img[py[ok], px[ok]] = color


def _draw_rect(img, x0, y0, x1, y1, color, filled):
    h, w = img.shape[:2]
    x0, x1 = sorted((max(min(x0, w - 1), 0), max(min(x1, w - 1), 0)))
    y0, y1 = sorted((max(min(y0, h - 1), 0), max(min(y1, h - 1), 0)))
    if filled:
        img[y0:y1 + 1, x0:x1 + 1] = color
    else:
        img[y0, x0:x1 + 1] = color
        img[y1, x0:x1 + 1] = color
        img[y0:y1 + 1, x0] = color
        img[y0:y1 + 1, x1] = color


def dots_or_clutter(img, rng, mode="dots", amount=200, max_size=4, color=None):
    if mode not in ("dots", "clutter"):
        raise ValueError("mode must be 'dots' or 'clutter'")

    out = img.copy()
    h, w = out.shape[:2]

    for _ in range(int(amount)):
        c = color if color is not None else _rand_color(out, rng)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))

        if mode == "dots":
            r = int(rng.integers(1, max(1, int(max_size)) + 1))
            _fill_ellipse(out, x, y, r, r, c)
        else:
            span = max(2, int(max_size) * 6)
            dx = int(rng.integers(-span, span + 1))
            dy = int(rng.integers(-span, span + 1))
            shape = int(rng.integers(0, 3))
            if shape == 0:
                _draw_line(out, x, y, x + dx, y + dy, c, int(rng.integers(1, 3)))
            elif shape == 1:
                _draw_rect(out, x, y, x + dx, y + dy, c, bool(rng.integers(0, 2)))
            else:
                _fill_ellipse(out, x, y, abs(dx) // 2, abs(dy) // 2, c)
    return out



def _upscale_bilinear(small, h, w):
    sh, sw = small.shape
    ys = np.linspace(0, sh - 1, h)
    xs = np.linspace(0, sw - 1, w)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, sh - 1), np.minimum(x0 + 1, sw - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = small[y0][:, x0] * (1 - fx) + small[y0][:, x1] * fx
    bot = small[y1][:, x0] * (1 - fx) + small[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def background_texture(shape, rng, base=200, strength=25):
    h, w = shape[:2]
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)
    small = rng.normal(0, 1, (max(2, h // 32), max(2, w // 32))).astype(np.float32)
    blotch = _upscale_bilinear(small, h, w)

    tex = base + strength * (0.6 * blotch + 0.4 * gaussian_blur(grain, ksize=3))
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    if len(shape) == 3:
        tex = np.repeat(tex[:, :, None], shape[2], axis=2)
    return tex


def brightness_contrast(img, rng, alpha_range=(0.6, 1.4), beta_range=(-40, 40)):

    alpha = rng.uniform(*alpha_range)
    beta = rng.uniform(*beta_range)

    img_f = img.astype(np.float32)
    mean = img_f.mean()
    out = alpha * (img_f - mean) + mean + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, rng, length_range=(3, 11), angle_range=(-20, 20)):

    length = int(rng.integers(length_range[0], length_range[1] + 1))
    angle = rng.uniform(*angle_range)

    k = length if length % 2 == 1 else length + 1      
    if k < 3:
        return img.copy()                              

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0                            

    center = (k - 1) / 2.0                             
    rot = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k), flags=cv2.INTER_LINEAR)

    s = kernel.sum()
    if s == 0:
        return img.copy()
    kernel /= s

    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

def rand_edge_point(rng, W, H, edge):
    """A random (x, y) point on one edge: 0 = left, 1 = right, 2 = top, 3 = bottom."""
    if edge == 0:
        return [0, rng.uniform(0, H)]
    if edge == 1:
        return [W - 1, rng.uniform(0, H)]
    if edge == 2:
        return [rng.uniform(0, W), 0]
    return [rng.uniform(0, W), H - 1]
 
 

def curved_lines(img, rng, count=3, thickness=2):
    """
    Draws "count" wavy lines across the image, each "thickness" pixels wide.

    Each image start from the left end and goes to right

    Returns a new array, the input img is not modified.
    """
    out = img.copy()
    H, W = out.shape[:2]
    channels = 1 if out.ndim == 2 else out.shape[2]
    t = np.linspace(0, 1, 300)[:, None]          # 300 points along each curve
    # 300 points are able to give the smoothness to the line

    for _ in range(count):

        e1, e2 = rng.choice(4, size=2, replace=False) # i randonmly select two edge so that we get start
        # and end on different edges
        start = rand_edge_point(rng, W, H, e1)
        end = rand_edge_point(rng, W, H, e2)
        
        
        # Two random points in the middle decide how the line bends
        mid1 = [rng.uniform(0, W), rng.uniform(0, H)]
        mid2 = [rng.uniform(0, W), rng.uniform(0, H)]
        P = np.array([start, mid1, mid2, end])

        # using this equation to find the 300 coordinates for the line, as t changes the the effect
        # of each of four selected points change
        B = ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] \
            + 3 * (1 - t) * (t ** 2) * P[2] + (t ** 3) * P[3]
        pts = np.round(B).astype(np.int32).reshape(-1, 1, 2)

        color = _rand_color(img, rng)   
        cv2.polylines(out, [pts], isClosed=False, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)

    return out


def occlusion_blobs(img, rng, count=3, max_size=8):
    """Covers the image with "count" irregular blobs, each at most "max_size" pixels across.
    we use a circle to create a blob, so maximum allowed radius would max_size/2

    Returns a new array; the input is not modified.
    """
    out = img.copy()
    H, W = out.shape[:2]
    channels = 1 if out.ndim == 2 else out.shape[2]
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)

    for n in range(count):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)       # blob centre
        r = rng.uniform(0.2, 1.0) * max_size / 2            # base radius

        # for every angle we add the values from three sin curves, each with different amounts of maxima and minima
        # so that we get different amounts of radius, for every angle
        wobble = sum(rng.uniform(0, 0.2) * np.sin(k * angles + rng.uniform(0, 2 * np.pi)) for k in (1, 2, 3))
        radius = np.minimum(r * (1 + wobble), max_size / 2)  # never exceed max_size

        pts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
        # converts polar form into cartesian form 
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        # We draw blob on another copy, so that we can implement the opacity thing
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color=_rand_color(img, rng), lineType=cv2.LINE_AA)
        alpha = rng.uniform(0.6, 1.0)                        # mostly opaque
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha)
        # basically adding two images on top of each other pixel by pixel to get
        # the opacity effect

    return out
