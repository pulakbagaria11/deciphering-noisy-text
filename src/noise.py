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

# ---------------------------------------------------------------------------
# Helpers (used by perspective_jitter and random_lines)
# ---------------------------------------------------------------------------

def _ink_color(img, rng, gray_range=(0, 90)):
    """Return a random dark gray, similar to the text, so drawn lines look like ink."""
    g = int(rng.integers(gray_range[0], gray_range[1] + 1))   # random gray level between 0 (black) and 90 (dark gray)
    return g if img.ndim == 2 else (g,) * img.shape[2]         # grayscale image -> one number; color image -> same value per channel


def _bg_color(img):
    """Return the background color, estimated as the median pixel value."""
    m = int(np.median(img))                                    # most pixels are background, so the median is a background pixel
    return m if img.ndim == 2 else (m,) * img.shape[2]         # grayscale -> one number; color -> same value per channel


# ---------------------------------------------------------------------------
# perspective_jitter
# ---------------------------------------------------------------------------

def perspective_jitter(img, rng, max_shift=0.15):
    """Simulates a photo of the text taken slightly off-angle.

    Each of the 4 image corners is moved randomly by up to max_shift * width
    (horizontally) and max_shift * height (vertically). A perspective transform
    (homography) maps the original corners onto the moved ones. The output canvas
    is sized to fit the warped shape exactly, so no text is cut off, and is then
    resized back to the input's shape.
    """
    h, w = img.shape[:2]                                       # image height and width in pixels

    # the 4 corners of the input image: top-left, top-right, bottom-right, bottom-left (x, y)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # random shift for each corner: x in [-max_shift*w, +max_shift*w], y in [-max_shift*h, +max_shift*h]
    shift = rng.uniform(-max_shift, max_shift, size=(4, 2)) * np.array([w, h])
    dst = src + shift                                          # where each corner ends up after the shift

    # move the shifted shape so its top-left-most point is at (0, 0)
    dst -= dst.min(axis=0)                                     # dst.min(axis=0) = smallest x and smallest y among the 4 corners

    # canvas just large enough to hold the warped shape, so nothing is pushed out of frame
    out_w = int(np.ceil(dst[:, 0].max())) + 1                  # width  = largest x after the move
    out_h = int(np.ceil(dst[:, 1].max())) + 1                  # height = largest y after the move

    # 3x3 homography matrix that sends each src corner to its dst corner
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    # apply the homography to every pixel; uncovered areas are filled with the background color
    warped = cv2.warpPerspective(img, M, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR,       # blend the 4 nearest pixels for smooth edges
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=_bg_color(img))

    # resize back to the original (h, w) so the next noise step gets the expected shape
    return cv2.resize(warped, (w, h), interpolation=cv2.INTER_AREA)   # INTER_AREA averages pixels, best for shrinking


# ---------------------------------------------------------------------------
# random_lines
# ---------------------------------------------------------------------------

def random_lines(img, rng, count=2, thickness=1):
    """Draws straight dark lines across the text, like strike-throughs or ruled lines.

    count     = number of lines to draw
    thickness = maximum line thickness in pixels (each line gets a random thickness from 1 to this)
    """
    out = img.copy()                                           # copy so the input image is not modified (cv2 draws in place)
    h, w = out.shape[:2]                                       # image height and width in pixels

    for _ in range(int(count)):                                # draw `count` lines
        p0 = (0, int(rng.integers(0, h)))                      # start point: left edge, random height
        p1 = (w - 1, int(rng.integers(0, h)))                  # end point: right edge, random height
        t = int(rng.integers(1, int(thickness) + 1))           # random thickness between 1 and `thickness`
        cv2.line(out, p0, p1, _ink_color(out, rng), t,         # draw the line in a random dark gray
                 lineType=cv2.LINE_AA)                         # anti-aliased: smooth edges like real ink

    return out                                                 # same shape and dtype as the input
