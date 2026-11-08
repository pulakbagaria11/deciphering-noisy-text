import numpy as np 
import cv2 
import random as rd 

def gaussian_noise(img, sigma, rng, mean=0.0):
    img_f = img.astype(np.float32)
    noise = rng.normal(loc = mean, scale = sigma, size=img.shape).astype(np.float32)
    noisy = img_f+noise 
    return np.clip(noisy, 0, 255).astype(np.uint8)



def salt_and_pepper(img, amt, salt_ratio, rng):
    out = img.copy()
    u = rng.random(img.shape)
    
    salt_mask = u <amt*salt_ratio
    pepper_mask = (u>= amt*salt_ratio) & (u<amt)
    out[salt_mask] = 255
    out[pepper_mask] = 0 
    return out 

def speckle_noise(img, sigma, rng, mean=0.0):
    img_f = img.astype(np.float32)
    noise  = rng.normal(loc=mean, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img_f + img_f*noise 
    return np.clip(noisy, 0, 255).astype(np.uint8)

def rotate_skew(img, rng, theta_range=(-15, 15), scale_range=(0.9, 1.1),shear_x_range=(-0.15, 0.15), shear_y_range=(-0.15, 0.15)):
    theta=rng.uniform(*theta_range)
    scale=rng.uniform(*scale_range)
    shear_x=rng.uniform(*shear_x_range)
    shear_y=rng.uniform(*shear_y_range)
    height,width=img.shape[:2]
    rotation_matrix=cv2.getRotationMatrix2D((width/2, height/2), theta, scale)
    rotation_matrix=np.vstack([rotation_matrix, [0, 0, 1]])
    shear_matrix=np.array([[1, shear_x, 0],[shear_y, 1, 0],[0, 0, 1]]).astype(np.float64)
    translate_to_origin=np.array([[1, 0, -(width/2)],[0, 1, -(height/2)],[0, 0, 1]])
    translate_back=np.array([[1, 0, (width/2)],[0, 1, height/2],[0, 0, 1]])
    centred_shear=translate_back@shear_matrix@translate_to_origin
    transformation_matrix=rotation_matrix@centred_shear
    return cv2.warpAffine(img,transformation_matrix[:2, :],(width, height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def jpeg_compression(img, rng, quality_range=(30, 95)):
    quality=rng.integers(quality_range[0], quality_range[1]+1)
    _,lowered_quality=cv2.imencode(".jpg",img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    output=cv2.imdecode(lowered_quality, cv2.IMREAD_UNCHANGED)
    return output

def elastic_warp(img,rng,sigma_range=(3,6),alpha_range=(20,80)):
    height, width = img.shape[:2]
    dx=rng.uniform(-1, 1, (height, width)).astype(np.float32)
    dy=rng.uniform(-1, 1, (height, width)).astype(np.float32)
    sigma=rng.uniform(*sigma_range)
    alpha=rng.uniform(*alpha_range)
    dx=cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma)
    dy=cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma)
    dx*=alpha
    dy*=alpha
    x, y=np.meshgrid(np.arange(width), np.arange(height))
    map_x=(x+dx).astype(np.float32)
    map_y=(y+dy).astype(np.float32)
    output=cv2.remap(img, map_x, map_y,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return output

def gaussian_blur(img, ksize=5, sigma=0):
    ksize = max(1, int(ksize))      
    if ksize % 2 == 0:
            ksize += 1		
	if sigma < 0:
    		raise ValueError("sigma must be >= 0")
	if sigma == 0:
    		sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float32) 
    k = np.exp(-(x * 2) / (2 * sigma * 2))
    k /= k.sum()


    src = img.astype(np.float32)
    h, w = src.shape[:2]
    pad = [(r, r), (r, r)] + [(0, 0)] * (src.ndim - 2)		
    p = np.pad(src, pad, mode=“reflect”) 	

    tmp = sum(k[i] * p[:, i:i + w] for i in range(ksize))   
   
    out = sum(k[i] * tmp[i:i + h] for i in range(ksize))           
    if np.issubdtype(img.dtype, np.integer):
        out = np.rint(out)				
    return np.clip(out, 0, 255).astype(img.dtype) if img.dtype == np.uint8 else out.astype(img.dtype)	





def _rand_color(img):
    if img.ndim == 2:
        return random.randint(0, 255)
    return [random.randint(0, 255) for _ in range(img.shape[2])]


def _fill_circle(img, cx, cy, rx, ry, color):
    h, w = img.shape[:2]
    x0, x1 = max(cx - rx, 0), min(cx + rx + 1, w) 
    y0, y1 = max(cy - ry, 0), min(cy + ry + 1, h)
    if x0 >= x1 or y0 >= y1:   
        return
    yy, xx = np.ogrid[y0:y1, x0:x1] 		
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


def dots_or_clutter(img, mode="dots", amount=200, max_size=4, color=None, seed=None):
    if seed is not None:
        random.seed(seed)
    out = img.copy()
    h, w = out.shape[:2]

    for _ in range(int(amount)):
        c = color if color is not None else _rand_color(out)
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)

        if mode == "dots":
            r = random.randint(1, max(1, int(max_size)))
            _fill_circle(out, x, y, r, r, c)
        elif mode == "clutter":
            span = max(2, int(max_size) * 6) 
            dx, dy = random.randint(-span, span), random.randint(-span, span)	
            shape = random.choice(["line", "rect", "ellipse"])
            if shape == "line":
                _draw_line(out, x, y, x + dx, y + dy, c, random.randint(1, 2))
            elif shape == "rect":
                _draw_rect(out, x, y, x + dx, y + dy, c, random.choice([True, False]))
            else:
                _fill_circle(out, x, y, max(1, abs(dx) // 2), max(1, abs(dy) // 2), c)
        else:
            raise ValueError("mode must be 'dots' or 'clutter'")
    return out


def _upscale_bilinear(small, h, w):
    sh, sw = small.shape
    ys = np.linspace(0, sh - 1, h) 	
    xs = np.linspace(0, sw - 1, w)	
    y0, x0 = np.floor(ys).astype(int), np.floor(XS).astype(int)           
    y1, x1 = np.minimum(y0 + 1, sh - 1), np.minimum(x0 + 1, sw - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :] 	
    top = small[y0][:, x0] * (1 - fx) + small[y0][:, x1] * fx   
    bot = small[y1][:, x0] * (1 - fx) + small[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def background_texture(shape, base=200, strength=25, seed=None):
    if seed is not None:
        random.seed(seed)
    h, w = shape[:2]
    rng = np.random.default_rng(seed)
    tex = np.full((h, w), float(base), dtype=np.float32)
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)

    small = rng.normal(0, 1, (max(2, h // 32), max(2, w // 32))).astype(np.float32)		
    blotch = _upscale_bilinear(small, h, w)

    tex += strength * (0.6 * blotch + 0.4 * gaussian_blur(grain, 3))

  

    tex = np.clip(tex, 0, 255).astype(np.uint8)
    if len(shape) == 3:
        tex = np.repeat(tex[:, :, None], shape[2], axis=2) 	
    return tex

