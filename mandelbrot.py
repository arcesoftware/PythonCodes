# mandelbrot_fast.py
import numpy as np
import matplotlib.pyplot as plt
import colorsys

# --- Configuration ---
VIEW_AREA = [-2.0, 1.0, -1.5, 1.5]  # xmin, xmax, ymin, ymax
WIDTH, HEIGHT = 800, 600
MAX_ITERATIONS = 256
COLOR_POWER = 0.5
INSIDE_COLOR = (0.0, 0.0, 0.0)  # black

# --- Palette creation (0..MAX_ITERATIONS-1) ---
def create_palette(max_iter):
    palette = []
    for i in range(max_iter):
        if i == max_iter - 1:
            # inside color
            r, g, b = INSIDE_COLOR
        else:
            # smooth mapping using fractional position and power control
            v = (i / float(max_iter)) ** COLOR_POWER
            # multiply hue factor to stretch colors around the wheel
            h = (v * 4.0) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(palette, dtype=np.uint8)

PALETTE = create_palette(MAX_ITERATIONS)

# --- Mandelbrot calculation (vectorized, correct escape-time logic) ---
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymax, ymin, height)  # top row -> imag = ymax
    c = real[np.newaxis, :] + 1j * imag[:, np.newaxis]

    z = np.zeros_like(c, dtype=np.complex128)
    # initialize all values to max_iter (meaning "did not escape")
    m = np.full(c.shape, max_iter, dtype=np.int32)

    # mask of points still being iterated (not yet escaped)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        # iterate z for points that haven't escaped
        z[mask] = z[mask] * z[mask] + c[mask]

        # find which points have just escaped
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & mask

        # set escape iteration for newly escaped points
        m[newly_escaped] = i

        # remove newly escaped from mask
        mask &= ~newly_escaped

        # if nothing left to iterate, break early
        if not mask.any():
            break

    # points that never escaped keep max_iter; map them to final palette index
    m[m == max_iter] = max_iter - 1
    return m

# --- Build RGB image from escape map ---
def build_image(escape_map, palette):
    # escape_map shape (H,W); palette shape (max_iter,3)
    img = palette[escape_map]  # vectorized indexing -> shape (H,W,3)
    return img

# --- Main plotting function ---
def plot_mandelbrot():
    xmin, xmax, ymin, ymax = VIEW_AREA
    escape_map = mandelbrot_numpy(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITERATIONS)
    img = build_image(escape_map, PALETTE)

    plt.figure(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
    plt.imshow(img, origin="upper", extent=[xmin, xmax, ymin, ymax])
    plt.axis("off")
    plt.title(f"Mandelbrot (size={WIDTH}x{HEIGHT}, iters={MAX_ITERATIONS})")
    plt.show()

if __name__ == "__main__":
    plot_mandelbrot()
