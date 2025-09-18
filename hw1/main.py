import numpy as np
from PIL import Image

img = Image.open("lena.png").convert("RGB")
img_np = np.array(img).astype(np.float32)  # shape: (H, W, 3)

R = img_np[:, :, 0]
G = img_np[:, :, 1]
B = img_np[:, :, 2]

# ---------- RGB -> YUV ----------
Y = 0.299 * R + 0.587 * G + 0.114 * B
# homework requrirement function
# U = -0.169 * R - 0.331 * G + 0.5 * B + 128
# V = 0.5 * R - 0.419 * G - 0.081 * B + 128

# slide function
# U = -0.147 * R - 0.287 * G + 0.436 * B
# V = 0.615 * R - 0.515 * G - 0.100 * B

# slide function with offset
U = -0.147 * R - 0.287 * G + 0.436 * B + 128
V = 0.615 * R - 0.515 * G - 0.100 * B + 128

# ---------- RGB -> YCbCr ----------
Cb = - 0.48 * R - 0.291 * G + 0.439 * B + 128
Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128

def save_gray(array, filename):
    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(filename)

save_gray(R, "R_channel.png")
save_gray(G, "G_channel.png")
save_gray(B, "B_channel.png")
save_gray(Y, "Y_channel.png")
save_gray(U, "U_channel.png")
save_gray(V, "V_channel.png")
save_gray(Cb, "Cb_channel.png")
save_gray(Cr, "Cr_channel.png")

