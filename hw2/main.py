import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def C(w):
    return 1 / np.sqrt(2) if w == 0 else 1

def dct2d(f):
    """Two-dimensional DCT"""
    N = f.shape[0]
    F = np.zeros((N, N))
    for u in range(N):
        # print(f"u= {u}")
        for v in range(N):
            # print(f"v= {v}")
            s = 0
            for x in range(N):
                for y in range(N):
                    s += f[x, y] * np.cos((2*x + 1) * u * np.pi / (2*N)) * np.cos((2*y + 1) * v * np.pi / (2*N))
            F[u, v] = (2 / N) * C(u) * C(v) * s
    return F

def idct2d(F):
    """Two-dimensional inverse DCT"""
    N = F.shape[0]
    f = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            s = 0
            for u in range(N):
                for v in range(N):
                    s += C(u) * C(v) * F[u, v] * np.cos((2*x + 1) * u * np.pi / (2*N)) * np.cos((2*y + 1) * v * np.pi / (2*N))
            f[x, y] = (2 / N) * s
    return f

import numpy as np

def C(w):
    return 1 / np.sqrt(2) if w == 0 else 1

def dct2(f):
    """Two-dimensional DCT"""
    N = f.shape[0]
    F = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            s = 0
            for x in range(N):
                for y in range(N):
                    s += f[x, y] * np.cos((2*x + 1) * u * np.pi / (2*N)) * np.cos((2*y + 1) * v * np.pi / (2*N))
            F[u, v] = (2 / N) * C(u) * C(v) * s
    return F

def idct2(F):
    """Two-dimensional inverse DCT"""
    N = F.shape[0]
    f = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            s = 0
            for u in range(N):
                for v in range(N):
                    s += C(u) * C(v) * F[u, v] * np.cos((2*x + 1) * u * np.pi / (2*N)) * np.cos((2*y + 1) * v * np.pi / (2*N))
            f[x, y] = (2 / N) * s
    return f


# ===== 1D DCT / IDCT ==================================================
def dct1d(vector):
    """Perform 1D Discrete Cosine Transform (DCT-II)"""
    N = len(vector)
    result = np.zeros(N)
    for k in range(N):
        alpha = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
        for n in range(N):
            result[k] += vector[n] * math.cos(math.pi * (2*n + 1) * k / (2*N))
        result[k] *= alpha
    return result


def idct1d(vector):
    """Perform 1D Inverse Discrete Cosine Transform (DCT-III)"""
    N = len(vector)
    result = np.zeros(N)
    for n in range(N):
        for k in range(N):
            alpha = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
            result[n] += alpha * vector[k] * math.cos(math.pi * (2*n + 1) * k / (2*N))
    return result


# ===== Two 1D-DCT =====================================================
def two_1d_dct(image):
    """Compute 2D DCT using two 1D-DCTs (row-wise then column-wise)"""
    # Step 1: apply 1D-DCT to each row
    temp = np.apply_along_axis(dct1d, axis=1, arr=image)
    # Step 2: apply 1D-DCT to each column
    result = np.apply_along_axis(dct1d, axis=0, arr=temp)
    return result


def two_1d_idct(coeff):
    """Compute 2D IDCT using two 1D-IDCTs (column-wise then row-wise)"""
    # Step 1: apply 1D-IDCT to each column
    temp = np.apply_along_axis(idct1d, axis=0, arr=coeff)
    # Step 2: apply 1D-IDCT to each row
    result = np.apply_along_axis(idct1d, axis=1, arr=temp)
    return result



def main():
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float64)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    cv2.imwrite("original_resize.png", np.clip(img, 0, 255))
    # --- 2D-DCT ---
    start = time.time()
    print(start)
    dct_coeff = dct2d(img)
    end = time.time()
    print(end)
    time_2d = end - start
    recon_2d = idct2d(dct_coeff)
    psnr_2d = psnr(img, recon_2d)

    # Visualization (log domain)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log(np.abs(dct_coeff) + 1), cmap='gray')
    plt.title("2D-DCT Coefficients (log scale)")
    plt.axis('off')
    plt.savefig("dct_coeff_log_2d.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # --- Two 1D-DCT ---
    start = time.time()
    dct_coeff_1d = two_1d_dct(img)
    end = time.time()
    time_1d = end - start
    recon_1d = two_1d_idct(dct_coeff_1d)
    psnr_1d = psnr(img, recon_1d)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.log(np.abs(dct_coeff_1d) + 1), cmap='gray')
    plt.title("1D-DCT Coefficients (log scale)")
    plt.axis('off')
    plt.savefig("dct_coeff_log_1d.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # --- Report results ---
    print("===== Results =====")
    print(f"2D-DCT time: {time_2d:.6f} sec, PSNR: {psnr_2d:.2f} dB")
    print(f"Two 1D-DCT time: {time_1d:.6f} sec, PSNR: {psnr_1d:.2f} dB")

    cv2.imwrite("reconstructed_2D.png", np.clip(recon_2d, 0, 255))
    cv2.imwrite("reconstructed_1D.png", np.clip(recon_1d, 0, 255))

if __name__ == "__main__":
    main()
