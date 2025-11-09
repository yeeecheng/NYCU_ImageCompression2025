import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def main():
    # 1. 讀入灰階影像
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    
    # 2. 使用 OpenCV 進行 DCT
    start = time.time()
    dct_img = cv2.dct(img)
    end = time.time()
    time_dct = end - start

    # 3. 在 log domain 可視化係數
    plt.imshow(np.log(np.abs(dct_img) + 1), cmap='gray')
    plt.title("DCT Coefficients (Log Scale)")
    plt.axis('off')
    plt.savefig("dct_coeff_log_2d_opencv.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 4. 使用 OpenCV 進行 IDCT
    start = time.time()
    recon_img = cv2.idct(dct_img)
    end = time.time()
    time_idct = end - start

    # 5. 計算 PSNR
    psnr_val = psnr(img, recon_img)

    print("===== OpenCV DCT/IDCT =====")
    print(f"DCT time: {time_dct:.6f} sec")
    print(f"IDCT time: {time_idct:.6f} sec")
    print(f"PSNR: {psnr_val:.2f} dB")

    # 6. 儲存重建影像
    cv2.imwrite("reconstructed_opencv.png", np.clip(recon_img, 0, 255))
    
if __name__ == "__main__":
    main()
