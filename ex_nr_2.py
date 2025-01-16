import cv2
import numpy as np

# Gaussian filtering in frequency domain
# Inputs: img - the grayscale image
# D0 - filter parameter, measure of spread
def gaussian_fourier_filter(img, d0):
    # Uncomment the line below if your computer is slow
    # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    m, n = img.shape
    G = np.zeros((m, n))
    x0 = m // 2
    y0 = n // 2

    img_data = np.asarray(img)
    f = np.fft.fft2(img_data)
    f_shift = np.fft.fftshift(f)

    # Create the filter
    for i in range(-(m // 2), (m // 2)):
        for j in range(-(n // 2), (n // 2)):
            x = i + x0
            y = j + y0
            dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
            G[x, y] = np.exp(-(dist ** 2) / (2 * (d0 ** 2)))

    filtered_image = f_shift * G
    filtered_image = np.fft.ifftshift(filtered_image)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image)
    filtered_image = np.array(filtered_image, dtype=np.uint8)
    return filtered_image


# Load a color image in grayscale
img = cv2.imread("city_hall.jpg")
cv2.imshow("Window", img)

img_filtered = gaussian_fourier_filter(img, 30)
cv2.imshow("Window2", img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
