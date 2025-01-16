import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Gaussian filtering in frequency domain
# Inputs: img - the grayscale image
# D0 - filter parameter, measure of spread
def gaussian_low_high_pass_filter(img, d0):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    m, n = img.shape
    G = np.zeros((m, n))
    x0 = m // 2
    y0 = n // 2

    img_data = np.asarray(img)
    f = np.fft.fft2(img_data)
    f_shift = np.fft.fftshift(f)

    # Create the Gaussian low-pass filter
    for i in range(m):
        for j in range(n):
            dist = ((i - x0) ** 2 + (j - y0) ** 2) ** 0.5
            G[i, j] = np.exp(-(dist ** 2) / (2 * (d0 ** 2)))

    # Apply the Gaussian low-pass filter
    low_pass = f_shift * G

    # Generate high-pass filter by subtracting the low-pass filter from 1
    H = 1 - G

    # Apply the high-pass filter
    high_pass = f_shift * H

    # Apply the inverse FFT to get the images back to spatial domain
    low_pass_image = np.fft.ifftshift(low_pass)
    low_pass_image = np.fft.ifft2(low_pass_image)
    low_pass_image = np.abs(low_pass_image)
    low_pass_image = np.array(low_pass_image, dtype=np.uint8)

    high_pass_image = np.fft.ifftshift(high_pass)
    high_pass_image = np.fft.ifft2(high_pass_image)
    high_pass_image = np.abs(high_pass_image)
    high_pass_image = np.array(high_pass_image, dtype=np.uint8)

    return low_pass_image, high_pass_image, f_shift, low_pass, H, G


# Load a color image
img = cv2.imread("city_hall.jpg")
cv2.imshow("Original Image", img)

# Get the low-pass and high-pass filtered images and FFTs
low_pass_image, high_pass_image, f_shift, low_pass, H, G = gaussian_low_high_pass_filter(img, 30)
cv2.imshow("Low-Pass Filtered Image", low_pass_image)
cv2.imshow("High-Pass Filtered Image", high_pass_image)

# Apply Canny edge detection to the high-pass filtered image
edges = cv2.Canny(high_pass_image, 50, 150)

# Apply Hough Line Transformation
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Create a copy of the original image to draw lines on it
line_image = np.copy(img)

# Draw the lines on the image
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Edges Detected", edges)
cv2.imshow("Hough Transform Result", line_image)

# Plot figure for FFT and filters visualization
fig = plt.figure(figsize=(12, 12))

# Show FFT of the original image
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
ax1 = fig.add_subplot(311)
ax1.imshow(magnitude_spectrum, cmap='gray')
ax1.set_title('FFT of Original Image')
ax1.set_xticks([]), ax1.set_yticks([])

# Show FFT of the low-pass filtered image
magnitude_spectrum_lowpass = 20 * np.log(np.abs(low_pass) + 1)
ax2 = fig.add_subplot(312)
ax2.imshow(magnitude_spectrum_lowpass, cmap='gray')
ax2.set_title('FFT of Low-Pass Filtered Image')
ax2.set_xticks([]), ax2.set_yticks([])

# Show FFT of the high-pass filtered image
magnitude_spectrum_highpass = 20 * np.log(np.abs(H * f_shift) + 1)
ax3 = fig.add_subplot(313)
ax3.imshow(magnitude_spectrum_highpass, cmap='gray')
ax3.set_title('FFT of High-Pass Filtered Image')
ax3.set_xticks([]), ax3.set_yticks([])

plt.tight_layout()

# 3D Plot of the Gaussian Low-Pass Filter H(f)
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(211, projection='3d')
X, Y = np.meshgrid(np.arange(G.shape[1]), np.arange(G.shape[0]))
ax1.plot_surface(X, Y, G, cmap='viridis')
ax1.set_title('Gaussian Low-Pass Filter H(f)')

# 3D Plot of the Gaussian High-Pass Filter H(f)
ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_surface(X, Y, H, cmap='viridis')
ax2.set_title('Gaussian High-Pass Filter H(f)')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()