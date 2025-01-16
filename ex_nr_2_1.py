import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to perform Gaussian low-pass and high-pass filtering in the frequency domain
# Inputs:
# - img: The input image (grayscale or color)
# - d0: The cutoff frequency for the Gaussian filter
def gaussian_low_high_pass_filter(img, d0):
    # Convert color images to grayscale
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get dimensions of the image
    m, n = img.shape

    # Initialize Gaussian filter array
    G = np.zeros((m, n))
    x0 = m // 2  # Center x-coordinate
    y0 = n // 2  # Center y-coordinate

    # Convert image to NumPy array and compute its FFT
    img_data = np.asarray(img)
    f = np.fft.fft2(img_data)
    f_shift = np.fft.fftshift(f)  # Shift the zero-frequency to the center

    # Create Gaussian low-pass filter in the frequency domain
    for i in range(m):
        for j in range(n):
            dist = ((i - x0) ** 2 + (j - y0) ** 2) ** 0.5  # Euclidean distance
            G[i, j] = np.exp(-(dist ** 2) / (2 * (d0 ** 2)))  # Gaussian formula

    # Apply the Gaussian low-pass filter
    low_pass = f_shift * G

    # Generate Gaussian high-pass filter by subtracting low-pass from 1
    H = 1 - G
    high_pass = f_shift * H  # Apply high-pass filter

    # Transform filtered frequency domains back to spatial domain
    low_pass_image = np.fft.ifftshift(low_pass)
    low_pass_image = np.fft.ifft2(low_pass_image)
    low_pass_image = np.abs(low_pass_image)
    low_pass_image = np.array(low_pass_image, dtype=np.uint8)

    high_pass_image = np.fft.ifftshift(high_pass)
    high_pass_image = np.fft.ifft2(high_pass_image)
    high_pass_image = np.abs(high_pass_image)
    high_pass_image = np.array(high_pass_image, dtype=np.uint8)

    # Return both low-pass and high-pass filtered images, along with filters and FFT
    return low_pass_image, high_pass_image, f_shift, low_pass, H, G


# Load the input image
# img = cv2.imread("city_hall.jpg")
img = cv2.imread("portrait_dog.jpg")
cv2.imshow("Original Image", img)

# Apply Gaussian low-pass and high-pass filtering
low_pass_image, high_pass_image, f_shift, low_pass, H, G = gaussian_low_high_pass_filter(img, 30)

# Display low-pass and high-pass filtered images
cv2.imshow("Low-Pass Filtered Image", low_pass_image)
cv2.imshow("High-Pass Filtered Image", high_pass_image)

# Perform edge detection on the high-pass filtered image
edges = cv2.Canny(high_pass_image, 50, 150)

# Apply Hough Line Transformation to detect straight lines in the edges
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Create a copy of the original image to draw detected lines
line_image = np.copy(img)

# Draw lines detected by Hough Transform
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display edges and the image with detected lines
cv2.imshow("Edges Detected", edges)
cv2.imshow("Hough Transform Result", line_image)

# Plot FFT and filters visualization
fig = plt.figure(figsize=(12, 12))

# FFT magnitude spectrum of the original image
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
ax1 = fig.add_subplot(311)
ax1.imshow(magnitude_spectrum, cmap='gray')
ax1.set_title('FFT of Original Image')
ax1.set_xticks([]), ax1.set_yticks([])

# FFT of the low-pass filtered image
magnitude_spectrum_lowpass = 20 * np.log(np.abs(low_pass) + 1)
ax2 = fig.add_subplot(312)
ax2.imshow(magnitude_spectrum_lowpass, cmap='gray')
ax2.set_title('FFT of Low-Pass Filtered Image')
ax2.set_xticks([]), ax2.set_yticks([])

# FFT of the high-pass filtered image
magnitude_spectrum_highpass = 20 * np.log(np.abs(H * f_shift) + 1)
ax3 = fig.add_subplot(313)
ax3.imshow(magnitude_spectrum_highpass, cmap='gray')
ax3.set_title('FFT of High-Pass Filtered Image')
ax3.set_xticks([]), ax3.set_yticks([])

plt.tight_layout()

# 3D visualization of Gaussian filters
fig = plt.figure(figsize=(12, 12))

# 3D plot of Gaussian low-pass filter
ax1 = fig.add_subplot(211, projection='3d')
X, Y = np.meshgrid(np.arange(G.shape[1]), np.arange(G.shape[0]))
ax1.plot_surface(X, Y, G, cmap='viridis')
ax1.set_title('Gaussian Low-Pass Filter H(f)')

# 3D plot of Gaussian high-pass filter
ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_surface(X, Y, H, cmap='viridis')
ax2.set_title('Gaussian High-Pass Filter H(f)')

plt.tight_layout()
plt.show()

# Close all OpenCV windows after key press
cv2.waitKey(0)
cv2.destroyAllWindows()