import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the input image in color mode (3 corresponds to cv2.IMREAD_COLOR)
img = cv2.imread('city_hall.jpg', 3)

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image into a NumPy array for further processing
img_data = np.asarray(gray_image)

# Perform a 2D Fast Fourier Transform on the image data
f = np.fft.fft2(img_data)

# Shift the zero-frequency component to the center of the spectrum
f = np.fft.fftshift(f)

# Compute the magnitude spectrum (absolute values of the Fourier transform)
f = abs(f)

# Apply logarithmic scaling for better visualization of the Fourier spectrum
fourier = np.log10(f)

# Find the lowest non-NaN finite value in the Fourier spectrum
lowest = np.nanmin(fourier[np.isfinite(fourier)])

# Find the highest non-NaN finite value in the Fourier spectrum
highest = np.nanmax(fourier[np.isfinite(fourier)])

# Calculate the range of contrast for normalization
contrast_range = highest - lowest

# Normalize the Fourier spectrum to the range [0, 255] for visualization
norm_fourier = (fourier - lowest) / contrast_range * 255

# Plot the original grayscale image and the normalized Fourier spectrum
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)  # Create the first subplot
ax2 = fig.add_subplot(1, 2, 2)  # Create the second subplot
ax1.imshow(gray_image, cmap="gray")  # Display the grayscale image
ax2.imshow(norm_fourier)  # Display the normalized Fourier spectrum
ax1.title.set_text("Original image")  # Set the title for the first subplot
ax2.title.set_text("Fourier image")  # Set the title for the second subplot
plt.show()  # Render the plots