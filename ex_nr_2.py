import cv2
import numpy as np


# Function to apply Gaussian filtering in the frequency domain
# Inputs:
# - img: The input image (assumed grayscale, converts if necessary)
# - d0: The standard deviation of the Gaussian filter (determines spread)
def gaussian_fourier_filter(img, d0):
    # Uncomment the line below to resize the image for performance on slower systems
    # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Convert image to grayscale if it is not already
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the image
    m, n = img.shape

    # Initialize the Gaussian filter array with zeros
    G = np.zeros((m, n))

    # Define the center coordinates of the frequency spectrum
    x0 = m // 2
    y0 = n // 2

    # Convert the image to a NumPy array
    img_data = np.asarray(img)

    # Perform a 2D Fourier Transform and shift zero-frequency to the center
    f = np.fft.fft2(img_data)
    f_shift = np.fft.fftshift(f)

    # Create the Gaussian filter in the frequency domain
    for i in range(-(m // 2), (m // 2)):
        for j in range(-(n // 2), (n // 2)):
            x = i + x0  # Map negative indices to positive
            y = j + y0
            dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5  # Calculate distance from center
            G[x, y] = np.exp(-(dist ** 2) / (2 * (d0 ** 2)))  # Gaussian filter equation

    # Apply the Gaussian filter to the frequency domain representation
    filtered_image = f_shift * G

    # Shift the frequency spectrum back to the original layout
    filtered_image = np.fft.ifftshift(filtered_image)

    # Perform the inverse Fourier Transform to return to spatial domain
    filtered_image = np.fft.ifft2(filtered_image)

    # Take the absolute value to get the real part of the image
    filtered_image = np.abs(filtered_image)

    # Convert the result to an 8-bit unsigned integer array for display
    filtered_image = np.array(filtered_image, dtype=np.uint8)
    return filtered_image


# Load a color image (assumes "city_hall.jpg" exists) in its original format
img = cv2.imread("city_hall.jpg")

# Display the original image in a window
cv2.imshow("Window", img)

# Apply the Gaussian Fourier filter with a spread parameter (e.g., d0 = 30)
img_filtered = gaussian_fourier_filter(img, 30)

# Display the filtered image in a second window
cv2.imshow("Window2", img_filtered)

# Wait for a key press before closing the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
