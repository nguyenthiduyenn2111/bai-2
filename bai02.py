import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images
def display_images(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.show()

# Step 1: Load the input image
image_path = 'b1.jpg '
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Sobel Edge Detection
# Sobel Kernels
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]])

# Applying the Sobel filters
sobel_x_edges = cv2.filter2D(image, -1, sobel_x)
sobel_y_edges = cv2.filter2D(image, -1, sobel_y)

# Magnitude of gradient
sobel_edges = np.sqrt(sobel_x_edges**2 + sobel_y_edges**2)
sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))

# Step 3: Laplacian of Gaussian (LoG) Edge Detection
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])

log_edges = cv2.filter2D(image, -1, log_kernel)

# Display the results
display_images([image, sobel_x_edges, sobel_y_edges, sobel_edges, log_edges],
               ['Original Image', 'Sobel X', 'Sobel Y', 'Sobel Edges', 'LoG Edges'])