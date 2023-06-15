import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

root_path = os.getcwd()
app_path = f"{root_path}/app"

## Reference that was used to apply these tranforms: https://docs.ufpr.br/~centeno/p_pdi/pdf/jaulapdi04.pdf

def calculate_fourier_transform(img):
    # Get the image size
    width, height = img.size

    # Create an empty array to store the pixel transforms
    pixel_transforms = np.zeros((height, width), dtype=np.complex128)

    # Calculate the Fourier Transform for each pixel
    for y in range(height):
        for x in range(width):
            # Initialize the real and imaginary parts of the transform
            real_part = 0
            imaginary_part = 0

            # Perform the manual Fourier Transform calculation
            for u in range(height):
                for v in range(width):
                    # Get the pixel value
                    pixel_value = img.getpixel((v, u))

                    angle = 2 * np.pi * ((y * v / width) + (x * u / height))
                    real_part += pixel_value * np.cos(angle)
                    imaginary_part += pixel_value * np.sin(angle)

            # Store the transform in the array
            pixel_transforms[y, x] = (real_part - imaginary_part * 1j) / (height * width)

    return pixel_transforms

def calculate_fourier_inverse_transform(transforms):
    # Get the image size
    height, width = transforms.shape

    # Create an empty array to store the inverse transforms
    inverse_transforms = np.zeros((height, width), dtype=np.float64)

    # Calculate the inverse Fourier Transform for each pixel
    for y in range(height):
        for x in range(width):
            # Initialize the real and imaginary parts of the inverse transform
            real_part = 0
            imaginary_part = 0

            # Perform the manual inverse Fourier Transform calculation
            for u in range(height):
                for v in range(width):
                    angle = 2 * np.pi * ((u * x / width) + (v * y / height))
                    real_part += transforms[u, v].real * np.cos(angle) - transforms[u, v].imag * np.sin(angle)
                    imaginary_part += transforms[u, v].real * np.sin(angle) + transforms[u, v].imag * np.cos(angle)

            # Normalize and store the inverse transform in the array
            inverse_transforms[y, x] = real_part + imaginary_part

    return inverse_transforms

def compress_frequency(f_transforms, compression_percentage):
    # Flatten and sort the transformed array
    sorted_f_transforms = np.sort(np.abs(f_transforms).flatten())

    # Determine the threshold value
    threshold_index = int((1 - (1 - compression_percentage)) * len(sorted_f_transforms))
    threshold = sorted_f_transforms[threshold_index]

    # Apply the threshold to the transformed array
    f_transforms[np.abs(f_transforms) < threshold] = 0

    return f_transforms

def main():
    compress_percentage = float(input('Please type a percentage of compression (decimal between 0 and 1): '))

    img_name = 'img_3'
    img = Image.open(f'{app_path}/assets/{img_name}.png')
    gray_img = img.convert('L')

    ### Discrete Fourier Transform
    f_transforms = calculate_fourier_transform(gray_img)

    ### Calculate the magnitude spectrum that is used to see the transform as a image
    magnitude_spectrum = np.log(np.abs(f_transforms))

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Transformed Image')
    plt.show()

    ### Compressing Transform
    compressed_f_transforms = compress_frequency(f_transforms, compress_percentage)

    ### Discrete Inverse Fourier Transform
    inverse_f_transforms = calculate_fourier_inverse_transform(compressed_f_transforms)

    reconstructed_image = Image.fromarray(inverse_f_transforms).convert('L')

    ### Save in results file
    results_path = f'{app_path}/results'
    os.makedirs(results_path, exist_ok=True)
    gray_img.save(f'{results_path}/{img_name}_grayscale.png')
    reconstructed_image.save(f'{results_path}/{img_name}_compressed.png')

    ### Plot the images
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.xticks([])
    plt.yticks([])

    plt.show()

main()