import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Load the .nii.gz file
img1 = nib.load(r"G:\my-projects\medical-imaging\Data\sub-01\ses-1\func\sub-01_ses-1_task-rest_echo-1_bold.nii.gz")

# Get data as a numpy array
img_data1 = img1.get_fdata()

# Display some information about the image
# print("Image shape:", img_data1.shape)
# print("Affine:", img1.affine)

# Visualize a slice of the image data
slice_index1 = img_data1.shape[2] // 2  # Choose a middle slice
img_echo_1 = img_data1[:, :, slice_index1,0]
# plt.imshow(img_data1[:, :, slice_index1,0], cmap='gray')
# plt.title(f'Slice {slice_index1}')
# plt.axis('off')
# plt.show()
####################################################################################################################
# Load the .nii.gz file
img3 = nib.load(r"G:\my-projects\medical-imaging\Data\sub-01\ses-1\func\sub-01_ses-1_task-rest_echo-3_bold.nii.gz")

# Get data as a numpy array
img_data3 = img3.get_fdata()

# Display some information about the image
# print("Image shape:", img_data3.shape)
# print("Affine:", img3.affine)

# Visualize a slice of the image data
slice_index3 = img_data3.shape[2] // 2  # Choose a middle slice
img_echo_3 = img_data3[:, :, slice_index3,0]
# plt.imshow(img_data3[:, :, slice_index3,0], cmap='gray')
# plt.title(f'Slice {slice_index3}')
# plt.axis('off')
# plt.show()
########################################################################################################################
# Load the .nii.gz file
img2 = nib.load(r"G:\my-projects\medical-imaging\Data\sub-01\ses-1\func\sub-01_ses-1_task-rest_echo-2_bold.nii.gz")

# Get data as a numpy array
img_data2 = img2.get_fdata()

# Display some information about the image
# print("Image shape:", img_data2.shape)
# print("Affine:", img2.affine)

# Visualize a slice of the image data
slice_index2 = img_data2.shape[2] // 2  # Choose a middle slice
img_echo_2 = img_data2[:, :, slice_index2,0]
# plt.imshow(img_data2[:, :, slice_index2,0], cmap='gray')
# plt.title(f'Slice {slice_index2}')
# plt.axis('off')
# plt.show()


def calculate_phase_shift(TE, B0=1.5):
    # Chemical shift frequency in Hz (3.5 ppm at given B0)
    f_shift = 3.4*1e-6 * B0 * 42.58 * 1e6  # Convert ppm to Hz for B0    ,  (3.4 = sigma_w-sigma_B = 4.7-1.3)
    # Phase shift in radians
    theta = 2 * np.pi * f_shift * TE
    # Convert to degrees if preferred
    # theta_degrees = np.degrees(theta) % 360  # Wrap within [0, 360]
    return theta

# Example echo times in seconds
TEs = [0.0137, 0.03, 0.047]

phi_1 = calculate_phase_shift(TEs[0])
phi_2 = calculate_phase_shift(TEs[1])
phi_3 = calculate_phase_shift(TEs[2])


def solve_water_fat_images(I1, I2, I3, theta1, theta2, theta3):
    # Convert phase angles to complex exponentials (scalars)
    E1 = np.exp(1j * theta1)
    E2 = np.exp(1j * theta2)
    E3 = np.exp(1j * theta3)
    
    # Stack the images along a new axis to form a (H, W, 3) array, where H and W are image dimensions
    I_stack = np.stack([I1, I2, I3], axis=-1)  # Shape: (H, W, 3) y
    
    # Set up the design matrix A for each pixel
    A = np.array([
        [1, E1],
        [1, E2],
        [1, E3]
    ])  # Shape: (3, 2)
    
    # Invert A to solve using least-squares for all pixels
    A_pseudo_inverse = np.linalg.pinv(A)  # Shape: (2, 3)

    # Apply the pseudo-inverse to each pixel's intensity vector
    WF_stack = I_stack @ A_pseudo_inverse.T  # Shape: (H, W, 2), contains [W, F] for each pixel

    # Separate W and F
    W = WF_stack[..., 0]  # Water component, shape (H, W)
    F = WF_stack[..., 1]  # Fat component, shape (H, W)
    
    return W, F

images = [img_echo_1,img_echo_2,img_echo_3]
normalized_images = []
for i, img in enumerate(images):
    # Convert to float32 for precision in calculations
    img_float = img.astype(np.float32)
    
    # Calculate mean and standard deviation of the image
    mean, stddev = cv2.meanStdDev(img_float)
    mean = mean[0][0]
    stddev = stddev[0][0]
    
    # Apply z-score normalization
    normalized_img = (img_float - mean) / stddev
    normalized_images.append(normalized_img)
    
    # Optional: save the normalized images to disk after rescaling to the range [0, 255]
    rescaled_img = cv2.normalize(normalized_img, None, 0, 255, cv2.NORM_MINMAX)
    

water_signal, fat_signal = solve_water_fat_images(img_echo_1,img_echo_2,img_echo_3, phi_1, phi_2, phi_3)

# postprocessing
def enhance_image(img):

    return img
    # Apply Gaussian Blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

    return blurred_img

water_signal = enhance_image(np.abs(water_signal))
fat_signal = enhance_image(np.abs(fat_signal))
# Display the separated water and fat images
plt.figure(figsize=(12, 12))

# Display Image 1
plt.subplot(2, 3, 1)
plt.imshow(normalized_images[0], cmap='gray')
plt.title(f'Slice {slice_index1}')
plt.axis('off')
# Display Image 2
plt.subplot(2, 3, 2)
plt.imshow(normalized_images[1], cmap='gray')
plt.title(f'Slice {slice_index2}')
plt.axis('off')
# Display Image 3
plt.subplot(2, 3, 3)
plt.imshow(normalized_images[2], cmap='gray')
plt.title(f'Slice {slice_index3}')
plt.axis('off')

# Display Water Image
plt.subplot(2, 3, 4)
plt.imshow(np.abs(water_signal), cmap='gray')
plt.title('Water Image')
plt.axis('off')

# Display Fat Image
plt.subplot(2, 3, 5)
plt.imshow(np.abs(fat_signal), cmap='gray')
plt.title('Fat Image')
plt.axis('off')

plt.show()




##########################################################
# Display In-phase Image
plt.subplot(1, 2, 1)
plt.imshow(np.abs(water_signal+ fat_signal), cmap='gray')
plt.title('In-Phase Image')
plt.axis('off')

# Display Opposed-phase Image
plt.subplot(1, 2, 2)
plt.imshow(np.abs(water_signal - fat_signal), cmap='gray')
plt.title('Opposed-phase Image')
plt.axis('off')

plt.show()

