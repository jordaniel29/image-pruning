import cv2
import numpy as np

# CONSTANT VARIABLE DECLARATION
block_size = [8, 8]
empty_block = np.ones((block_size[0], block_size[1])) * 128
threshold = 1280

# Reshape the image to 512x512, change it to gray color, and convert it to matrix
def image_to_matrix(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (512, 512))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Calculate total loss / difference between 2 blocks
def calculate_loss(block1, block2):
    return np.sum(np.abs(block1 - block2))

# Return pruned image based on 2 input images
def prune_image(img_left, img_right):
    img_out = np.copy(img_right)

    for i in range(0, len(img_right) - block_size[0] + 1, block_size[0]):
        print(i)
        for j in range(0, len(img_right[0]) - block_size[1] + 1, block_size[1]):
            
            for k in range(0, len(img_left) - block_size[0] + 1, block_size[0]):
                for l in range(0, len(img_left[0]) - block_size[1] + 1, block_size[1]):
                    current_loss = calculate_loss(img_right[i:i+block_size[0], j:j+block_size[1]],
                                                  img_left[k:k+block_size[0], l:l+block_size[1]])
                    if current_loss <= threshold:
                       img_out[i:i+block_size[0], j:j+block_size[1]] = empty_block
                       break

    return img_out

# MAIN FUNCTION
if __name__ == "__main__":
    # Input images
    imgleft_path = "test1.bmp"
    imgright_path = "test2.bmp"

    # Convert the image to a matrix
    imgleft_matrix = image_to_matrix(imgleft_path)
    imgright_matrix = image_to_matrix(imgright_path)

    pruned_img = prune_image(imgleft_matrix, imgright_matrix)

    # Saved pruned image to a file
    output_path = "pruned_image2.jpg"
    cv2.imwrite(output_path, pruned_img)
    print("Output image saved to: " + output_path)