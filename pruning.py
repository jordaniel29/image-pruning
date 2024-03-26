import cv2
import numpy as np

# CONSTANT VARIABLE DECLARATION
block_size = [4, 4]
empty_block = np.ones((block_size[0], block_size[1])) * 128
threshold = 320

# Reshape the image to 512x512, change it to gray color, and convert it to matrix
def image_to_matrix(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (512, 512))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Sum the difference between each pixels in the 2 blocks
def calculate_loss(block1, block2):
    return np.sum(np.abs(block1 - block2))

# Return pruned image based on 2 input images using motion estimation
def prune_image(img_left, img_right):
    img_out = np.copy(img_right)

    for i in range(0, len(img_right) - block_size[0] + 1, block_size[0]):
        print(i) # Logging purposes 
        for j in range(0, len(img_right[0]) - block_size[1] + 1, block_size[1]):
            
            for k in range(0, len(img_left) - block_size[0] + 1, block_size[0]):
                for l in range(0, len(img_left[0]) - block_size[1] + 1, block_size[1]):
                    current_loss = calculate_loss(img_right[i:i+block_size[0], j:j+block_size[1]],
                                                  img_left[k:k+block_size[0], l:l+block_size[1]])
                    
                    # If there is a block in the left image that has loss less or equal than the threshold,
                    # stop the looping in the left image and remove the block the pruned image 
                    if current_loss <= threshold:
                       img_out[i:i+block_size[0], j:j+block_size[1]] = empty_block
                       break

                if current_loss <= threshold:
                    break

    return img_out

if __name__ == "__main__":
    # Input & Output File Location
    img = [["./input/coffee_1.jpg", "./input/coffee_2.jpg"], ["./input/bag_1.jpg", "./input/bag_2.jpg"], ["./input/bottle_1.jpg", "./input/bottle_2.jpg"]]
    img_out = ["./output/coffee_pruned", "./output/bag_pruned", "./output/bottle_pruned"]

    # Looping for each input image
    for i in range(len(img)):
        imgleft_path =  img[i][0]
        imgright_path = img[i][1]

        # Preprocess the image (convert to grayscale, resize, and convert to matrix)
        imgleft_matrix = image_to_matrix(imgleft_path)
        imgright_matrix = image_to_matrix(imgright_path)

        # Prune the 2 images using motion estimation
        pruned_img = prune_image(imgleft_matrix, imgright_matrix)

        # Saved pruned image to a file
        output_path = img_out[i] + "_8x8_th" + str(threshold) + ".jpg"
        cv2.imwrite(output_path, pruned_img)
        print("Output image saved to: " + output_path)