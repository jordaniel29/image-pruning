import cv2
import numpy as np

block_size = [20, 20]
block_empty = np.ones((block_size[0], block_size[1]))*255
threshold = 5000


def loss_function(value1, value2):
    # Convert pixel values to a signed integer type to handle overflow
    value1 = np.int16(value1)
    value2 = np.int16(value2)
    return np.abs(value1 - value2)


def calculate_loss(block1, block2):
    total_loss = 0

    for i in range(len(block1)):
        for j in range(len(block1)):
            total_loss += loss_function(block1[i][j], block2[i][j])

    return total_loss


def prune_image(img_left, img_right):
    # img_out = np.zeros((len(img_left), len(img_left[0])))
    img_out = np.copy(img_right)

    for i in range(0, len(img_left) - block_size[0] + 1, block_size[0]):
        for j in range(0, len(img_left[0]) - block_size[1] + 1, block_size[1]):

            loss = calculate_loss(img_left[i:i+block_size[0], j:j+block_size[1]],
                                  img_right[i:i+block_size[0], j:j+block_size[1]])
            if loss < threshold:
                img_out[i:i+block_size[0], j:j + block_size[1]] = block_empty

            # min_loss = 1000000
            # for k in range(len(img_right) - block_size[0]+1):
            #     for l in range(len(img_right[0]) - block_size[1]+1):
            #         current_loss = calculate_loss(img_left[i:i+block_size[0], j:j+block_size[1]],
            #                                       img_right[k:k+block_size[0], l:l+block_size[1]])
            #         if current_loss < min_loss:
            #             min_loss = current_loss

            # if min_loss >= threshold:
            #     # Assign the pixel value from the right image if the loss is below threshold
            #     img_out[i:i+block_size[0], j:j+block_size[1]
            #             ] = img_right[k:k+block_size[0], l:l+block_size[1]]

    return img_out


def image_to_matrix(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to a NumPy array
    image_matrix = np.array(gray_image)

    return image_matrix


if __name__ == "__main__":
    # Input images
    imgleft_path = "pic1.jpeg"
    imgright_path = "pic2.jpeg"

    # Convert the image to a matrix
    imgleft_matrix = image_to_matrix(imgleft_path)
    imgright_matrix = image_to_matrix(imgright_path)
    print(imgleft_matrix.shape)

    pruned_img = prune_image(imgleft_matrix, imgright_matrix)

    # Display pruned image
    cv2.imshow("Pruned Image", pruned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
