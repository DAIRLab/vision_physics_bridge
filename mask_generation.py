import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
frame_id = 1

# img = './rgb_data/frame00000{}.png'.format(frame_id)
# save = './rgb_data/generated/frame00000{}.png'.format(frame_id)

# im = cv2.imread(img)
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# result = np.zeros_like(im)
# cnt = contours[6]
# # cv2.drawContours(result, [cnt], 0, (0,255,0), 3)
# cv2.drawContours(result, contours, -1, (0,255,0), 3)

# # save results
# print("Saving image mask ")
# cv2.imwrite(save, result)
# print(contours)


# Simple thresholding type on an image
def simple_thresholding():
    # path to input image is specified and 
    # image is loaded with imread command
    image1 = cv2.imread('./rgb_data/frame00000{}.png'.format(frame_id))
    
    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    
    # applying different thresholding
    # techniques on the input image
    # all pixels value above 120 will
    # be set to 255
    ret, thresh1 = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 230, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 230, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 230, 255, cv2.THRESH_TOZERO_INV)
    
    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input images
    cv2.imshow('Binary Threshold', thresh1)
    cv2.imshow('Binary Threshold Inverted', thresh2)
    cv2.imshow('Truncated Threshold', thresh3)
    cv2.imshow('Set to 0', thresh4)
    cv2.imshow('Set to 0 Inverted', thresh5)
    
    # De-allocate any associated memory usage 
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
def otsu_implementation(img_title="./rgb_data/frame000001.png", is_normalized=False, is_reduce_noise=False):
    # Read the image in a greyscale mode
    image = cv2.imread(img_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

def call_otsu_threshold(img_title="./rgb_data/frame000001.png", is_reduce_noise=False):
    # Read the image in a greyscale mode
    image = cv2.imread(img_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # View initial image histogram
    plt.hist(image.ravel(), 256)
    plt.xlabel('Colour intensity')
    plt.ylabel('Number of pixels')
    plt.savefig("image_hist.png")
    plt.close()

    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    print("Obtained threshold: ", otsu_threshold)

    # View the resulting image histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image_result.ravel(), 256)
    ax.set_xlabel('Colour intensity')
    ax.set_ylabel('Number of pixels')
    # Get rid of 1e7
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: ('%1.1fM') % (x*1e-6)))
    plt.savefig("image_hist_result.png")
    plt.close()

    # Visualize the image after the Otsu's method application
    cv2.imshow("Otsu's thresholding result", image_result)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def otsu():
    call_otsu_threshold()
    otsu_implementation()

# if __name__ == "__main__":s
    # simple_thresholding()
    # otsu()
