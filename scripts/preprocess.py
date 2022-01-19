import cv2
import numpy as np
import pandas as pd
from skimage import exposure, morphology, filters, img_as_ubyte, img_as_float
from skimage.color.adapt_rgb import adapt_rgb, each_channel


def enlarge_image(img):
    '''
    Enlarge image with dark border areas
    :param img: ndarray, BGR image
    :return: ndarray, BGR image
    '''
    f = np.zeros((700, 700,3), np.uint8)
    ax, ay = (700 - img.shape[1]) // 2, (700 - img.shape[0]) // 2
    f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
    return f


def reduce_image(img):
    '''
    Resize image to size 600x600 and remove dark border areas
    :param img: ndarray, BGR image
    :return: ndarray, BGR image
    '''
    ax, ay = (img.shape[1]-600)//2, (img.shape[0] - 600) // 2
    f = img[ay:600 + ay, ax:ax + 600]
    return f


def reduce_mask(img):
    '''
    Reuce mask size to 600x600
    :param img: ndarray, binary image
    :return: ndarray, binary image
    '''
    ax, ay = (img.shape[1]-600)//2, (img.shape[0] - 600) // 2
    f = img[ay:600 + ay, ax:ax + 600]
    return f


def remove_black_border(gray_image):
    '''
    Remove black border areas
    :param gray_image: ndarray, grayscale image
    :return: ndarray, binary image
    '''
    _, mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY);
    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    mask_a = np.zeros((700, 700), np.uint8)
    cv2.drawContours(mask_a, [c], -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((15, 15), np.uint8)
    eroded = cv2.erode(mask_a, kernel, iterations=3)
    _, mask_a = cv2.threshold(eroded, 10, 255, cv2.THRESH_BINARY);
    return mask_a


@adapt_rgb(each_channel)
def morph_closing_each(image, struct_element):
    return morphology.closing(image, struct_element)


@adapt_rgb(each_channel)
def median_filter_each(image, struct_element):
    return filters.median(image, struct_element)


structuring_element = morphology.disk(7)


def crop_center_rgb(img, cropx, cropy):
    '''
    Crop image from the center
    :param img: ndarray, BGR image
    :param cropx: width in int
    :param cropy: height in int
    :return: ndarray, BGR cropped image
    '''
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]


def noise_removal(img):
    '''
    Remove noise in the image
    :param img: ndarray, BGR image
    :return: ndarray, BGR filtered image
    '''
    img = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    equalized_adapthist = exposure.equalize_adapthist(img)
    img_morph_closing = morph_closing_each(equalized_adapthist, structuring_element)
    img_filtered = median_filter_each(img_morph_closing, structuring_element)
    img_filtered = cv2.cvtColor(img_as_ubyte(img_filtered), cv2.COLOR_RGB2BGR)
    return img_filtered


# Not used anymore
# See https://www.kaggle.com/apacheco/shades-of-gray-color-constancy
def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)


def preprocess_image(img):
    '''
    Preprocess image
    :param img: ndarray, BGR image
    :return:  ndarray, BGR image
    '''
    img = noise_removal(img)
    return img


def crop_image(img):
    '''
    Remove dark border areas and crop image
    :param img: ndarray, BGR image
    :return: ndarray, BGR image
    '''
    img = cv2.resize(img, (600, 600))
    inpaint_image = enlarge_image(img)

    # Remove black border
    gray = cv2.cvtColor(inpaint_image, cv2.COLOR_BGR2GRAY)

    mask = remove_black_border(gray)
    #mean = cv2.mean(inpaint_image, mask)
    mask = reduce_mask(mask)
    inpaint_image = reduce_image(inpaint_image)
    inpaint_image[mask == 0] = 0

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    inpaint_image = inpaint_image[y:y + h, x:x + w]
    inpaint_image = cv2.resize(inpaint_image, (600, 600))
    # inpaint_image = shade_of_gray_cc(inpaint_image)

    # Crop from the center the image if the borders are black
    gray_img = cv2.cvtColor(inpaint_image, cv2.COLOR_BGR2GRAY)
    if gray_img[0][0] < 10 and gray_img[0][-1] < 10 and gray_img[-1][0] < 10 and gray_img[-1][-1] < 10:
        inpaint_image = cv2.resize(crop_center_rgb(inpaint_image, 400, 400), (600,600))
    return inpaint_image


def preprocess_dataset(dataset_path, dataset_csv_path, preprocessed_dataset_path):
    '''
    Preprocess training and validation dataset
    :param dataset_path: Path to dataset
    :param dataset_csv_path: Path to training csv file
    :param val_csv_path: Path to validation csv file
    :param preprocessed_dataset_path: Path where to save preprocessed  images
    :return:
    '''

    df_train = pd.read_csv(dataset_csv_path)

    print('Start of preprocessing step of dataset ')
    for i, image in df_train.iterrows():
        img = cv2.imread(dataset_path + image['image']+'.jpg', cv2.IMREAD_COLOR)
        img = preprocess_image(img)
        cv2.imwrite(preprocessed_dataset_path + image['image']+'.jpg', img)
        print(str(i) + ': Preprocessed image ' + image['image'])
    print('Finished preprocess step of dataset')
    print('preprocessed images saved in ' + preprocessed_dataset_path)
    print('Finished script')


def crop_dataset(dataset_path, dataset_csv_path):
    '''
    Crop training and validation dataset
    :param dataset_path: Path to training dataset
    :param dataset_csv_path: Path to training csv file
    :return:
    '''

    df_dataset = pd.read_csv(dataset_csv_path)

    print('Start of dataset cropping step')
    for i, image in df_dataset.iterrows():
        img = cv2.imread(dataset_path + image['image']+'.jpg', cv2.IMREAD_COLOR)
        img = crop_image(img)
        cv2.imwrite(dataset_path + image['image']+'.jpg', img)
        print(str(i) + ': Cropped image ' + image['image'])
    print('Finished cropping step of dataset')
    print('cropped  images saved in ' + dataset_path)