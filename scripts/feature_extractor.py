import numpy as np
import cv2
import mahotas
from skimage.feature import local_binary_pattern
from skimage import feature
import pandas as pd


# Features implementation is from following repo:
# https://github.com/data-metrics/skin-lesion-classification

def extract_hu_moments(img):
    """Extract Hu Moments feature of an image. Hu Moments are shape descriptors.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains 7 Hu Moments of the image
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature


def extract_zernike_moments(img, radius=21, degree=8):
    """Extract Zernike Moments feature of an image. Zernike Moments are shapre descriptors.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains 25 Zernike Moments of the image
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.zernike_moments(gray, radius, degree)
    return feature


def extract_haralick(img):
    """Extract Haralick features of an image. Haralick features are texture descriptors.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains 13 Haralick features of the image
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(gray).mean(axis=0)
    return feature


def extract_lbp(img, numPoints=24, radius=8):
    """Extract Local Binary Pattern histogram of an image. Local Binary Pattern features are texture descriptors.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains (numPoints+2) Local Binary Pattern histogram of the image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, numPoints, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    feature, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return feature


def extract_color_histogram(img, n_bins=8):
    """Extract Color histogram of an image.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains n_bins*n_bins*n_bins HSV histogram features of the image
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert the image to HSV color-space
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    feature = hist.flatten()
    return feature

def get_texture(img):
    '''
    Get texture information of the image
    :param img: ndarray, BGR image
    :return: correlation, homogeneity, energy, contrast computed by skimage.feature.greycoprops()
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = feature.greycomatrix(image=gray, distances=[1],
                                angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 2],
                                symmetric=True, normed=True)

    correlation = np.mean(feature.greycoprops(glcm, prop='correlation'))
    homogeneity = np.mean(feature.greycoprops(glcm, prop='homogeneity'))
    energy = np.mean(feature.greycoprops(glcm, prop='energy'))
    contrast = np.mean(feature.greycoprops(glcm, prop='contrast'))

    return correlation, homogeneity, energy, contrast


def extract_global_features(img):
    """Extract global features (shape, texture and color features) of an image.
    :param img: ndarray, BGR image
    :return global_feature: ndarray, contains shape, texture and color features of the image
    """

    hu_moments = extract_hu_moments(img)
    zernike_moments = extract_zernike_moments(img)
    haralick = extract_haralick(img)
    lbp_histogram = extract_lbp(img)
    color_histogram = extract_color_histogram(img)
    correlation, homogeneity, energy, contrast = get_texture(img)
    global_feature = np.hstack([hu_moments, zernike_moments, haralick, lbp_histogram, color_histogram,
                                correlation, homogeneity, energy, contrast])

    return global_feature


def extract_features(train, val, train_path, val_path):
    '''
    Extract all the features for training and validation set
    :param train: Dict of all training image names with groundtruth class
    :param val: Dict of all validation image names with groundtruth class
    :param train_path: Path to the training images
    :param val_path: Path to the validation images
    :return: np.array: global features of training images and global features of validation images
    '''
    train_global_features = []
    val_global_features = []

    dataset = {**train, **val}
    for idx, image_name in enumerate(dataset):

        if image_name in train:
            image_path = train_path + image_name + '.jpg'
        elif image_name in val:
            image_path = val_path + image_name + '.jpg'
        else:
            print('Error: Cannot find {}'.format(image_name))
            return None

        img = cv2.imread(image_path)
        global_feature = extract_global_features(img)
        if image_name in train:
            train_global_features.append(global_feature)
        elif image_name in val:
            val_global_features.append(global_feature)
        else:
            print('Error: Cannot find {}'.format(image_name))
            return None
    return np.array(train_global_features), np.array(val_global_features)


def extract_testing_features(testing_csv, testing_set_path):
    '''
    Extract features of the testing dataset
    :param testing_csv: Path to the testing csv file
    :param testing_set_path: Path to the testing images
    :return: np.array: global features of testing images
    '''
    testing_global_features = []
    
    df_testing = pd.read_csv(testing_csv)

    for idx, image in df_testing.iterrows():
        image_path = testing_set_path + image['image'] + '.jpg'
        img = cv2.imread(image_path)
        global_feature = extract_global_features(img)
        testing_global_features.append(global_feature)

    return np.array(testing_global_features)