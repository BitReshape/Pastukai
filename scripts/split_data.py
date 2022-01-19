import pandas as pd
import shutil
import cv2
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte, img_as_float
import random


def get_classification(data_series):
    '''

    :param data_series: dataserie containing the one hotencoding
    :return: classification as string
    '''
    classification = None
    for index, value in data_series.items():
        if value == 1.0:
            classification = index
    return classification


# Augmentation functions
def flip_image(img, vflip=False, hflip=False):
    '''
    Flip image vertically or horizontally
    :param img: ndarray, BGR image
    :param vflip: bool if vertically flip
    :param hflip: bool if horizontally flip
    :return: ndarray, BGR image
    '''
    if hflip or vflip:
        if hflip and vflip:
            c = -1
        else:
            c = 0 if vflip else 1
        image = cv2.flip(img, flipCode=c)
    return image


def decrease_brightness(img):
    '''
    Decrease brightness of image
    :param img: ndarray, BGR image
    :return: ndarray, BGR image
    '''
    bright = np.ones(img.shape, dtype="uint8") * -50
    bright_image = cv2.add(img,bright)
    return bright_image


def add_noise_image(img):
    '''
    Add random noise to image
    :param img: ndarray, BGR image
    :return: ndarray, BGR image
    '''
    img = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    noise = random_noise(img, mode='s&p', amount=0.011)
    noise = cv2.cvtColor(img_as_ubyte(noise), cv2.COLOR_RGB2BGR)
    return noise


def rotate_image(img, angle):
    '''
    Rotate image
    :param img: ndarray, BGR image
    :param angle: angle of rotation as int
    :return: ndarray, BGR image
    '''
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, matrix, (w, h))
    return img


def prepare_dataset(dataset_path, new_set_path, csv_path,
                    reduced_csv_path, sample_number=50, validation=False):
    '''
    Prepare dataset and split it into training and validation dataset
    :param dataset_path: Path to the original dataset with all images
    :param new_set_path: Path where to save  images
    :param csv_path: Path to the original  csv
    :param reduced_csv_path: Path to the new  csv containing the pictures
    :param sample_number: Number of samples per class for training dataset , maximum 1000
    :param validation: Bool if dataset is validation set
    :return:
    '''
    try:
        new_dataset_dict = []
        dataset_classes_dict = {}

        dataset_path = dataset_path
        new_set_path = new_set_path
        df_dataset = pd.read_csv(csv_path)
        classes_counter = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}

        # Shuffle dataset
        if not validation:
            df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)

        # Iterate over dataset
        for i, image in df_dataset.iterrows():
            dict1 = {}
            if min(classes_counter.values()) > sample_number:
                break
            image_class = get_classification(image)
            if image_class is not None and ((classes_counter[image_class] < sample_number) or validation):
                img = cv2.imread(dataset_path + image['image'] + '.jpg')
                img = cv2.resize(img, (600,600))
                cv2.imwrite(new_set_path + image['image'] + '.jpg', img)
                shutil.copy(dataset_path + image['image']+'.jpg', new_set_path)
                dict1.update(image)
                new_dataset_dict.append(dict1)
                dataset_classes_dict[image['image']] = image_class
                classes_counter[image_class] += 1


        # Get distribution printed before augmentation
        for index, value in classes_counter.items():
            print('Dataset distribution:')
            print('Class: ' + index)
            print('Number of images: ' + str(value))
        print('##########################################')

        if not validation:
            # Data augmentation
            for index, value in classes_counter.items():
                print('\n Data augmentation for class ' + index)

                if value >= sample_number:
                    print(index + ' has ' + str(value) + ' images. Skipping augmenatation for this class')
                    continue
                else:
                    for i, image in df_dataset.iterrows():
                        if classes_counter[index] > sample_number:
                            break
                        image_class = get_classification(image)
                        if image_class == index:
                            print('Add augmented images for ' + image['image'])
                            print('Number of samples for class: ' + index + '     ' + str(classes_counter[index]))
                            img = cv2.imread(dataset_path + image['image']+'.jpg')
                            img = cv2.resize(img, (600, 600))
                            flip_h = flip_image(img,hflip=True)
                            flip_v = flip_image(img, vflip=True)
                            noise = add_noise_image(img)
                            rotate = rotate_image(img, 90)
                            rotate2 = rotate_image(img, 270)

                            augmented_images = [flip_h, flip_v, rotate2, noise, rotate]

                            for idx, aug_image in enumerate(augmented_images):
                                dict1 = {}
                                aug_image_name = f"{image['image']}_{idx}"
                                cv2.imwrite(new_set_path + aug_image_name + ".jpg", aug_image)
                                dict1.update(image)
                                dict1['image'] = aug_image_name
                                new_dataset_dict.append(dict1)
                                dataset_classes_dict[aug_image_name] = image_class
                                classes_counter[index]+=1

            print('############################################')
            # Get distribution printed
            for index, value in classes_counter.items():
                print('Training set distribution:')
                print('Class: ' + index)
                print('Number of images: ' + str(value))

        df_updated = pd.DataFrame(new_dataset_dict)
        df_updated.to_csv(reduced_csv_path, index=False)

    except OSError as e:
        print(e)

    print("Data structure created")


def read_csv_files(train_csv, val_csv):
    '''
    Read csv files for training and validation
    :param train_csv: Path to train csv file
    :param val_csv: Path to val csv file
    :return: dict train_classes (key: image_name, value: class), dict val_classes (key: image_name, value: class)
    '''
    train_classes = {}
    val_classes = {}
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    for i, image in df_train.iterrows():
        image_class = get_classification(image)
        if image_class is not None:
            train_classes[image['image']] = image_class

    for i, image in df_val.iterrows():
        image_class = get_classification(image)
        if image_class is not None:
            val_classes[image['image']] = image_class

    return train_classes, val_classes
