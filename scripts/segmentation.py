import numpy as np
import pandas as pd
from skimage import io, morphology, filters, color,\
                    segmentation, measure
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy.spatial import distance


# Not used in our new jupyter notebook which we used for our final submission
def segment_lesion(img, debug=False):

    gray_img = color.rgb2gray(img)
    image_center = np.asarray(gray_img.shape) / 2
    #Apply Sobel filter for edge detection
    elevation_map = filters.sobel_v(gray_img)
    #Build image markers using the threshold obtained through the ISODATA filter
    markers = np.zeros_like(gray_img)
    threshold = filters.threshold_isodata(gray_img)
    markers[gray_img > threshold] = 1
    markers[gray_img < threshold] = 2
    #Apply Wathershed algorithm in order to segment the image filtered using the markers
    segmented_img = segmentation.watershed(elevation_map, markers)
    # Fill small holes
    segmented_img = ndi.binary_fill_holes(segmented_img - 1)
    # Remove small objects
    segmented_img = morphology.remove_small_objects(segmented_img, min_size=800)
    #  Clear regions connected to the image borders.
    #Apply connected components labeling algorithm:
    labeled_img = morphology.label(segmented_img)
    if debug:
        # create a subplot of 3 figures in order to show elevation map,
        # markers and the segmented image
        fig, ax = plt.subplots(1, 4, figsize=(10, 8))
        ax[0].imshow(elevation_map, cmap=plt.cm.gray)
        ax[0].set_title('elevation map')
        ax[0].set_axis_off()

        ax[1].imshow(markers, cmap=plt.cm.nipy_spectral)
        ax[1].set_title('markers')
        ax[1].set_axis_off()

        ax[2].imshow(segmented_img, cmap=plt.cm.gray)
        ax[2].set_title('segmentation')
        ax[2].set_axis_off()

        ax[3].imshow(gray_img, cmap=plt.cm.gray)
        ax[3].set_title('gray')
        ax[3].set_axis_off()

        plt.tight_layout()
        plt.show();
    # 6] Lesion identification algorithm to compute properties for the regions
    props = measure.regionprops(labeled_img)
    # num labels -> num regions
    num_labels = len(props)
    # Get all the area of detected regions
    areas = [region.area for region in props]

    if not areas:
        target_label = None
        return target_label

    central = [region.centroid for region in props]
    extents = [region.extent for region in props]
    distance_to_center = [distance.euclidean(image_center, center) for center in central]
    if debug:
        print(f'Num labels: {num_labels}')
        print(f'Areas: {areas}')
        print(f'Extents: {extents}')
        print(f'Distance: {distance_to_center}')


    possible_target = []
    if areas[np.argmin(distance_to_center)] >=1500:
        target_label = props[np.argmin(distance_to_center)].label
    else:
        target_label = props[np.argmax(areas)].label
    # Get the index of the region having the largest area
    region_max1 = np.argmax(areas)
    if num_labels > 1:
        areas_copy = areas.copy()
        areas_copy[region_max1] = 0
        region_max2 = np.argmax(areas_copy)
    if num_labels > 2:
        areas_copy[region_max2] = 0
        region_max3 = np.argmax(areas_copy)
    if num_labels > 3:
        areas_copy[region_max3] = 0
        region_max4 = np.argmax(areas_copy)
    # Get the 4 biggest regions if extent is bigger than 0.3 and area bigger than 1200px
    if extents[region_max1] > 0.40 and areas[region_max1] >= 1500:
        possible_target.append(region_max1)
    if num_labels > 1 and extents[region_max2] > 0.40 and areas[region_max2] >= 1500:
        possible_target.append(region_max2)
    if num_labels > 2 and extents[region_max3] > 0.40 and areas[region_max3] >= 1500:
        possible_target.append(region_max3)
    if num_labels > 3 and extents[region_max4] > 0.40 and areas[region_max4] >= 1500:
        possible_target.append(region_max4)

    # Select the region with min distance to image center
    if possible_target:
        possible_target_distances = [distance_to_center[index] for index in possible_target]
        min_index = np.argmin(possible_target_distances)
        target_label = props[possible_target[min_index]].label
        if debug:
            print(f'Possible targets: {possible_target}')
            print(f'Possible distances: {possible_target_distances}')
            print(f"Chosen index {possible_target[min_index]}")



    if debug:
        for row, col in np.ndindex(labeled_img.shape):
            if labeled_img[row, col] != target_label:
                labeled_img[row, col] = 0
        image_label_overlay = color.label2rgb(labeled_img, gray_img)
        print(f'Chosen label: {target_label}')
        # Plot the original image ('image') in which the contours of all the
        # segmented regions are highlighted
        fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
        axes[0].imshow(img)
        axes[0].contour(segmented_img, [0.5], linewidths=1.2, colors='y')
        axes[0].axis('off')
        # Plot 'image_label_overlay' that contains the target region highlighted
        axes[1].imshow(image_label_overlay)
        axes[1].axis('off')
        plt.tight_layout()
        plt.show();
    return props[target_label -1]


def get_lesion_region(train_csv_path, val_csv_path, preprocessed_train_set_path,
                      preprocessed_val_set_path, debug=False):
    try:
        df_train = pd.read_csv(train_csv_path)
        df_val = pd.read_csv(val_csv_path)
        segmented_lesion_train_set = {}
        segmented_lesion_val_set = {}
        current_image_name = ""
        print('Start to segment lesion out of training set')
        for i, image in df_train.iterrows():
            current_image_name = image['image']
            img = io.imread(preprocessed_train_set_path + image['image']+'.jpg')
            region = segment_lesion(img,debug)
            segmented_lesion_train_set[image['image']] = region
            print('Training image name ' + image['image'])
            print('Segmented training image ' + str(i))
        print('Finished lesion segmentation for training set')


        print('Start to segment lesion out of validation set')
        for i, image in df_val.iterrows():
            current_image_name = image['image']
            img = io.imread(preprocessed_val_set_path + image['image']+'.jpg')
            region = segment_lesion(img,debug)
            segmented_lesion_val_set[image['image']] = region
            print('Validation image name' + image['image'])
            print('Segmented validation image ' + str(i))
        print('Finished lesion segmentation for validation set')

        k = 0
        for key,value in segmented_lesion_train_set.items():
            if value is None:
                k+=1
        print('Found this amount of none values for training set ' + str(k))

        k = 0
        for key, value in segmented_lesion_val_set.items():
            if value is None:
                k += 1
        print('Found this amount of none values for validation set ' + str(k))

        print('Remove images where algorithm could not segment out lesion')

        # Remove entries which are None
        lesion_train_set = {}
        for key, value in segmented_lesion_train_set.items():
            if value is not None:
                lesion_train_set[key] = value
            else:
                print(f'Remove image from training set {key}')

        # Remove entries which are None
        lesion_val_set = {}
        for key, value in segmented_lesion_val_set.items():
            if value is not None:
                lesion_val_set[key] = value
            else:
                print(f'Remove image from validation set {key}')

        print('Done. Removed all images where algorithm could not segment out lesion')
        return lesion_train_set, lesion_val_set
    except Exception as e:
        print(f'Problem with following image {current_image_name}')
