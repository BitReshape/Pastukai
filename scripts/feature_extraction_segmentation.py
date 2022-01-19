import numpy as np
import pandas as pd
from skimage import io,color, feature, img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
# %matplotlib inline

# Not used in our new jupyter notebook which we used for our final submission


# See https://github.com/biagiom/skin-lesions-classifier/blob/master/skin_lesions_classifier.ipynb
def imshow_all(*images, **kwargs):
    """
    Plot a series of images side-by-side.

    Convert all images to float so that images have a common intensity range.

    Parameters
    ----------
    limits : str
        Control the intensity limits. By default, 'image' is used set the
        min/max intensities to the min/max of all images. Setting `limits` to
        'dtype' can also be used if you want to preserve the image exposure.
    titles : list of str
        Titles for subplots. If the length of titles is less than the number
        of images, empty strings are appended.
    kwargs : dict
        Additional keyword-arguments passed to `imshow`.
    """
    images = [img_as_float(img) for img in images]

    titles = kwargs.pop('titles', [])
    if len(titles) != len(images):
        titles = list(titles) + [''] * (len(images) - len(titles))

    limits = kwargs.pop('limits', 'image')
    if limits == 'image':
        kwargs.setdefault('vmin', min(img.min() for img in images))
        kwargs.setdefault('vmax', max(img.max() for img in images))

    nrows, ncols = kwargs.get('shape', (1, len(images)))

    axes_off = kwargs.pop('axes_off', False)

    size = nrows * kwargs.pop('size', 5)
    width = size * len(images)
    if nrows > 1:
        width /= nrows * 1.33
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, size))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)
        ax.grid(False)
        if axes_off:
            ax.set_axis_off()


def get_asymmetry(lesion_region, debug=False):
    area_total = lesion_region.area
    img_mask = lesion_region.image
    horizontal_flip = np.fliplr(img_mask)
    diff_horizontal = img_mask * ~horizontal_flip

    vertical_flip = np.flipud(img_mask)
    diff_vertical = img_mask * ~vertical_flip

    diff_horizontal_area = np.count_nonzero(diff_horizontal)
    diff_vertical_area = np.count_nonzero(diff_vertical)
    asymm_idx = 0.5 * ((diff_horizontal_area / area_total) + (diff_vertical_area / area_total))
    ecc = lesion_region.eccentricity

    if debug:
        print(f'Diff area horizontal:{np.count_nonzero(diff_horizontal)}')
        print(f'Diff area vertical: {np.count_nonzero(diff_vertical)}')
        print(f'Asymmetric Index: {asymm_idx}')
        print(f'Eccentricity: {ecc}')

        imshow_all(img_mask, horizontal_flip, diff_horizontal,
                   titles=['image mask', 'horizontal flip', 'difference'], size=4, cmap='gray')
        imshow_all(img_mask, vertical_flip, diff_vertical,
                   titles=['image mask', 'vertical flip', 'difference'], size=4, cmap='gray')
        plt.show();
    return asymm_idx, ecc


def get_border_irregularity(lesion_region):
    area_total = lesion_region.area
    compact_index = (lesion_region.perimeter ** 2) / (4 * np.pi * area_total)
    return compact_index


def get_color_variegation(slice, debug):
    lesion_r = slice[:, :, 0]
    lesion_g = slice[:, :, 1]
    lesion_b = slice[:, :, 2]

    channel_red = np.std(lesion_r) / np.max(lesion_r)
    channel_green = np.std(lesion_g) / np.max(lesion_g)
    channel_blue = np.std(lesion_b) / np.max(lesion_b)

    if debug:
        print('\n-- COLOR VARIEGATION --')
        print(f'Red Std Deviation: {channel_red}')
        print(f'Green Std Deviation: {channel_green}')
        print(f'Blue Std Deviation: {channel_blue}')
        imshow_all(lesion_r, lesion_g, lesion_b)
        plt.show();
    return channel_red, channel_green, channel_blue


def get_texture(img):
    glcm = feature.greycomatrix(image=img, distances=[1],
                                angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 2],
                                symmetric=True, normed=True)

    correlation = np.mean(feature.greycoprops(glcm, prop='correlation'))
    homogeneity = np.mean(feature.greycoprops(glcm, prop='homogeneity'))
    energy = np.mean(feature.greycoprops(glcm, prop='energy'))
    contrast = np.mean(feature.greycoprops(glcm, prop='contrast'))

    return correlation, homogeneity, energy, contrast


def features_extraction(segmented_region_train, segmented_region_test,
                         train_set_path, val_set_path, debug=False):
    print('-- STARTING FEATURE EXTRACTION --')

    train = {}
    test = {}

    segmented_regions = {**segmented_region_train, **segmented_region_test}
    for idx, image_name in enumerate(segmented_regions):


        if image_name in segmented_region_train:
            image_path = train_set_path + image_name + '.jpg'
        elif image_name in segmented_region_test:
            image_path = val_set_path + image_name + '.jpg'
        else:
            print('Error: Cannot find {}'.format(image_name))
            return None

        image = io.imread(image_path)
        gray_img = color.rgb2gray(image)

        lesion_region = segmented_regions[image_name]

        # 1] ASYMMETRY

        asymm_idx, ecc = get_asymmetry(lesion_region, debug)

        # 2] Border irregularity:
        compact_index = get_border_irregularity(lesion_region)
        if debug:
            print('\n-- BORDER IRREGULARITY --')
            print(f'Compact Index: {compact_index}')

        # 3] Color variegation:
        channel_r, channel_green, channel_blue = get_color_variegation(image[lesion_region.slice], debug)

        # 4] Diameter:
        eq_diameter = lesion_region.equivalent_diameter
        if debug:
            print('\n-- DIAMETER --')
            print(f'Equivalent diameter: {eq_diameter}')
        # 5] Texture:
        correlation, homogeneity, energy, contrast = get_texture(img_as_ubyte(gray_img))
        if debug:
            print('\n-- TEXTURE --')
            print(f'Correlation: {correlation}')
            print(f'Homogeneity: {homogeneity}')
            print(f'Energy: {energy}')
            print(f'Contrast: {contrast}')

        if image_name in segmented_region_train:

            dataset = train
        elif image_name in segmented_region_test:
            dataset = test
        else:
            print(f'Error: Cannot find {image_name}')
            return None

        dataset[image_name] = [asymm_idx, ecc, compact_index, channel_r, channel_green, channel_blue,
                    eq_diameter, correlation, homogeneity, energy, contrast]

    return train, test
