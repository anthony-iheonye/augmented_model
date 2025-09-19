
"""
Classes for processing images and masks for visual attributes model training
"""

import os
import re
from typing import Tuple, Union, Optional, Dict, Type, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny
from skimage.filters import (gaussian, threshold_otsu)
from skimage.measure import label
from skimage.segmentation import find_boundaries

from utils import create_directory


class ImageAndMaskDatasetBuilder:

    def __init__(self,
                 images_directory: str,
                 masks_directory: str,
                 image_mask_channels: Tuple[int, int],
                 final_image_shape: Optional[Tuple[int, int]] = None,
                 crop_image_and_mask: bool = False,
                 crop_dimension: Optional[Tuple[int, int, int, int]] = None,
                 normalize_image: bool = False,
                 normalization_divisor: Union[int, float] = 255,
                 split_mask_into_channels: bool = False,
                 batch_size: Optional[int] = None,
                 shuffle_buffer_size: Optional[int] = None,
                 return_2d_mask: bool = False,
                 prefetch_data: Optional[bool] = None,
                 return_dict: bool = False,
                 image_key: str = 'image',
                 mask_key: str = 'mask',
                 cache_directory: Optional[str] = None,
                 ):
        """
        Builds a dataset made of images and their corresponding masks.

        :param images_directory: str - Directory containing images.
        :param masks_directory: str - Directory containing masks.
        :param image_mask_channels: (int, int) - The number of channels for the image and mask data.
        :param final_image_shape: (int, int) The final image shape.
        :param crop_image_and_mask: bool - Whether the image and mask should be cropped.
        :param crop_dimension: (int, int, int, int) A tuple (offset_height, offset_width, target_height, target_width).
        :param normalize_image: bool - Whether the image should be normalized.
        :param normalization_divisor: int or float - Value used to scale pixel intensity (e.g., 255 or 127.5).
        :param split_mask_into_channels: bool - Whether to split mask classes into channels.
        :param batch_size: int or None - Batch size. If None, no batching is applied.
        :param shuffle_buffer_size: int or None - Shuffle buffer size. If None, no shuffling is applied.
        :param prefetch_data: bool - Whether to prefetch data using tf.data.AUTOTUNE.
        :param cache_directory: str - Path to a directory for caching the dataset to disk. If `None`, caching is skipped. To cache the dataset in memory, set cache_directory to "".
        :param return_dict: If True, the image-mask pair will be returned as a dict. for example
            if image_key and mask_key is set to 'image' and 'mask', the dataset will return as
            {'image': tf.Tensor, 'mask': tf.Tensor}
        :param image_key: key for the image tensor when returned as a dict.
        :param mask_key: key for the mask tensor, when returned as a dict.
        """

        self.check_type(params_and_types={
            'images_directory': (images_directory, str, False),
            'masks_directory': (masks_directory, str, False),
            'image_mask_channels': (image_mask_channels, tuple, False),
            'final_image_shape': (final_image_shape, tuple, True),
            'crop_image_and_mask': (crop_image_and_mask, bool, False),
            'crop_dimension': (crop_dimension, tuple, True),
            'normalize_image': (normalize_image, bool, False),
            'normalization_divisor': (normalization_divisor, (int, float), False),
            'split_mask_into_channels': (split_mask_into_channels, bool, False),
            'batch_size': (batch_size, int, True),
            'shuffle_buffer_size': (shuffle_buffer_size, int, True),
            'return_2d_mask': (return_2d_mask, bool, False),
            'prefetch_data': (prefetch_data, bool, True),
            'return_dict': (return_dict, bool, False),
            'image_key': (image_key, str, False),
            'mask_key': (mask_key, str, False),
            'cache_directory': (cache_directory, str, True),
        })

        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.crop_dimension = crop_dimension
        self.crop_image_and_mask = crop_image_and_mask
        self.image_key = image_key
        self.mask_key = mask_key

        # Generate filepaths to the images and masks
        self.original_image_paths, self.original_mask_paths = self._get_sorted_filepaths_to_images_and_masks(
            images_directory, masks_directory)

        # Extract image and mask format
        img_path = self.original_image_paths[0]
        mask_path = self.original_mask_paths[0]
        self.image_format = self._get_image_format(image_path=img_path)
        self.mask_format = self._get_image_format(image_path=mask_path)
        # Choose the method for decoding images
        if self.image_format.lower() in ['.jpg', '.jpeg']:
            self.decode_image = tf.image.decode_jpeg
        elif self.image_format.lower() == '.png':
            self.decode_image = tf.image.decode_png
        elif self.image_format.lower() == '.bmp':
            self.decode_image = tf.image.decode_bmp
        else:
            self.decode_image = tf.image.decode_png

        # Choose the method for decoding masks
        if self.mask_format.lower() in ['.jpg', '.jpeg']:
            self.decode_mask = tf.image.decode_jpeg
        elif self.mask_format.lower() == '.png':
            self.decode_mask = tf.image.decode_png
        elif self.mask_format.lower() == '.bmp':
            self.decode_mask = tf.image.decode_bmp
        else:
            self.decode_mask = tf.image.decode_png

        # set the number of channels for the image and mask
        self.image_mask_channels = image_mask_channels
        self.image_channels = self.image_mask_channels[0]
        self.mask_channels = self.image_mask_channels[1]
        self.return_2d_mask = return_2d_mask
        self.return_dict = return_dict

        # set the initial shape of the images and masks.
        self.image_shape, self.mask_shape = self._get_image_and_mask_shape(image_path=img_path,
                                                                           mask_path=mask_path)

        # Check if images and masks are to be resized and/or cropped.
        self.resize_images = False

        if crop_dimension is not None and crop_image_and_mask:
            self.offset_height = crop_dimension[0]
            self.offset_width = crop_dimension[1]
            self.target_height = crop_dimension[2]
            self.target_width = crop_dimension[3]
        else:
            self.crop_image_and_mask = False

        if final_image_shape is not None:
            self.final_image_shape = final_image_shape + (self.image_channels,)
            self.final_mask_shape = final_image_shape + (self.mask_channels,)
            self.new_image_height = tuple(self.final_image_shape)[0]
            self.new_image_width = tuple(self.final_image_shape)[1]

            if self.crop_image_and_mask:
                if self.target_height != self.new_image_height or self.target_width != self.new_image_width:
                    self.resize_images = True
            else:
                if self.new_image_height != self.image_shape[0] or self.new_image_width != self.image_shape[1]:
                    self.resize_images = True


        elif final_image_shape is None and self.crop_image_and_mask and crop_dimension is not None:
            self.final_image_shape = (self.target_height, self.target_width, self.image_channels)
            self.final_mask_shape = (self.target_height, self.target_width, self.mask_channels)
            self.new_image_height = self.target_height
            self.new_image_width = self.target_width

        else:
            self.final_image_shape = self.image_shape
            self.final_mask_shape = self.mask_shape
            self.new_image_height = self.image_shape[0]
            self.new_image_width = self.image_shape[1]

        self.image_mask_dataset = None
        self.tune = tf.data.experimental.AUTOTUNE
        self.unique_intensities = None

        # Generate class labels for the segmentation masks
        self.get_mask_class_labels()
        self.lookup_table = self.build_lookup_table()
        self.split_mask_into_channels = split_mask_into_channels

        self.normalize_image = normalize_image
        self.normalization_divisor = normalization_divisor

        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_data = prefetch_data

        # Set cache directory
        if isinstance(cache_directory, str):
            if cache_directory != '':
                self.cache_directory = create_directory(dir_name=cache_directory,
                                                        return_dir=True,
                                                        overwrite_if_existing=True)
            else:
                self.cache_directory = ''
        else:
            self.cache_directory = None


    @staticmethod
    def check_type(params_and_types: Dict[str, Tuple[Any, Union[Type, Tuple[Type, ...]], bool]]):
        """
        Validates that each parameter value matches its expected type, and handles optional parameters.

        :param params_and_types: A dictionary where each key is a parameter name, and the value is a tuple:
            (value, expected_type(s), is_optional), where:
                - value: The actual value to validate.
                - expected_type(s): A type or tuple of types that the value is expected to match.
                - is_optional: If True, allows the value to be None.

            Example:
                check_type({
                    "name": ("Alice", str, False),
                    "age": (None, int, True),
                    "scores": ([90, 95], list, False),
                })

        :raises TypeError:
            - If a value is None and is_optional is False.
            - If a value does not match the expected type(s).
        """
        for param_name, (value, expected_types, is_optional) in params_and_types.items():
            if value is None:
                if is_optional:
                    continue
                else:
                    raise TypeError(f"{param_name} must not be None.")

            if not isinstance(value, expected_types):
                allowed_types = (
                    expected_types.__name__
                    if isinstance(expected_types, type)
                    else ', '.join(t.__name__ for t in expected_types)
                )
                raise TypeError(f"{param_name} must be of type {allowed_types}. Got {type(value).__name__}.")

    @staticmethod
    def sort_filenames(file_paths):
        return sorted(file_paths, key=lambda var: [
            int(x) if x.isdigit() else x.lower() for x in re.findall(r'\D+|\d+', var)
        ])

    def _get_sorted_filepaths_to_images_and_masks(self, images_dir, masks_dir):
        """
        Generates the two lists containing sorted paths of images and masks, respectively.

        :param images_dir: Directory containing image files.
        :param masks_dir: Directory containing mask files.
        :return: Two lists – paths to image files and corresponding mask files.

        """
        image_file_list = os.listdir(path=images_dir)
        mask_file_list = os.listdir(path=masks_dir)
        image_paths = [os.path.join(images_dir, filename) for filename in image_file_list]
        mask_paths = [os.path.join(masks_dir, filename) for filename in mask_file_list]

        # sort the file paths in ascending other
        image_paths = self.sort_filenames(image_paths)
        mask_paths = self.sort_filenames(mask_paths)

        return image_paths, mask_paths

    @staticmethod
    def _get_image_format(image_path):
        return os.path.splitext(image_path)[-1]

    def _get_image_and_mask_shape(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = self.decode_image(image, channels=self.image_channels)

        mask = tf.io.read_file(mask_path)
        mask = self.decode_mask(mask, channels=self.mask_channels)

        return image.shape, mask.shape

    def _set_original_shape(self, image, mask):
        """
        Sets width and height information to the image and mask tensors.
        """
        image.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)
        return image, mask

    def _set_final_shape(self, image, mask):
        """
        Sets width and height information to the image and mask tensors.
        """
        image.set_shape(self.final_image_shape)
        mask.set_shape(self.final_mask_shape)
        return image, mask

    def _read_and_decode_image_and_mask(self, image_path: str, mask_path: str):
        """
        Reads and decodes and image and its corresponding masks.
        :param image_path: (str) The image's filepath
        :param mask_path: (str) The mask's filepath
        :return: (tensors) Image and corresponding mask
        """
        # Read image and mask
        image = tf.io.read_file(image_path)
        mask = tf.io.read_file(mask_path)

        image = self.decode_image(contents=image, channels=self.image_channels)
        mask = self.decode_image(contents=mask, channels=self.mask_channels)
        image, mask = self._set_original_shape(image, mask)
        return image, mask

    def _read_and_decode_mask(self, mask_path: str):
        """
        Reads and decodes a segmentation mask.

        :param mask_path: (str) Path to the segmentation mask file.
        :return: Decoded TF tensor representing the mask.
        """

        # Read image and mask
        mask = tf.io.read_file(mask_path)

        mask = self.decode_mask(contents=mask, channels=self.mask_channels)
        # mask.set_shape(self.mask_shape)
        return mask

    @staticmethod
    def _cast_image_mask_to_uint8(image, mask):
        image = tf.cast(image, tf.uint8)
        mask = tf.cast(mask, tf.uint8)
        return image, mask

    @staticmethod
    def _cast_image_mask_to_float(image, mask):
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return image, mask

    @staticmethod
    def _denormalize_image_mask_to_0_255(image, mask):
        """convert image and mask to uint8. The values in the image and mask are scaled between 0 and 255."""
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.uint8)
        return image, mask

    @staticmethod
    def _normalize_image_mask_to_0_1(image, mask):
        """convert image and mask to float32. The values in the image and mask are scaled between 0 and 1."""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
        return image, mask

    @staticmethod
    def _normalize_image_to_0_1(image, mask):
        """convert image to float32. The values in the image are scaled between 0 and 1."""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, mask

    def _crop_image_and_mask(self, image, mask):
        """Crops out a portion of the image and mask."""
        # crop image and mask
        if self.crop_image_and_mask and self.crop_dimension is not None:
            image = tf.image.crop_to_bounding_box(image, self.offset_height, self.offset_width,
                                                  self.target_height, self.target_width)

            # mask = tf.expand_dims(mask, axis=-1) if len(mask.shape) == 2 else mask

            mask = tf.image.crop_to_bounding_box(mask, self.offset_height, self.offset_width,
                                                 self.target_height, self.target_width)
        return image, mask

    def _resize_image_and_mask(self, image, mask):
        """Resize the image and mask to the predefined dimension."""
        if self.resize_images:
            image = tf.expand_dims(image, axis=-1) if image.ndim == 2 else image
            image = tf.image.resize(images=image, size=(self.new_image_height, self.new_image_width),
                                    method='bilinear')
            image = tf.reshape(tensor=image, shape=(self.new_image_height, self.new_image_width, self.image_channels))

            mask = tf.expand_dims(mask, axis=-1) if mask.ndim == 2 else mask
            mask = tf.image.resize(images=mask, size=(self.new_image_height, self.new_image_width),
                                   method='nearest')
            mask = tf.reshape(tensor=mask, shape=(self.new_image_height, self.new_image_width, self.mask_channels))

            # mask = tf.reshape(tensor=mask, shape=(self.new_image_height, self.new_image_width))

            # The resize operation returns image & mask in float values (eg. 125.2, 233. 4),
            # before augmentation, these pixel values need to be normalized to the range [0 - 1],
            # because the tensorflow.keras augmentation layer only accept values in the normalize range of [0 - 1]. To ensure we correctly normalize , we will first
            # round up the current float pixel intensities to whole numbers using tf.cast(image, tf.uint8).
            image, mask = self._cast_image_mask_to_uint8(image, mask)

        if mask.ndim == 3 and self.return_2d_mask and not self.split_mask_into_channels:
            if mask.shape[-1] == 1:
                mask = tf.reshape(tensor=mask, shape=(self.new_image_height, self.new_image_width))
            else:
                mask = tf.math.reduce_sum(mask, axis=-1)

        return image, mask

    def get_mask_unique_intensities(self, mask_path):
        """Return the unique pixel intensity for an image."""
        mask = self._read_and_decode_mask(mask_path=mask_path)
        return np.unique(mask).tolist()

    def get_mask_class_labels(self):
        """Compute the unique pixel intensities on all the masks in the dataset."""
        # check the unique intensities of the first fifty or 10% of the image (the larger number)
        num = max(50, (len(self.original_mask_paths) // 10) + 1)
        img_paths = self.original_mask_paths[:num]

        unique_intensities = set()

        for path in img_paths:
            unique_intensities.update(self.get_mask_unique_intensities(path))
        self.unique_intensities = list(unique_intensities)

    def build_lookup_table(self):
        unique_intensities = tf.constant(self.unique_intensities)

        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=unique_intensities,
                values=tf.range(len(unique_intensities), dtype=tf.int32)
            ),
            default_value=-1    # Optional: helps catch invalid classes
        )
        return table

    def map_class_to_contiguous_indices(self, image, mask):
        mask = tf.cast(mask, dtype=tf.int32)
        return image, self.lookup_table.lookup(keys=mask)

    def _split_mask_into_channels(self, image, mask):
        """Split mask into channels."""

        # mask = tf.cast(mask, dtype=tf.uint8)
        # stack_list = []
        #
        # # For each class intensity, generate a binary channel showing pixels belonging to that class
        # for intensity in self.unique_intensities:
        #     # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
        #     # that have the same pixel intensity as  the integer 'class_index'. we want to
        #     temp_mask = tf.equal(mask[..., 0], tf.constant(intensity, dtype=tf.uint8))
        #     # add each temporary mask to the stack_list.
        #     stack_list.append(tf.cast(temp_mask, dtype=tf.uint8))
        #
        # # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        # mask = tf.stack(stack_list, axis=-1)  # Axis starts from 0, so axis of 2 represents the third axis
        if mask.shape[-1] == 1:
            mask = tf.squeeze(mask, axis=-1)
        mask = tf.one_hot(indices=mask, depth=len(self.unique_intensities), dtype=tf.float32)
        return image, mask

    def _normalize_image(self, image, mask):
        """Normalize the image using the divisor.
        If divisor == 255, scales to [0, 1]; otherwise to [-1, 1]."""
        if self.normalize_image:
            image = tf.cast(image, dtype=tf.float32)
            if self.normalization_divisor != 255:
                image = (image - self.normalization_divisor) / self.normalization_divisor
            else:
                image /= 255.0

        return image, mask

    def _read_crop_resize_image_and_mask(self, image_path: str, mask_path: str):
        """
        Read, crop and resize image and corresponding mask.

        :param image_path: (str) path to an Image
        :param mask_path: (str) path to the image's segmentation mask
        :return: (image_tensor, mask_tensor) after preprocessing. If enabled, mask will be split into class channels..
        """
        image, mask = self._read_and_decode_image_and_mask(image_path=image_path, mask_path=mask_path)
        image, mask = self._crop_image_and_mask(image=image, mask=mask)
        image, mask = self._resize_image_and_mask(image=image, mask=mask)
        image, mask = self._normalize_image(image=image, mask=mask)
        image, mask = self.map_class_to_contiguous_indices(image=image, mask=mask)

        if self.split_mask_into_channels:
            image, mask = self._split_mask_into_channels(image=image, mask=mask)
        return image, mask

    def _get_dataset(self, image_paths, mask_paths):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self._read_crop_resize_image_and_mask, num_parallel_calls=self.tune)

        if self.return_dict:
            dataset = dataset.map(lambda image, mask: {self.image_key: image, self.mask_key: mask})
        dataset = dataset.take(count=-1)
        if self.cache_directory is not None:
            dataset = dataset.cache(filename=self.cache_directory)

        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.batch_size:
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)

        if self.prefetch_data:
            dataset = dataset.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.image_mask_dataset = dataset

    def run(self):
        self._get_dataset(self.original_image_paths, self.original_mask_paths)


class ImageMaskVisualDatasetBuilderOld:

    def __init__(self,
                 train_directory: Optional[str] = None,
                 validation_directory: Optional[str] = None,
                 test_directory: Optional[str] = None,
                 final_image_shape: Tuple[int, int] = (1024, 1024),
                 apply_weight_maps: bool = True,
                 use_log_weights: bool = False,
                 apply_unet_edge_weights: bool = False,
                 image_mask_channels: Tuple[int, int] = (3, 1),
                 class_names: tuple = ('background', 'pea', 'outline'),
                 batch_size: Optional[int] = 2,
                 normalize_using_255: bool = True,
                 crop_image_and_mask: bool = False,
                 crop_dimension: Optional[Tuple[int, int, int, int]] = None,
                 w0: int = 10,
                 sigma: int = 5,
                 shuffle: bool = True,
                 train_shuffle_size: Optional[int] = 100,
                 shuffle_validation_data=False,
                 validation_shuffle_size: Optional[int] =100,
                 prefetch_data: bool = True,
                 cache_directory: Optional[str]=None,
                 overwrite_existing_cache_directory: bool = True,
                 include_visual_attributes: bool = True,
                 include_visual_attribute_weights: bool = True,
                 visual_attributes: tuple = ('L', 'a', 'b', 'contrast', 'correlation', 'energy', 'entropy',
                                             'homogeneity', 'uniformity', 'equivalent_diameter', 'eccentricity',
                                             'feret_diameter_max', 'filled_area', 'perimeter', 'roundness'),
                 train_visual_attributes_json_path: Optional[str] = None,
                 val_visual_attributes_json_path: Optional[str] = None,
                 test_visual_attributes_json_path: Optional[str] = None,
                 visual_attributes_stats_save_dir: Optional[str] = None):
        """
        Builds a TensorFlow dataset for semantic segmentation tasks enriched with image-mask pairs, weight maps, and object-level visual attributes.

        This dataset builder prepares training, validation, and test datasets by reading RGB images
        and corresponding segmentation masks. It includes preprocessing steps such as cropping,
        resizing, normalization, optional edge detection, weight map generation using the UNet
        weighting method, and integration of auxiliary visual attributes (e.g., color and texture
        features) for multi-target learning.

        If `class_names` includes 'outline' (e.g., ('background', 'pea', 'outline')), the class
        will extract and label object edges (e.g., pea outlines) in the final mask. If only
        'background' and 'pea' are provided, outlines will be omitted.

        :param train_directory: (str)  Path to the directory containing the training dataset.
            It must include two subdirectories named 'images' and 'masks', containing the
            input images and corresponding masks, respectively.
        :param validation_directory: (str) Path to the directory containing the validation dataset.
            It must include two subdirectories named 'images' and 'masks', containing the
            input images and corresponding masks, respectively.
        :param test_directory: (str) Path to the directory containing the test dataset.
            It must include two subdirectories named 'images' and 'masks', containing the
            input images and corresponding masks, respectively.
        :param final_image_shape: (Tuple[int, int]) Target shape to resize images and masks to.
        :param apply_weight_maps: (bool) Whether to generate weight maps for each mask.
        :param use_log_weights: (bool) Whether to compute class weights using log scaling.
        :param apply_unet_edge_weights: (bool) Whether to apply UNet-style edge-aware weighting.
        :param image_mask_channels: (Tuple[int, int]) Number of channels for image and mask respectively.
        :param class_names: (Tuple[str]) Names of segmentation classes (e.g., 'background', 'pea').
        :param batch_size: (int) Number of samples per training/validation/test batch.
        :param normalize_using_255: (bool) Normalize images to [0, 1] if True, else to [-1, 1].
        :param crop_image_and_mask: (bool) Whether to crop the image and mask before resizing.
        :param crop_dimension: (Tuple[int, int, int, int]) Crop window in the format
            (offset_height, offset_width, target_height, target_width).
        :param w0: (int) UNet edge-weighting parameter to control influence of object boundaries.
        :param sigma: (int) UNet edge-weighting Gaussian smoothing parameter.
        :param shuffle: (bool) Whether to shuffle training data before batching.
        :param train_shuffle_size: (int) Size of buffer used when shuffling the training data.
        :param shuffle_validation_data: (bool) Whether to shuffle validation data before batching.
        :param validation_shuffle_size: (int) Size of buffer used when shuffling validation data.
        :param prefetch_data: (bool) Whether to prefetch data for improved I/O performance.
        :param cache_directory: (str) Directory to cache the preprocessed datasets.
        :param overwrite_existing_cache_directory: (bool) Whether to overwrite existing cached datasets.
        :param include_visual_attributes: (bool) Whether to include object-level visual attributes.
        :param include_visual_attribute_weights: (bool) Whether to compute weights for visual attributes.
        :param visual_attributes: (Tuple[str]) List of visual attributes to include
            (e.g., 'L', 'a', 'b', 'contrast', 'eccentricity', etc.).
        :param train_visual_attributes_json_path: (str) Path to JSON file containing training set visual attributes.
        :param val_visual_attributes_json_path: (str) Path to JSON file containing validation set visual attributes.
        :param test_visual_attributes_json_path: (str) Path to JSON file containing test set visual attributes.
        :param visual_attributes_stats_save_dir: (str) Directory to save summary statistics (mean, std, etc.) for normalization.
        """

        # Generate unprocess training set
        train_dataset_builder = ImageAndMaskDatasetBuilder(
            images_directory=f"{train_directory}/images",
            masks_directory=f"{train_directory}/masks",
            image_mask_channels=image_mask_channels,
            final_image_shape=final_image_shape,
            crop_image_and_mask=crop_image_and_mask,
            crop_dimension=crop_dimension,
            return_2d_mask=True
        )
        train_dataset_builder.run()
        self.training_dataset = train_dataset_builder.image_mask_dataset


        # Generate unprocess validation set
        val_dataset_builder = ImageAndMaskDatasetBuilder(
            images_directory=f"{validation_directory}/images",
            masks_directory=f"{validation_directory}/masks",
            image_mask_channels=image_mask_channels,
            final_image_shape=final_image_shape,
            crop_image_and_mask=crop_image_and_mask,
            crop_dimension=crop_dimension,
            return_2d_mask=True
        )
        val_dataset_builder.run()
        self.validation_dataset = val_dataset_builder.image_mask_dataset


        # Generate unprocess testing set
        test_dataset_builder = ImageAndMaskDatasetBuilder(
            images_directory=f"{test_directory}/images",
            masks_directory=f"{test_directory}/masks",
            image_mask_channels=image_mask_channels,
            final_image_shape=final_image_shape,
            crop_image_and_mask=crop_image_and_mask,
            crop_dimension=crop_dimension,
            return_2d_mask=True
        )
        test_dataset_builder.run()
        self.test_dataset = test_dataset_builder.image_mask_dataset


        self.apply_unet_edge_weights = apply_unet_edge_weights
        self.test_directory = test_directory
        self.validation_directory = validation_directory
        self.train_directory = train_directory

        self.original_image_paths = []
        self.original_mask_paths = []
        # 'no_pea_threshold' value that would be returned as threshold when the mask contains not infocus pea. The
        # threshold would be used to separate the foreground (infocus peas) from the background.
        self.no_pea_thresh = 400.0


        self.use_log_weights = use_log_weights
        self.class_names = class_names

        self.image_size = train_dataset_builder.final_image_shape
        self.height = self.image_size[0]
        self.width = self.image_size[1]

        self.apply_weight_maps = apply_weight_maps
        self.image_mask_channels = image_mask_channels
        self.crop_image_and_mask = crop_image_and_mask
        self.crop_dimension = crop_dimension
        self.w0 = w0
        self.sigma = sigma
        self.normalize_using_255 = normalize_using_255

        self.shuffle = shuffle
        self.train_shuffle_size = train_shuffle_size
        self.validation_shuffle_size = validation_shuffle_size
        self.shuffle_validation_data = shuffle_validation_data

        self.batch_size = batch_size

        self.number_of_classes = len(self.class_names)
        self.detect_outline = True if self.number_of_classes == 3 else False


        self.tune = tf.data.experimental.AUTOTUNE
        self.cache_directory = cache_directory
        self.prefetch_data = prefetch_data

        # Cache directories
        self.overwrite_cache_dir = overwrite_existing_cache_directory
        if isinstance(cache_directory, str):
            if cache_directory != '':
                self.train_cache = create_directory(os.path.join(self.cache_directory, 'train'),
                                                    return_dir=True,
                                                    overwrite_if_existing=self.overwrite_cache_dir)
                self.val_cache = create_directory(os.path.join(self.cache_directory, 'val'),
                                                  return_dir=True,
                                                  overwrite_if_existing=self.overwrite_cache_dir)
                self.test_cache = create_directory(os.path.join(self.cache_directory, 'test'),
                                                   return_dir=True,
                                                   overwrite_if_existing=self.overwrite_cache_dir)
            else:
                self.train_cache = self.val_cache = self.test_cache = ''
        else:
            self.train_cache = self.val_cache = self.test_cache = None


        if train_visual_attributes_json_path is not None:
            self.training_visual_props_json_path = train_visual_attributes_json_path
        else:
            self.training_visual_props_json_path = None

        if val_visual_attributes_json_path is not None:
            self.validation_visual_props_json_path = val_visual_attributes_json_path
        else:
            self.validation_visual_props_json_path = None

        if test_visual_attributes_json_path is not None:
            self.test_visual_props_json_path = test_visual_attributes_json_path
        else:
            self.test_visual_props_json_path = None

        self.visual_attributes_stats_save_dir = visual_attributes_stats_save_dir
        self.training_visual_attributes_json_path = train_visual_attributes_json_path
        self.validation_visual_attributes_json_path = val_visual_attributes_json_path
        self.test_visual_attributes_json_path = test_visual_attributes_json_path

        self.include_visual_attributes = include_visual_attributes
        self.include_visual_attribute_weights = include_visual_attribute_weights

        if self.include_visual_attributes:
            self.visual_attributes = visual_attributes
            self.visual_props_processor = VisualAttributesDatasetCreator(
                train_visual_attributes_json_path=self.training_visual_attributes_json_path,
                val_visual_attributes_json_path=self.validation_visual_attributes_json_path,
                test_visual_attributes_json_path=self.test_visual_attributes_json_path,
                visual_properties=self.visual_attributes)

            self.visual_props_processor.process_data()
            self.train_unnorm_visual_props_dataframe = self.visual_props_processor.train_unnorm_visual_props_dataframe
            self.val_unnorm_visual_props_dataframe = self.visual_props_processor.val_unnorm_visual_props_dataframe
            self.test_unnorm_visual_props_dataframe = self.visual_props_processor.test_unnorm_visual_props_dataframe
            self.train_visual_props_weights_dataframe = self.visual_props_processor.train_visual_props_weights_dataframe
            self.train_visual_props_weights_dataset = self.visual_props_processor.train_visual_props_weights_dataset

            self.train_normalized_visual_props_dataframe = self.visual_props_processor.train_normalized_visual_props_df
            self.val_normalized_visual_props_dataframe = self.visual_props_processor.val_normalized_visual_props_df
            self.test_normalized_visual_props_df = self.visual_props_processor.test_normalized_visual_props_df

            self.training_visual_props_dataset = self.visual_props_processor.train_visual_props_dataset
            self.validation_visual_props_dataset = self.visual_props_processor.val_visual_props_dataset
            self.test_visual_props_dataset = self.visual_props_processor.test_visual_props_dataset
        else:
            self.visual_attributes = None


    def process_data(self):
        """Reads the images and masks from their stored location, then prepares a tensorflow training, validation and
         test dataset."""

        # produce the training validation and test datasets
        self._produce_train_val_test_datasets()

    @staticmethod
    def _apply_gaussian_filter(mask):
        """De-noises the image using a gaussian filter.

        :param mask:  (numpy array) A numpy array of the mask
        :return: (numpy array) De-noised/filtered mask
        """
        mask_smooth = gaussian(mask)
        return mask_smooth

    def _tf_apply_gaussian_filter(self, image, mask):
        """De-noises the mask using a Gaussian filter"""
        mask_shape = mask.shape
        image_shape = image.shape

        [mask_smooth, ] = tf.py_function(self._apply_gaussian_filter, [mask], [tf.float64])

        mask_smooth.set_shape(mask_shape)
        image.set_shape(image_shape)
        return image, mask_smooth

    def _compute_otsu_threshold(self, mask_smooth):
        """
        Return threshold value (for detecting the infocus peas) using the Otsu's method.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :return: (float) Upper threshold value. All pixels with an intensity higher
            than this value are assumed to be the object of interest (in-focus pea)
        """
        mask_smooth = mask_smooth.numpy()

        # confirm if there is an in-focus pea in the image. If the maximum pixel intensity if
        # greater than zero, there is an infocus pea, else there isn't.
        if mask_smooth.max() == 0:
            threshold = self.no_pea_thresh
        else:
            threshold = threshold_otsu(mask_smooth)
        return threshold

    def _tf_compute_otsu_threshold(self, image, mask_smooth):
        """
        Return threshold value based on Otsu's method.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :return: (float) Upper threshold value. All pixels with an intensity higher
            than this value are assumed to be the object of interest (in-focus pea)
        """

        image_shape = image.shape
        mask_shape = mask_smooth.shape

        threshold = tf.py_function(self._compute_otsu_threshold, [mask_smooth], tf.float64)

        threshold.set_shape(shape=())
        image.set_shape(image_shape)
        mask_smooth.set_shape(mask_shape)
        return image, mask_smooth, threshold

    @staticmethod
    def _detect_regions_containing_peas(mask_smooth, threshold):
        """
        Produce a mask representing the pixel locations occupied by the pea.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :param threshold: (float) The Otsu's threshold
        :return: (tensor) Mask of in-focus peas alone, without the edges
        """
        mask_smooth = mask_smooth.numpy()
        threshold = threshold.numpy()

        if threshold == 300:
            pea_mask = np.where(mask_smooth == 0, False, True)
            temp_mask = np.where(mask_smooth == 0, 0, 1)
        else:
            # Region containing in-focus peas
            pea_mask = np.where(mask_smooth > threshold, True, False)
            temp_mask = np.where(mask_smooth > threshold, 1, 0)

        # convert the pea mask to float, so the canny edge diagram can be produced.
        temp_mask = np.asarray(temp_mask, dtype=np.float64)
        return pea_mask, temp_mask, threshold

    def _tf_detect_regions_containing_peas(self, image, mask_smooth, threshold):
        """
        Produce a mask representing the pixel locations occupied by the pea.

        :param image: (Tensor) RGB image of all the peas
        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :param threshold: (float) The Otsu's threshold
        :return: (tensor) Mask of in-focus peas alone, without the edges
        """
        mask_shape = mask_smooth.shape
        image_shape = image.shape

        [pea_mask, temp_mask, threshold] = tf.py_function(self._detect_regions_containing_peas,
                                                          [mask_smooth, threshold],
                                                          [tf.bool, tf.float64, tf.float64])

        temp_mask.set_shape(shape=mask_shape)
        pea_mask.set_shape(shape=mask_shape)
        image.set_shape(image_shape)
        threshold.set_shape(shape=())

        return image, pea_mask, temp_mask, threshold

    def _produce_whole_peas_without_outline(self, mask_smooth, threshold):
        """
        Produce a mask representing the pixel locations occupied by the pea.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :param threshold: (float) The Otsu's threshold
        :return: (tensor) Mask of in-focus peas alone, without the edges
        """
        mask_smooth = mask_smooth.numpy()
        threshold = threshold.numpy()

        if threshold == self.no_pea_thresh:
            pea = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)
        else:
            # Region containing in-focus peas
            pea_mask = np.where(mask_smooth > threshold, True, False)

            # Add the pea body to the blank mask that already contains the edge
            pea = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)

            pea[pea_mask] = 1

        # convert the pea image to float, so the canny edge diagram can be produced.
        pea = tf.convert_to_tensor(pea, dtype=tf.float32)
        return pea, pea, threshold

    def _tf_produce_whole_peas_without_outline(self, image, mask_smooth, threshold):
        """Produce a mask representing the pixel locations occupied by the pea."""
        mask_shape = (self.height, self.width, 1)
        image_shape = image.shape

        [pea, temp_mask, threshold] = tf.py_function(self._produce_whole_peas_without_outline, [mask_smooth, threshold],
                                                     [tf.float32, tf.float32, tf.float64])
        temp_mask.set_shape(shape=mask_shape)
        pea.set_shape(shape=mask_shape)
        image.set_shape(image_shape)
        threshold.set_shape(shape=())
        return image, pea, temp_mask, threshold

    @staticmethod
    def _produce_canny_edges(temp_mask, threshold):
        """
        Produce the edges of the peas using the canny edge detector

        :param temp_mask: (Tensor) Mask of the in-focus peas
        :return: (Tensor) Canny edge diagram of the peas
        """
        temp_mask = temp_mask.numpy()
        threshold = threshold.numpy()

        if threshold == 0:
            pea_outline_mask = np.zeros_like(a=temp_mask, dtype=np.bool)
        else:
            pea_outline_mask = canny(temp_mask)
        return pea_outline_mask

    def _tf_produce_canny_edge(self, image, pea_mask, temp_mask, threshold):
        """
        Produces the edges of the peas using the canny edge detector

        :param image: (Tensor) RGB image of all the peas
        :param pea_mask:  (Tensor) Mask of the in-focus peas
        :param temp_mask: (Tensor) Mask of the in-focus peas. temp_mask would be used to obtain
            the canny edge diagram of the in-focus peas
        :return: (Tensor) Canny edge diagram of the peas
        """
        mask_shape = pea_mask.shape
        [pea_outline_mask, ] = tf.py_function(self._produce_canny_edges, [temp_mask, threshold], [tf.bool])
        pea_outline_mask.set_shape(shape=mask_shape)
        return image, pea_mask, pea_outline_mask

    def _produce_final_mask(self, pea_mask, pea_outline_mask):
        """
        Produce the overall mask, made up pixel values  representing of the body of the in-focus
        peas, their exterior outline (edges), and the background. The pixel intensity for each class
        is labeled as follows:
            - background of image:  0
            - pea outline:  1
            - peas body: 2
        :param pea_mask: (Tensor) Mask of the in-focus peas
        :param pea_outline_mask: (Tensor) Canny edge diagram of the infocus peas
        :return: mask of pea, their outline and background all together.
        """
        pea_outline_mask = pea_outline_mask.numpy()
        pea_mask = pea_mask.numpy()

        # Add the pea body to the blank mask that already contains the edge
        new_mask = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)

        new_mask[pea_mask] = 1

        new_mask[pea_outline_mask] = 2
        new_mask = tf.convert_to_tensor(new_mask, dtype=tf.float32)

        return new_mask

    def _tf_produce_final_mask(self, image, pea_mask, pea_outline_mask):
        mask_shape = pea_mask.shape
        [new_mask, ] = tf.py_function(self._produce_final_mask, [pea_mask, pea_outline_mask], [tf.float32])
        new_mask.set_shape(mask_shape + (1,))
        return image, new_mask

    def _prepare_initial_dataset(self, image: tf.Tensor, mask: tf.Tensor):
        """
        Reads and decodes and image and its corresponding masks.
        This function also assigns labels to the the background of the image, the pea's outline and its body. The labels
        are as follows:
        - background of image:  0
        - peas body: 1
        - pea outline:  2

        :return: (Tensors) Two TF Tensors - one for the rgb image, and the other
            for the mask, with the pixel locations properly labelled as background (1)
            pea's body (1), or pea outline (3).
        """
        # image, mask = self._tf_read_and_decode_image_and_mask(image_path=image_path, mask_path=mask_path)
        # image, mask = self._crop_image_and_mask(image=image, mask=mask)
        # image, mask = self._resize_image_and_mask(image=image, mask=mask) if self.resize_image else (image, mask)
        image, mask = self._tf_apply_gaussian_filter(image=image, mask=mask)
        image, mask, threshold = self._tf_compute_otsu_threshold(image=image, mask_smooth=mask)

        image, mask, temp_mask, threshold = self._tf_detect_regions_containing_peas(image=image, mask_smooth=mask,
                                                                                    threshold=threshold) if \
            self.detect_outline else self._tf_produce_whole_peas_without_outline(image=image, mask_smooth=mask,
                                                                                 threshold=threshold)

        image, mask, edge_mask = self._tf_produce_canny_edge(image=image, pea_mask=mask,
                                                             temp_mask=temp_mask, threshold=threshold) if \
            self.detect_outline else (image, mask, temp_mask)
        image, mask = self._tf_produce_final_mask(image, mask, edge_mask) if self.detect_outline else (image, mask)
        return image, mask

    # STEP-2
    # Functions for label channel addition and image normalization
    def _create_weight_map(self, mask):
        """
        Creates a weight map of a mask, according to the Unet method
        :param mask: (numpy array) The mask of the infocus peas
        :return: (numpy array) The weight map of of the infocus peas (and their outline, if outlines are used)
        """
        # The shape of the mask is in the form [height, width, 1], but we need a 2-D array, hence, we
        # extract the first layer
        pea_mask = mask[..., 0]

        # If the mask contains no infocus peas, there is no need to continue with the computations below, instead,
        # return a weight map containing ones, so that when the predicted mask is multiplied by the weight map of ones,
        # the predicted mask does not changed.
        if pea_mask.numpy().max() == 0:
            weight_map = tf.ones(shape=(self.height, self.width, 1), dtype=tf.float64)
            return weight_map
        else:
            # label each pea with integer values
            labelled_pea_mask = label(pea_mask)

            # obtain the regions on the mask containing infocus peas, and the region occupied by the background
            pea_binary_mask = labelled_pea_mask > 0
            background = (labelled_pea_mask == 0)

            # compute the number of peas within the mask
            no_of_peas = (len(np.unique(labelled_pea_mask)) // 2) - 1 if self.apply_unet_edge_weights else \
                len(np.unique(labelled_pea_mask)) - 1

            # check if more than one peas is contained in the mask
            multiple_peas_exist = (no_of_peas > 1)

            if self.apply_unet_edge_weights and multiple_peas_exist:
                # locate the inner segmentation boundaries of the peas within the mask
                pea_boundaries = find_boundaries(label_img=labelled_pea_mask, mode='inner')

                # obtain the body of the pea, within the pea's inner boundary
                pea_body_excluding_the_boundary = np.bitwise_xor(pea_binary_mask, pea_boundaries)

                # compute the class weight for the foreground and background, based on the area occupied by each class.
                # Since the foreground occupies less area, it should be given more weight so that its loss would be
                # penalized more. By so doing, the model would identity the foreground more accurately.
                total_foreground_pixels = np.sum(pea_body_excluding_the_boundary, axis=(0, 1))
                total_background_pixels = np.sum(background, axis=(0, 1))
                total_foreground_and_background_pixels = pea_mask.numpy().size

                if self.use_log_weights:
                    background_weight = np.log10(total_foreground_and_background_pixels / total_background_pixels)

                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = np.log10(total_foreground_and_background_pixels / total_foreground_pixels)

                else:
                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = 1 - total_foreground_pixels / total_foreground_and_background_pixels
                    background_weight = 1 - foreground_weight

                # produce a list containing the unique label id for each infocus pea in the mask
                pea_ids = [x for x in np.unique(labelled_pea_mask) if x > 0]

                # produce a 3-D array to store the euclidean distances for each of the infocus pea. Each
                # layer in the third axis would be used to store the distance a single pea and other peas.
                euclidean_distances = np.zeros(shape=(self.height, self.width, len(pea_ids)))

                # compute the Euclidean distance of individual peas, starting from a particular pea that has a given
                # pea_id
                for i, pea_id in enumerate(pea_ids):
                    euclidean_distances[..., i] = distance_transform_edt(labelled_pea_mask != pea_id)

                # Sort the Euclidean distance so we could choose the least two for each pea (connected component)
                euclidean_distances.sort(axis=-1)

                # Select the smallest and second to smallest euclidean distance for each pea
                d1 = euclidean_distances[..., 0]
                d2 = euclidean_distances[..., 1]
                weight_map = self.w0 * np.exp(-(1 / (2 * self.sigma ** 2)) * ((d1 + d2) ** 2))

                # Adjusting the weight map associated with the body of the pea (excluding the boundaries), using
                # the class weight for  pea's body (foreground)
                weight_map[pea_body_excluding_the_boundary] = foreground_weight

                # Adjusting the weight map associated with the background, using the class weight for
                weight_map[~pea_body_excluding_the_boundary] += background_weight

                # make the weight map a 3-D array, by adding an additional axis, so that the weight map would be able
                # to multiply the 3-D activation value (output) from the Unet
                weight_map = weight_map[..., np.newaxis]

                return weight_map
            else:
                # compute the class weight for the foreground and background, based on the area occupied by each class.
                # Since the foreground occupies less area, it should be given more weight so that its loss would be
                # penalized more. By so doing, the model would identity the foreground more accurately.
                total_foreground_pixels = np.sum(pea_binary_mask, axis=(0, 1))
                total_background_pixels = np.sum(background, axis=(0, 1))
                total_foreground_and_background_pixels = pea_mask.numpy().size

                if self.use_log_weights:
                    background_weight = np.log10(total_foreground_and_background_pixels / total_background_pixels)

                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = np.log10(total_foreground_and_background_pixels / total_foreground_pixels)

                else:
                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = 1 - total_foreground_pixels / total_foreground_and_background_pixels
                    background_weight = 1 - foreground_weight

                weight_map = np.zeros(shape=(self.height, self.width), dtype=np.float64)

                # Adjusting the weight map associated with the body of the pea (excluding the boundaries), using
                # the class weight for  pea's body (foreground)
                weight_map[pea_binary_mask] = foreground_weight

                # Adjusting the weight map associated with the background, using the class weight for
                weight_map[~pea_binary_mask] += background_weight

                # make the weight map a 3-D array, so that it would be able to multiply the 3-D activation map/mask
                # (output) produce by Unet
                weight_map = weight_map[..., np.newaxis]
                return weight_map

    def _tf_create_weight_map(self, image, mask):
        # pea_mask = mask.numpy()
        [weight_map, ] = tf.py_function(func=self._create_weight_map, inp=[mask], Tout=[tf.float64])
        weight_map.set_shape(shape=(self.height, self.width, 1))
        return image, mask, weight_map

    def _introduce_weight_map_of_ones(self, image, mask):
        """This function introduces a weight map of ones"""
        weight_map_of_ones = tf.ones(shape=(self.height, self.width, 1), dtype=tf.float64)
        return image, mask, weight_map_of_ones

    def _add_label_channels_to_mask(self, image, mask, weight_map):
        """Adds label channels to the mask."""
        mask = tf.cast(mask, dtype=tf.uint8)
        stack_list = []

        # Create individual tensor array for each class. Each tensor would display the area on the original
        # mask that is occupied by individual classes. if we have three class - background, pea, and outline,
        # we would have three tensor arrays.
        for class_index in range(self.number_of_classes):
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the the integer 'class_index'. we want to
            temp_mask = tf.equal(mask[:, :, 0], tf.constant(class_index, dtype=tf.uint8))
            # add each temporary mask to the stack_list.
            stack_list.append(tf.cast(temp_mask, dtype=tf.uint8))

        # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        mask = tf.stack(stack_list, axis=2)  # Axis starts from 0, so axis of 2 represents the third axis
        return image, mask, weight_map

    def _add_label_channels_to_val_or_test_mask(self, image, mask):
        """Adds label channels to the mask."""
        mask = tf.cast(mask, dtype=tf.uint8)
        stack_list = []

        # Create individual tensor array for each class. Each tensor would display the area on the original
        # mask that is occupied by individual classes. if we have three class - background, pea, and outline,
        # we would have three tensor arrays.
        for class_index in range(self.number_of_classes):
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the the integer 'class_index'. we want to
            temp_mask = tf.equal(mask[:, :, 0], tf.constant(class_index, dtype=tf.uint8))
            # add each temporary mask to the stack_list.
            stack_list.append(tf.cast(temp_mask, dtype=tf.uint8))

        # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        mask = tf.stack(stack_list, axis=2)  # Axis starts from 0, so axis of 2 represents the third axis
        return image, mask

    def _normalize_image(self, image, mask, weight_map):
        """Normalizes the pixel values of image to lie between [0, 1] or [-1, 1]."""
        if self.normalize_using_255:
            image = tf.cast(image, dtype=tf.float32) / 255.0
        else:
            image = tf.cast(image, dtype=tf.float32) / 127.5
            image -= 1
        return image, mask, weight_map

    def _normalize_val_or_test_image(self, image, mask):
        """Normalizes the pixel values of image to lie between [0, 1] or [-1, 1]."""
        if self.normalize_using_255:
            image = tf.cast(image, dtype=tf.float32) / 255.0
        else:
            image = tf.cast(image, dtype=tf.float32) / 127.5
            image -= 1
        return image, mask

    @tf.function
    def _step2_produce_weightmap_add_channels_and_normalize_train_set(self, image, mask):
        """
        Augments the image and mask, then adds appropriate number of channels to the mask, in line with
        the number of classes
        :param image: (Tensor) - RGB image
        :param mask: (Tensor) - Mask
        :return: (Tensors) - image and mask
        """
        image, mask, weight_map = self._tf_create_weight_map(image, mask) if self.apply_weight_maps \
            else self._introduce_weight_map_of_ones(image, mask)
        image, mask, weight_map = self._add_label_channels_to_mask(image, mask, weight_map)
        image, mask, weight_map = self._normalize_image(image, mask, weight_map)
        return image, mask, weight_map

    @tf.function
    def _step2_add_channels_and_normalize_validation_sets(self, image, mask):
        """
        Augments the image and mask, then adds appropriate number of channels to the mask, in line with
        the number of classes
        :param image: (Tensor) - RGB image
        :param mask: (Tensor) - Mask
        :return: (Tensors) - image and mask
        """
        image, mask = self._add_label_channels_to_val_or_test_mask(image, mask)
        image, mask = self._normalize_val_or_test_image(image, mask)
        return image, mask

    @tf.function
    def _step2_add_channels_and_normalize_test_sets(self, image, mask):
        """
        Augments the image and mask, then adds appropriate number of channels to the mask, in line with
        the number of classes
        :param image: (Tensor) - RGB image
        :param mask: (Tensor) - Mask
        :return: (Tensors) - image and mask
        """
        image, mask = self._add_label_channels_to_val_or_test_mask(image, mask)
        image, mask = self._normalize_val_or_test_image(image, mask)
        return image, mask

    # Get training, validation and test sets
    def _get_training_dataset(self):
        """
        Prepares shuffled batches of the training set.

        :return: tf Dataset containing the preprocessed training set
        """
        training_dataset = self.training_dataset.map(self._prepare_initial_dataset,
                                                     num_parallel_calls=self.tune)
        training_dataset = training_dataset.map(self._step2_produce_weightmap_add_channels_and_normalize_train_set,
                                                num_parallel_calls=self.tune)
        if self.include_visual_attributes and self.include_visual_attribute_weights:
            training_dataset = tf.data.Dataset.zip((training_dataset, self.training_visual_props_dataset,
                                                    self.train_visual_props_weights_dataset))

            training_dataset = training_dataset.map(
                lambda a, b, c: ({'image': a[0]},  # image
                                 # #ground truth data
                                 {'predicted_mask': a[1],  # mask for computing loss of predicted mask
                                  'processed_mask': a[1]},  # mask for computing loss of morph processed mask
                                 {self.visual_attributes[i]: value for i, value in enumerate(b.values())},

                                 # sample weights
                                 {'predicted_mask': a[2]},
                                 # weight for computing for adjusting weight of predicted mask
                                 {self.visual_attributes[i]: weight for i, weight in enumerate(c.values())}))

            training_dataset = training_dataset.map(
                lambda a, b, c, d, e: (a,
                                       {**b, **c},  # merge ground true data
                                       {**d, **e}))  # merge weights

        elif self.include_visual_attributes:
            training_dataset = tf.data.Dataset.zip((training_dataset, self.training_visual_props_dataset))
            # a is a tuple containing the rgb image, ground truth mask and sample weight map. While b is a dictionary
            # that contains the corresponding visual attributes for each each rgb image.
            training_dataset = training_dataset.map(
                lambda a, b: ({'image': a[0]},  # image
                              # ground truth data
                              {'predicted_mask': a[1],  # mask for computing loss of predicted mask
                               'processed_mask': a[1]},  # mask for computing loss of morph processed mask
                              {self.visual_attributes[i]: value for i, value in enumerate(b.values())},

                              # sample weights
                              {'predicted_mask': a[2]}))

            training_dataset = training_dataset.map(
                lambda a, b, c, d: (a,
                                    {**b, **c},  # merge masks and visual attributes
                                    d))
        else:
            training_dataset = training_dataset.map(
                lambda x, y, z: ({'image': x},  # image
                                 # mask for computing loss of the predicted and processed mask
                                 {'predicted_mask': y,
                                  'processed_mask': y},
                                 # weight to compute loss of the morphologically processed mask
                                 {'predicted_mask': z}))

        # training_dataset = training_dataset.take(count=-1)
        training_dataset = training_dataset.cache(filename=self.train_cache) \
            if self.train_cache else training_dataset

        training_dataset = training_dataset.shuffle(buffer_size=self.train_shuffle_size) \
            if self.shuffle else training_dataset
        training_dataset = training_dataset.batch(self.batch_size, drop_remainder=True)
        training_dataset = training_dataset.repeat() if self.prefetch_data else training_dataset
        training_dataset = training_dataset.prefetch(self.tune) if self.prefetch_data else training_dataset
        return training_dataset

    def _get_validation_dataset(self):
        """
        Prepares batches of the validation set.

        :return: tf Dataset containing the preprocessed validation set
        """
        validation_dataset = self.validation_dataset.map(self._prepare_initial_dataset,
                                                         num_parallel_calls=self.tune)
        validation_dataset = validation_dataset.map(self._step2_add_channels_and_normalize_validation_sets,
                                                    num_parallel_calls=self.tune)
        if self.include_visual_attributes:
            validation_dataset = tf.data.Dataset.zip((validation_dataset, self.validation_visual_props_dataset))
            # a is a tuple containing the rgb image, ground truth mask and sample weight map. While b is a dictionary
            # that contains the corresponding visual attributes for each each rgb image.
            validation_dataset = validation_dataset.map(
                lambda a, b: ({'image': a[0]},  # image
                              # ground truth data
                              {'predicted_mask': a[1],  # mask for computing loss of predicted mask
                               'processed_mask': a[1]},  # mask for computing loss of morph processed mask
                              {self.visual_attributes[i]: value for i, value in enumerate(b.values())}))

            validation_dataset = validation_dataset.map(
                lambda a, b, c: (a,
                                 {**b, **c}))  # merge ground true data

        else:
            validation_dataset = validation_dataset.map(lambda x, y: ({'image': x},  # image
                                                                      {'predicted_mask': y,
                                                                       'processed_mask': y}
                                                                      ))  # mask

        # validation_dataset = validation_dataset.take(count=-1)
        validation_dataset = validation_dataset.cache(filename=self.val_cache) \
            if self.val_cache else validation_dataset
        validation_dataset = validation_dataset.shuffle(
            buffer_size=self.validation_shuffle_size) if self.shuffle_validation_data else validation_dataset
        validation_dataset = validation_dataset.batch(self.batch_size, drop_remainder=True)
        validation_dataset = validation_dataset.repeat() if self.prefetch_data else validation_dataset
        validation_dataset = validation_dataset.prefetch(buffer_size=self.tune) \
            if self.prefetch_data else validation_dataset
        return validation_dataset

    def _get_test_dataset(self):
        """
        Prepares batches of the test set.

        :return: tf Dataset containing the preprocessed test set
        """
        test_dataset = self.test_dataset.map(self._prepare_initial_dataset,
                                             num_parallel_calls=self.tune)
        test_dataset = test_dataset.map(self._step2_add_channels_and_normalize_test_sets, num_parallel_calls=self.tune)
        if self.include_visual_attributes:
            test_dataset = tf.data.Dataset.zip((test_dataset, self.test_visual_props_dataset))

            # a is a tuple containing the rgb image, ground truth mask and sample weight map. While b is a dictionary
            # that contains the corresponding visual attributes for each each rgb image.
            test_dataset = test_dataset.map(
                lambda a, b: ({'image': a[0]},  # image
                              # ground truth data
                              {'predicted_mask': a[1],  # mask for computing loss of predicted mask
                               'processed_mask': a[1]},  # mask for computing loss of morph processed mask
                              {self.visual_attributes[i]: value for i, value in enumerate(b.values())}))

            test_dataset = test_dataset.map(
                lambda a, b, c: (a,
                                 {**b, **c}))  # merge ground true data
        else:
            test_dataset = test_dataset.map(
                lambda x, y: ({'image': x},  # image
                              {'predicted_mask': y,  # to compute loss of the predicted mask
                               'processed_mask': y}))  # to compute loss of the morphologically processed mask

        # test_dataset = test_dataset.take(count=-1)
        test_dataset = test_dataset.cache(filename=self.test_cache) \
            if self.test_cache else test_dataset
        test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)
        test_dataset = test_dataset.repeat() if self.prefetch_data else test_dataset
        return test_dataset

    def _produce_train_val_test_datasets(self):
        """
        Produces the training, validation and test datasets.
        """
        if self.train_directory is not None:
            self.training_dataset = self._get_training_dataset()

        if self.validation_directory is not None:
            self.validation_dataset = self._get_validation_dataset()

        if self.test_directory is not None:
            self.test_dataset = self._get_test_dataset()

    @staticmethod
    def get_images_and_masks_from_dataset(dataset, no_of_images):
        """
        This function produces a list of images, and a list of corresponding masks, from a tf dataset object.

        :param dataset: (tf dataset): Dataset containing images and corresponding mask
        :param no_of_images: Number of images to be in the lis
        :return: List of images and list of masks (label maps)
        """
        images = []
        masks = []

        ds = dataset.unbatch()
        ds = ds.batch(no_of_images)

        for image_batch, mask_batch in ds.take(1):
            images = image_batch
            # The mask has n number of slices equal to the number of classes, get the slice pixel locations with the
            # highest probability.
            masks = np.argmax(mask_batch, axis=3)
        return images, masks


class ImageMaskVisualDatasetBuilder:

    def __init__(self,
                 train_directory: Optional[str],
                 validation_directory: Optional[str] = None,
                 test_directory: Optional[str] = None,
                 final_image_shape: Tuple[int, int] = (1024, 1024),
                 image_key: str = 'images',
                 mask_key: str = 'targets',
                 apply_weight_maps: bool = True,
                 use_log_weights: bool = False,
                 apply_unet_edge_weights: bool = False,
                 image_mask_channels: Tuple[int, int] = (3, 1),
                 class_names: tuple = ('background', 'pea', 'outline'),
                 batch_size: Optional[int] = 2,
                 normalize_image: bool = True,
                 normalize_using_255: bool = True,
                 crop_image_and_mask: bool = False,
                 crop_dimension: Optional[Tuple[int, int, int, int]] = None,
                 w0: int = 10,
                 sigma: int = 5,
                 shuffle: bool = True,
                 train_shuffle_size: Optional[int] = 100,
                 shuffle_validation_data=False,
                 validation_shuffle_size: Optional[int] =100,
                 prefetch_data: bool = True,
                 cache_directory: Optional[str]=None,
                 overwrite_existing_cache_directory: bool = True,
                 include_visual_attributes: bool = True,
                 include_visual_attribute_weights: bool = True,
                 visual_attributes: tuple = ('L', 'a', 'b', 'contrast', 'correlation', 'energy', 'entropy',
                                             'homogeneity', 'uniformity', 'equivalent_diameter', 'eccentricity',
                                             'feret_diameter_max', 'filled_area', 'perimeter', 'roundness'),
                 train_visual_attributes_json_path: Optional[str] = None,
                 val_visual_attributes_json_path: Optional[str] = None,
                 test_visual_attributes_json_path: Optional[str] = None,
                 visual_attributes_stats_save_dir: Optional[str] = None):
        """
        Builds a TensorFlow dataset for semantic segmentation tasks enriched with image-mask pairs, weight maps, and object-level visual attributes.

        This dataset builder prepares training, validation, and test datasets by reading RGB images
        and corresponding segmentation masks. It includes preprocessing steps such as cropping,
        resizing, normalization, optional edge detection, weight map generation using the UNet
        weighting method, and integration of auxiliary visual attributes (e.g., color and texture
        features) for multi-target learning.

        If `class_names` includes 'outline' (e.g., ('background', 'pea', 'outline')), the class
        will extract and label object edges (e.g., pea outlines) in the final mask. If only
        'background' and 'pea' are provided, outlines will be omitted.

        :param train_directory: (str)  Path to the directory containing the training dataset.
            It must include two subdirectories named 'images' and 'masks', containing the
            input images and corresponding masks, respectively.
        :param validation_directory: (str) Path to the directory containing the validation dataset.
            It must include two subdirectories named 'images' and 'masks', containing the
            input images and corresponding masks, respectively.
        :param test_directory: (str) Path to the directory containing the test dataset.
            It must include two subdirectories named 'images' and 'masks', containing the
            input images and corresponding masks, respectively.
        :param final_image_shape: (Tuple[int, int]) Target shape to resize images and masks to.
        :param apply_weight_maps: (bool) Whether to generate weight maps for each mask.
        :param use_log_weights: (bool) Whether to compute class weights using log scaling.
        :param apply_unet_edge_weights: (bool) Whether to apply UNet-style edge-aware weighting.
        :param image_mask_channels: (Tuple[int, int]) Number of channels for image and mask respectively.
        :param class_names: (Tuple[str]) Names of segmentation classes (e.g., 'background', 'pea').
        :param batch_size: (int) Number of samples per training/validation/test batch.
        :param normalize_using_255: (bool) Normalize images to [0, 1] if True, else to [-1, 1].
        :param crop_image_and_mask: (bool) Whether to crop the image and mask before resizing.
        :param crop_dimension: (Tuple[int, int, int, int]) Crop window in the format
            (offset_height, offset_width, target_height, target_width).
        :param w0: (int) UNet edge-weighting parameter to control influence of object boundaries.
        :param sigma: (int) UNet edge-weighting Gaussian smoothing parameter.
        :param shuffle: (bool) Whether to shuffle training data before batching.
        :param train_shuffle_size: (int) Size of buffer used when shuffling the training data.
        :param shuffle_validation_data: (bool) Whether to shuffle validation data before batching.
        :param validation_shuffle_size: (int) Size of buffer used when shuffling validation data.
        :param prefetch_data: (bool) Whether to prefetch data for improved I/O performance.
        :param cache_directory: (str) Directory to cache the preprocessed datasets.
        :param overwrite_existing_cache_directory: (bool) Whether to overwrite existing cached datasets.
        :param include_visual_attributes: (bool) Whether to include object-level visual attributes.
        :param include_visual_attribute_weights: (bool) Whether to compute weights for visual attributes.
        :param visual_attributes: (Tuple[str]) List of visual attributes to include
            (e.g., 'L', 'a', 'b', 'contrast', 'eccentricity', etc.).
        :param train_visual_attributes_json_path: (str) Path to JSON file containing training set visual attributes.
        :param val_visual_attributes_json_path: (str) Path to JSON file containing validation set visual attributes.
        :param test_visual_attributes_json_path: (str) Path to JSON file containing test set visual attributes.
        :param visual_attributes_stats_save_dir: (str) Directory to save summary statistics (mean, std, etc.) for normalization.
        """
        self.check_parameter_type(params_and_types={
            'train_directory': (train_directory, str,  False),
            'validation_directory': (validation_directory, str,  True),
            'test_directory': (test_directory, str,  True),
            'final_image_shape': (final_image_shape, tuple,  False),
            'image_key': (image_key, str,  False),
            'mask_key': (mask_key, str,  False),
            'apply_weight_maps': (apply_weight_maps, bool,  False),
            'use_log_weights': (use_log_weights, bool,  False),
            'image_mask_channels': (image_mask_channels, tuple, False),
            'class_names': (class_names, tuple,  False),
            'batch_size': (batch_size, int,  False),
            'normalize_using_255': (normalize_using_255, bool,  False),
            'crop_image_and_mask': (crop_image_and_mask, bool,  False),
            'crop_dimension': (crop_dimension, tuple,  True),
            'w0': (w0, int,  False),
            'sigma': (sigma, int, False),
            'train_shuffle_size': (train_shuffle_size, int,  True),
            'shuffle_validation_data': (shuffle_validation_data, bool,  False),
            'validation_shuffle_size': (validation_shuffle_size, int,  True),
            'prefetch_data': (prefetch_data, bool,  False),
            'cache_directory': (cache_directory, str,  True),
            'overwrite_existing_cache_directory': (overwrite_existing_cache_directory, bool,  False),
            'include_visual_attributes': (include_visual_attributes, bool,  False),
            'include_visual_attribute_weights': (include_visual_attribute_weights, bool, False),
            'visual_attributes': (visual_attributes, tuple, False),
            'train_visual_attributes_json_path': (train_visual_attributes_json_path, str,  True),
            'val_visual_attributes_json_path': (val_visual_attributes_json_path, str,  True),
            'test_visual_attributes_json_path': (test_visual_attributes_json_path, str,  True),
            'visual_attributes_stats_save_dir': (visual_attributes_stats_save_dir, str,  True),
        })

        train_dataset_builder = ImageAndMaskDatasetBuilder(
            images_directory=f"{train_directory}/images",
            masks_directory=f"{train_directory}/masks",
            image_mask_channels=image_mask_channels,
            final_image_shape=final_image_shape,
            image_key=image_key,
            mask_key=mask_key,
            crop_image_and_mask=crop_image_and_mask,
            crop_dimension=crop_dimension,
            return_2d_mask=True
        )
        train_dataset_builder.run()
        self.training_dataset = train_dataset_builder.image_mask_dataset
        self.image_key = train_dataset_builder.image_key
        self.mask_key = train_dataset_builder.mask_key


        # Generate unprocess validation set
        if validation_directory is not None:
            val_dataset_builder = ImageAndMaskDatasetBuilder(
                images_directory=f"{validation_directory}/images",
                masks_directory=f"{validation_directory}/masks",
                image_mask_channels=image_mask_channels,
                final_image_shape=final_image_shape,
                image_key=image_key,
                mask_key=mask_key,
                crop_image_and_mask=crop_image_and_mask,
                crop_dimension=crop_dimension,
                return_2d_mask=True
            )
            val_dataset_builder.run()
            self.validation_dataset = val_dataset_builder.image_mask_dataset
        else:
            self.validation_dataset = None


        # Generate unprocess testing set
        if test_directory is not None:
            test_dataset_builder = ImageAndMaskDatasetBuilder(
                images_directory=f"{test_directory}/images",
                masks_directory=f"{test_directory}/masks",
                image_mask_channels=image_mask_channels,
                final_image_shape=final_image_shape,
                image_key=image_key,
                mask_key=mask_key,
                crop_image_and_mask=crop_image_and_mask,
                crop_dimension=crop_dimension,
                return_2d_mask=True
            )
            test_dataset_builder.run()
            self.test_dataset = test_dataset_builder.image_mask_dataset
        else:
            self.test_dataset = None


        self.apply_unet_edge_weights = apply_unet_edge_weights
        self.test_directory = test_directory
        self.validation_directory = validation_directory
        self.train_directory = train_directory

        self.original_image_paths = []
        self.original_mask_paths = []
        # 'no_pea_threshold' value that would be returned as threshold when the mask contains not infocus pea. The
        # threshold would be used to separate the foreground (infocus peas) from the background.
        self.no_pea_thresh = 400.0
        self.normalize_image = normalize_image


        self.use_log_weights = use_log_weights
        self.class_names = class_names

        self.image_size = train_dataset_builder.final_image_shape
        self.height = self.image_size[0]
        self.width = self.image_size[1]

        self.apply_weight_maps = apply_weight_maps
        self.image_mask_channels = image_mask_channels
        self.crop_image_and_mask = crop_image_and_mask
        self.crop_dimension = crop_dimension
        self.w0 = w0
        self.sigma = sigma
        self.normalize_using_255 = normalize_using_255

        self.shuffle = shuffle
        self.train_shuffle_size = train_shuffle_size
        self.validation_shuffle_size = validation_shuffle_size
        self.shuffle_validation_data = shuffle_validation_data

        self.batch_size = batch_size

        self.number_of_classes = len(self.class_names)
        self.detect_outline = True if self.number_of_classes == 3 else False


        self.tune = tf.data.experimental.AUTOTUNE
        self.cache_directory = cache_directory
        self.prefetch_data = prefetch_data

        # Cache directories
        self.overwrite_cache_dir = overwrite_existing_cache_directory
        if isinstance(cache_directory, str):
            if cache_directory != '':
                self.train_cache = create_directory(os.path.join(self.cache_directory, 'train'),
                                                    return_dir=True,
                                                    overwrite_if_existing=self.overwrite_cache_dir)
                self.val_cache = create_directory(os.path.join(self.cache_directory, 'val'),
                                                  return_dir=True,
                                                  overwrite_if_existing=self.overwrite_cache_dir)
                self.test_cache = create_directory(os.path.join(self.cache_directory, 'test'),
                                                   return_dir=True,
                                                   overwrite_if_existing=self.overwrite_cache_dir)
            else:
                self.train_cache = self.val_cache = self.test_cache = ''
        else:
            self.train_cache = self.val_cache = self.test_cache = None


        if train_visual_attributes_json_path is not None:
            self.training_visual_props_json_path = train_visual_attributes_json_path
        else:
            self.training_visual_props_json_path = None

        if val_visual_attributes_json_path is not None:
            self.validation_visual_props_json_path = val_visual_attributes_json_path
        else:
            self.validation_visual_props_json_path = None

        if test_visual_attributes_json_path is not None:
            self.test_visual_props_json_path = test_visual_attributes_json_path
        else:
            self.test_visual_props_json_path = None

        self.visual_attributes_stats_save_dir = visual_attributes_stats_save_dir
        self.training_visual_attributes_json_path = train_visual_attributes_json_path
        self.validation_visual_attributes_json_path = val_visual_attributes_json_path
        self.test_visual_attributes_json_path = test_visual_attributes_json_path

        self.include_visual_attributes = include_visual_attributes
        self.include_visual_attribute_weights = include_visual_attribute_weights

        if self.include_visual_attributes:
            self.visual_attributes = visual_attributes
            self.visual_props_processor = VisualAttributesDatasetCreator(
                train_visual_attributes_json_path=self.training_visual_attributes_json_path,
                val_visual_attributes_json_path=self.validation_visual_attributes_json_path,
                test_visual_attributes_json_path=self.test_visual_attributes_json_path,
                visual_properties=self.visual_attributes)

            self.visual_props_processor.process_data()
            self.train_unnorm_visual_props_dataframe = self.visual_props_processor.train_unnorm_visual_props_dataframe
            self.val_unnorm_visual_props_dataframe = self.visual_props_processor.val_unnorm_visual_props_dataframe
            self.test_unnorm_visual_props_dataframe = self.visual_props_processor.test_unnorm_visual_props_dataframe
            self.train_visual_props_weights_dataframe = self.visual_props_processor.train_visual_props_weights_dataframe
            self.train_visual_props_weights_dataset = self.visual_props_processor.train_visual_props_weights_dataset

            self.train_normalized_visual_props_dataframe = self.visual_props_processor.train_normalized_visual_props_df
            self.val_normalized_visual_props_dataframe = self.visual_props_processor.val_normalized_visual_props_df
            self.test_normalized_visual_props_df = self.visual_props_processor.test_normalized_visual_props_df

            self.training_visual_props_dataset = self.visual_props_processor.train_visual_props_dataset
            self.validation_visual_props_dataset = self.visual_props_processor.val_visual_props_dataset
            self.test_visual_props_dataset = self.visual_props_processor.test_visual_props_dataset
        else:
            self.visual_attributes = None


    def process_data(self):
        """Reads the images and masks from their stored location, then prepares a tensorflow training, validation and
         test dataset."""

        # produce the training validation and test datasets
        self._produce_train_val_test_datasets()

    @staticmethod
    def check_parameter_type(params_and_types: Dict[str, Tuple[Any, Union[Type, Tuple[Type, ...]], bool]]):
        """
        Validates that each parameter value matches its expected type, and handles optional parameters.

        :param params_and_types: A dictionary where each key is a parameter name, and the value is a tuple:
            (value, expected_type(s), is_optional), where:
                - value: The actual value to validate.
                - expected_type(s): A type or tuple of types that the value is expected to match.
                - is_optional: If True, allows the value to be None.

            Example:
                check_type({
                    "name": ("Alice", str, False),
                    "age": (None, int, True),
                    "scores": ([90, 95], list, False),
                })

        :raises TypeError:
            - If a value is None and is_optional is False.
            - If a value does not match the expected type(s).
        """
        for param_name, (value, expected_types, is_optional) in params_and_types.items():
            if value is None:
                if is_optional:
                    continue
                else:
                    raise TypeError(f"{param_name} must not be None.")

            if not isinstance(value, expected_types):
                allowed_types = (
                    expected_types.__name__
                    if isinstance(expected_types, type)
                    else ', '.join(t.__name__ for t in expected_types)
                )
                raise TypeError(f"{param_name} must be of type {allowed_types}. Got {type(value).__name__}.")


    @staticmethod
    def _apply_gaussian_filter(mask):
        """De-noises the image using a gaussian filter.

        :param mask:  (numpy array) A numpy array of the mask
        :return: (numpy array) De-noised/filtered mask
        """
        mask_smooth = gaussian(mask)
        return mask_smooth

    def _tf_apply_gaussian_filter(self, image, mask):
        """De-noises the mask using a Gaussian filter"""
        mask_shape = mask.shape
        image_shape = image.shape

        [mask_smooth, ] = tf.py_function(self._apply_gaussian_filter, [mask], [tf.float64])

        mask_smooth.set_shape(mask_shape)
        image.set_shape(image_shape)
        return image, mask_smooth

    def _compute_otsu_threshold(self, mask_smooth):
        """
        Return threshold value (for detecting the infocus peas) using the Otsu's method.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :return: (float) Upper threshold value. All pixels with an intensity higher
            than this value are assumed to be the object of interest (in-focus pea)
        """
        mask_smooth = mask_smooth.numpy()

        # confirm if there is an in-focus pea in the image. If the maximum pixel intensity if
        # greater than zero, there is an infocus pea, else there isn't.
        if mask_smooth.max() == 0:
            threshold = self.no_pea_thresh
        else:
            threshold = threshold_otsu(mask_smooth)
        return threshold

    def _tf_compute_otsu_threshold(self, image, mask_smooth):
        """
        Return threshold value based on Otsu's method.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :return: (float) Upper threshold value. All pixels with an intensity higher
            than this value are assumed to be the object of interest (in-focus pea)
        """

        image_shape = image.shape
        mask_shape = mask_smooth.shape

        threshold = tf.py_function(self._compute_otsu_threshold, [mask_smooth], tf.float64)

        threshold.set_shape(shape=())
        image.set_shape(image_shape)
        mask_smooth.set_shape(mask_shape)
        return image, mask_smooth, threshold

    @staticmethod
    def _detect_regions_containing_peas(mask_smooth, threshold):
        """
        Produce a mask representing the pixel locations occupied by the pea.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :param threshold: (float) The Otsu's threshold
        :return: (tensor) Mask of in-focus peas alone, without the edges
        """
        mask_smooth = mask_smooth.numpy()
        threshold = threshold.numpy()

        if threshold == 300:
            pea_mask = np.where(mask_smooth == 0, False, True)
            temp_mask = np.where(mask_smooth == 0, 0, 1)
        else:
            # Region containing in-focus peas
            pea_mask = np.where(mask_smooth > threshold, True, False)
            temp_mask = np.where(mask_smooth > threshold, 1, 0)

        # convert the pea mask to float, so the canny edge diagram can be produced.
        temp_mask = np.asarray(temp_mask, dtype=np.float64)
        return pea_mask, temp_mask, threshold

    def _tf_detect_regions_containing_peas(self, image, mask_smooth, threshold):
        """
        Produce a mask representing the pixel locations occupied by the pea.

        :param image: (Tensor) RGB image of all the peas
        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :param threshold: (float) The Otsu's threshold
        :return: (tensor) Mask of in-focus peas alone, without the edges
        """
        mask_shape = mask_smooth.shape
        image_shape = image.shape

        [pea_mask, temp_mask, threshold] = tf.py_function(self._detect_regions_containing_peas,
                                                          [mask_smooth, threshold],
                                                          [tf.bool, tf.float64, tf.float64])

        temp_mask.set_shape(shape=mask_shape)
        pea_mask.set_shape(shape=mask_shape)
        image.set_shape(image_shape)
        threshold.set_shape(shape=())

        return image, pea_mask, temp_mask, threshold

    def _produce_whole_peas_without_outline(self, mask_smooth, threshold):
        """
        Produce a mask representing the pixel locations occupied by the pea.

        :param mask_smooth: (Tensor) grayscale mask smoothened with a gaussian filter
        :param threshold: (float) The Otsu's threshold
        :return: (tensor) Mask of in-focus peas alone, without the edges
        """
        mask_smooth = mask_smooth.numpy()
        threshold = threshold.numpy()

        if threshold == self.no_pea_thresh:
            pea = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)
        else:
            # Region containing in-focus peas
            pea_mask = np.where(mask_smooth > threshold, True, False)

            # Add the pea body to the blank mask that already contains the edge
            pea = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)

            pea[pea_mask] = 1

        # convert the pea image to float, so the canny edge diagram can be produced.
        pea = tf.convert_to_tensor(pea, dtype=tf.float32)
        return pea, pea, threshold

    def _tf_produce_whole_peas_without_outline(self, image, mask_smooth, threshold):
        """Produce a mask representing the pixel locations occupied by the pea."""
        mask_shape = (self.height, self.width, 1)
        image_shape = image.shape

        [pea, temp_mask, threshold] = tf.py_function(self._produce_whole_peas_without_outline, [mask_smooth, threshold],
                                                     [tf.float32, tf.float32, tf.float64])
        temp_mask.set_shape(shape=mask_shape)
        pea.set_shape(shape=mask_shape)
        image.set_shape(image_shape)
        threshold.set_shape(shape=())
        return image, pea, temp_mask, threshold

    @staticmethod
    def _produce_canny_edges(temp_mask, threshold):
        """
        Produce the edges of the peas using the canny edge detector

        :param temp_mask: (Tensor) Mask of the in-focus peas
        :return: (Tensor) Canny edge diagram of the peas
        """
        temp_mask = temp_mask.numpy()
        threshold = threshold.numpy()

        if threshold == 0:
            pea_outline_mask = np.zeros_like(a=temp_mask, dtype=np.bool)
        else:
            pea_outline_mask = canny(temp_mask)
        return pea_outline_mask

    def _tf_produce_canny_edge(self, image, pea_mask, temp_mask, threshold):
        """
        Produces the edges of the peas using the canny edge detector

        :param image: (Tensor) RGB image of all the peas
        :param pea_mask:  (Tensor) Mask of the in-focus peas
        :param temp_mask: (Tensor) Mask of the in-focus peas. temp_mask would be used to obtain
            the canny edge diagram of the in-focus peas
        :return: (Tensor) Canny edge diagram of the peas
        """
        mask_shape = pea_mask.shape
        [pea_outline_mask, ] = tf.py_function(self._produce_canny_edges, [temp_mask, threshold], [tf.bool])
        pea_outline_mask.set_shape(shape=mask_shape)
        return image, pea_mask, pea_outline_mask

    def _produce_final_mask(self, pea_mask, pea_outline_mask):
        """
        Produce the overall mask, made up pixel values  representing of the body of the in-focus
        peas, their exterior outline (edges), and the background. The pixel intensity for each class
        is labeled as follows:
            - background of image:  0
            - pea outline:  1
            - peas body: 2
        :param pea_mask: (Tensor) Mask of the in-focus peas
        :param pea_outline_mask: (Tensor) Canny edge diagram of the infocus peas
        :return: mask of pea, their outline and background all together.
        """
        pea_outline_mask = pea_outline_mask.numpy()
        pea_mask = pea_mask.numpy()

        # Add the pea body to the blank mask that already contains the edge
        new_mask = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)

        new_mask[pea_mask] = 1

        new_mask[pea_outline_mask] = 2
        new_mask = tf.convert_to_tensor(new_mask, dtype=tf.float32)

        return new_mask

    def _tf_produce_final_mask(self, image, pea_mask, pea_outline_mask):
        mask_shape = pea_mask.shape
        [new_mask, ] = tf.py_function(self._produce_final_mask, [pea_mask, pea_outline_mask], [tf.float32])
        new_mask.set_shape(mask_shape + (1,))
        return image, new_mask

    def _prepare_initial_dataset(self, image: tf.Tensor, mask: tf.Tensor):
        """
        Reads and decodes and image and its corresponding masks.
        This function also assigns labels to the the background of the image, the pea's outline and its body. The labels
        are as follows:
        - background of image:  0
        - peas body: 1
        - pea outline:  2

        :return: (Tensors) Two TF Tensors - one for the rgb image, and the other
            for the mask, with the pixel locations properly labelled as background (1)
            pea's body (1), or pea outline (3).
        """
        # image, mask = self._tf_read_and_decode_image_and_mask(image_path=image_path, mask_path=mask_path)
        # image, mask = self._crop_image_and_mask(image=image, mask=mask)
        # image, mask = self._resize_image_and_mask(image=image, mask=mask) if self.resize_image else (image, mask)
        image, mask = self._tf_apply_gaussian_filter(image=image, mask=mask)
        image, mask, threshold = self._tf_compute_otsu_threshold(image=image, mask_smooth=mask)

        image, mask, temp_mask, threshold = self._tf_detect_regions_containing_peas(image=image, mask_smooth=mask,
                                                                                    threshold=threshold) if \
            self.detect_outline else self._tf_produce_whole_peas_without_outline(image=image, mask_smooth=mask,
                                                                                 threshold=threshold)

        image, mask, edge_mask = self._tf_produce_canny_edge(image=image, pea_mask=mask,
                                                             temp_mask=temp_mask, threshold=threshold) if \
            self.detect_outline else (image, mask, temp_mask)
        image, mask = self._tf_produce_final_mask(image, mask, edge_mask) if self.detect_outline else (image, mask)
        return image, mask

    # STEP-2
    # Functions for label channel addition and image normalization
    def _create_weight_map(self, mask):
        """
        Creates a weight map of a mask, according to the Unet method
        :param mask: (numpy array) The mask of the infocus peas
        :return: (numpy array) The weight map of of the infocus peas (and their outline, if outlines are used)
        """
        # The shape of the mask is in the form [height, width, 1], but we need a 2-D array, hence, we
        # extract the first layer
        pea_mask = mask[..., 0]

        # If the mask contains no infocus peas, there is no need to continue with the computations below, instead,
        # return a weight map containing ones, so that when the predicted mask is multiplied by the weight map of ones,
        # the predicted mask does not changed.
        if pea_mask.numpy().max() == 0:
            weight_map = tf.ones(shape=(self.height, self.width, 1), dtype=tf.float64)
            return weight_map
        else:
            # label each pea with integer values
            labelled_pea_mask = label(pea_mask)

            # obtain the regions on the mask containing infocus peas, and the region occupied by the background
            pea_binary_mask = labelled_pea_mask > 0
            background = (labelled_pea_mask == 0)

            # compute the number of peas within the mask
            no_of_peas = (len(np.unique(labelled_pea_mask)) // 2) - 1 if self.apply_unet_edge_weights else \
                len(np.unique(labelled_pea_mask)) - 1

            # check if more than one peas is contained in the mask
            multiple_peas_exist = (no_of_peas > 1)

            if self.apply_unet_edge_weights and multiple_peas_exist:
                # locate the inner segmentation boundaries of the peas within the mask
                pea_boundaries = find_boundaries(label_img=labelled_pea_mask, mode='inner')

                # obtain the body of the pea, within the pea's inner boundary
                pea_body_excluding_the_boundary = np.bitwise_xor(pea_binary_mask, pea_boundaries)

                # compute the class weight for the foreground and background, based on the area occupied by each class.
                # Since the foreground occupies less area, it should be given more weight so that its loss would be
                # penalized more. By so doing, the model would identity the foreground more accurately.
                total_foreground_pixels = np.sum(pea_body_excluding_the_boundary, axis=(0, 1))
                total_background_pixels = np.sum(background, axis=(0, 1))
                total_foreground_and_background_pixels = pea_mask.numpy().size

                if self.use_log_weights:
                    background_weight = np.log10(total_foreground_and_background_pixels / total_background_pixels)

                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = np.log10(total_foreground_and_background_pixels / total_foreground_pixels)
                else:
                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = 1 - total_foreground_pixels / total_foreground_and_background_pixels
                    background_weight = 1 - foreground_weight

                # produce a list containing the unique label id for each infocus pea in the mask
                pea_ids = [x for x in np.unique(labelled_pea_mask) if x > 0]

                # produce a 3-D array to store the euclidean distances for each of the infocus pea. Each
                # layer in the third axis would be used to store the distance a single pea and other peas.
                euclidean_distances = np.zeros(shape=(self.height, self.width, len(pea_ids)))

                # compute the Euclidean distance of individual peas, starting from a particular pea that has a given
                # pea_id
                for i, pea_id in enumerate(pea_ids):
                    euclidean_distances[..., i] = distance_transform_edt(labelled_pea_mask != pea_id)

                # Sort the Euclidean distance so we could choose the least two for each pea (connected component)
                euclidean_distances.sort(axis=-1)

                # Select the smallest and second to smallest euclidean distance for each pea
                d1 = euclidean_distances[..., 0]
                d2 = euclidean_distances[..., 1]
                weight_map = self.w0 * np.exp(-(1 / (2 * self.sigma ** 2)) * ((d1 + d2) ** 2))

                # Adjusting the weight map associated with the body of the pea (excluding the boundaries), using
                # the class weight for  pea's body (foreground)
                weight_map[pea_body_excluding_the_boundary] = foreground_weight

                # Adjusting the weight map associated with the background, using the class weight for
                weight_map[~pea_body_excluding_the_boundary] += background_weight

                # make the weight map a 3-D array, by adding an additional axis, so that the weight map would be able
                # to multiply the 3-D activation value (output) from the Unet
                weight_map = weight_map[..., np.newaxis]

                return weight_map
            else:
                # compute the class weight for the foreground and background, based on the area occupied by each class.
                # Since the foreground occupies less area, it should be given more weight so that its loss would be
                # penalized more. By so doing, the model would identity the foreground more accurately.
                total_foreground_pixels = np.sum(pea_binary_mask, axis=(0, 1))
                total_background_pixels = np.sum(background, axis=(0, 1))
                total_foreground_and_background_pixels = pea_mask.numpy().size

                if self.use_log_weights:
                    background_weight = np.log10(total_foreground_and_background_pixels / total_background_pixels)

                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = np.log10(total_foreground_and_background_pixels / total_foreground_pixels)

                else:
                    if total_foreground_pixels == 0:
                        foreground_weight = 1
                    else:
                        foreground_weight = 1 - total_foreground_pixels / total_foreground_and_background_pixels
                    background_weight = 1 - foreground_weight

                weight_map = np.zeros(shape=(self.height, self.width), dtype=np.float64)

                # Adjusting the weight map associated with the body of the pea (excluding the boundaries), using
                # the class weight for  pea's body (foreground)
                weight_map[pea_binary_mask] = foreground_weight

                # Adjusting the weight map associated with the background, using the class weight for
                weight_map[~pea_binary_mask] += background_weight

                # make the weight map a 3-D array, so that it would be able to multiply the 3-D activation map/mask
                # (output) produce by Unet
                weight_map = weight_map[..., np.newaxis]
                return weight_map

    def _tf_create_weight_map(self, image, mask):
        # pea_mask = mask.numpy()
        [weight_map, ] = tf.py_function(func=self._create_weight_map, inp=[mask], Tout=[tf.float64])
        weight_map.set_shape(shape=(self.height, self.width, 1))
        return image, mask, weight_map

    def _introduce_weight_map_of_ones(self, image, mask):
        """This function introduces a weight map of ones"""
        weight_map_of_ones = tf.ones(shape=(self.height, self.width, 1), dtype=tf.float64)
        return image, mask, weight_map_of_ones

    def _add_label_channels_to_mask(self, image, mask, weight_map):
        """Adds label channels to the mask."""
        mask = tf.cast(mask, dtype=tf.uint8)
        stack_list = []

        # Create individual tensor array for each class. Each tensor would display the area on the original
        # mask that is occupied by individual classes. if we have three class - background, pea, and outline,
        # we would have three tensor arrays.
        for class_index in range(self.number_of_classes):
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the the integer 'class_index'. we want to
            temp_mask = tf.equal(mask[:, :, 0], tf.constant(class_index, dtype=tf.uint8))
            # add each temporary mask to the stack_list.
            stack_list.append(tf.cast(temp_mask, dtype=tf.uint8))

        # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        mask = tf.stack(stack_list, axis=2)  # Axis starts from 0, so axis of 2 represents the third axis
        return image, mask, weight_map

    def _add_label_channels_to_val_or_test_mask(self, image, mask):
        """Adds label channels to the mask."""
        mask = tf.cast(mask, dtype=tf.uint8)
        stack_list = []

        # Create individual tensor array for each class. Each tensor would display the area on the original
        # mask that is occupied by individual classes. if we have three class - background, pea, and outline,
        # we would have three tensor arrays.
        for class_index in range(self.number_of_classes):
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the the integer 'class_index'. we want to
            temp_mask = tf.equal(mask[:, :, 0], tf.constant(class_index, dtype=tf.uint8))
            # add each temporary mask to the stack_list.
            stack_list.append(tf.cast(temp_mask, dtype=tf.uint8))

        # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        mask = tf.stack(stack_list, axis=2)  # Axis starts from 0, so axis of 2 represents the third axis
        return image, mask

    def _normalize_image(self, image, mask, weight_map):
        """Normalizes the pixel values of image to lie between [0, 1] or [-1, 1]."""
        if not self.normalize_image:
            return image, mask, weight_map

        if self.normalize_using_255:
            image = tf.cast(image, dtype=tf.float32) / 255.0
        else:
            image = tf.cast(image, dtype=tf.float32) / 127.5
            image -= 1
        return image, mask, weight_map

    def _normalize_val_or_test_image(self, image, mask):
        """Normalizes the pixel values of image to lie between [0, 1] or [-1, 1]."""
        if not self.normalize_image:
            return image, mask

        if self.normalize_using_255:
            image = tf.cast(image, dtype=tf.float32) / 255.0
        else:
            image = tf.cast(image, dtype=tf.float32) / 127.5
            image -= 1
        return image, mask

    @tf.function
    def _step2_produce_weightmap_add_channels_and_normalize_train_set(self, image, mask):
        """
        Augments the image and mask, then adds appropriate number of channels to the mask, in line with
        the number of classes
        :param image: (Tensor) - RGB image
        :param mask: (Tensor) - Mask
        :return: (Tensors) - image and mask
        """
        image, mask, weight_map = self._tf_create_weight_map(image, mask) if self.apply_weight_maps \
            else self._introduce_weight_map_of_ones(image, mask)
        image, mask, weight_map = self._add_label_channels_to_mask(image, mask, weight_map)
        image, mask, weight_map = self._normalize_image(image, mask, weight_map)
        return image, mask, weight_map

    @tf.function
    def _step2_add_channels_and_normalize_validation_sets(self, image, mask):
        """
        Augments the image and mask, then adds appropriate number of channels to the mask, in line with
        the number of classes
        :param image: (Tensor) - RGB image
        :param mask: (Tensor) - Mask
        :return: (Tensors) - image and mask
        """
        image, mask = self._add_label_channels_to_val_or_test_mask(image, mask)
        image, mask = self._normalize_val_or_test_image(image, mask)
        return image, mask

    @tf.function
    def _step2_add_channels_and_normalize_test_sets(self, image, mask):
        """
        Augments the image and mask, then adds appropriate number of channels to the mask, in line with
        the number of classes
        :param image: (Tensor) - RGB image
        :param mask: (Tensor) - Mask
        :return: (Tensors) - image and mask
        """
        image, mask = self._add_label_channels_to_val_or_test_mask(image, mask)
        image, mask = self._normalize_val_or_test_image(image, mask)
        return image, mask

    # Get training, validation and test sets
    def _get_training_dataset(self):
        """
        Prepares shuffled batches of the training set.

        :return: tf Dataset containing the preprocessed training set
        """
        training_dataset = self.training_dataset.map(self._prepare_initial_dataset,
                                                     num_parallel_calls=self.tune)
        training_dataset = training_dataset.map(self._step2_produce_weightmap_add_channels_and_normalize_train_set,
                                                num_parallel_calls=self.tune)
        if self.include_visual_attributes and self.include_visual_attribute_weights:
            training_dataset = tf.data.Dataset.zip((training_dataset, self.training_visual_props_dataset,
                                                    self.train_visual_props_weights_dataset))

            training_dataset = training_dataset.map(
                lambda a, b, c: ({self.image_key: a[0]},  # image
                                 # #ground truth data
                                 {self.mask_key: a[1],  # mask for computing loss of predicted mask
                                  f"processed_{self.mask_key}": a[1]},  # mask for computing loss of morph processed mask
                                 {self.visual_attributes[i]: value for i, value in enumerate(b.values())},

                                 # sample weights
                                 {self.mask_key: a[2]},
                                 # weight for computing for adjusting weight of predicted mask
                                 {self.visual_attributes[i]: weight for i, weight in enumerate(c.values())}))

            training_dataset = training_dataset.map(
                lambda a, b, c, d, e: (a,
                                       {**b, **c},  # merge ground true data
                                       {**d, **e}))  # merge weights

        elif self.include_visual_attributes:
            training_dataset = tf.data.Dataset.zip((training_dataset, self.training_visual_props_dataset))
            # a is a tuple containing the rgb image, ground truth mask and sample weight map. While b is a dictionary
            # that contains the corresponding visual attributes for each each rgb image.
            training_dataset = training_dataset.map(
                lambda a, b: ({self.image_key: a[0]},  # image
                              # ground truth data
                              {self.mask_key: a[1],  # mask for computing loss of predicted mask
                               f"processed_{self.mask_key}": a[1]},  # mask for computing loss of morph processed mask
                              {self.visual_attributes[i]: value for i, value in enumerate(b.values())},

                              # sample weights
                              {self.mask_key: a[2]}))

            training_dataset = training_dataset.map(
                lambda a, b, c, d: (a,
                                    {**b, **c},  # merge masks and visual attributes
                                    d))
        else:
            training_dataset = training_dataset.map(
                lambda x, y, z: ({self.image_key: x},  # image
                                 # mask for computing loss of the predicted and processed mask
                                 {self.mask_key: y,
                                  f"processed_{self.mask_key}": y},
                                 # weight to compute loss of the morphologically processed mask
                                 {self.mask_key: z}))

        # training_dataset = training_dataset.take(count=-1)
        training_dataset = training_dataset.cache(filename=self.train_cache) \
            if self.train_cache else training_dataset

        training_dataset = training_dataset.shuffle(buffer_size=self.train_shuffle_size) \
            if self.shuffle else training_dataset
        training_dataset = training_dataset.batch(self.batch_size, drop_remainder=True)
        training_dataset = training_dataset.repeat() if self.prefetch_data else training_dataset
        training_dataset = training_dataset.prefetch(self.tune) if self.prefetch_data else training_dataset
        return training_dataset

    def _get_validation_dataset(self):
        """
        Prepares batches of the validation set.

        :return: tf Dataset containing the preprocessed validation set
        """
        validation_dataset = self.validation_dataset.map(self._prepare_initial_dataset,
                                                         num_parallel_calls=self.tune)
        validation_dataset = validation_dataset.map(self._step2_add_channels_and_normalize_validation_sets,
                                                    num_parallel_calls=self.tune)
        if self.include_visual_attributes:
            validation_dataset = tf.data.Dataset.zip((validation_dataset, self.validation_visual_props_dataset))
            # a is a tuple containing the rgb image, ground truth mask and sample weight map. While b is a dictionary
            # that contains the corresponding visual attributes for each each rgb image.
            validation_dataset = validation_dataset.map(
                lambda a, b: ({'image': a[0]},  # image
                              # ground truth data
                              {'predicted_mask': a[1],  # mask for computing loss of predicted mask
                               'processed_mask': a[1]},  # mask for computing loss of morph processed mask
                              {self.visual_attributes[i]: value for i, value in enumerate(b.values())}))

            validation_dataset = validation_dataset.map(
                lambda a, b, c: (a,
                                 {**b, **c}))  # merge ground true data

        else:
            validation_dataset = validation_dataset.map(lambda x, y: ({'image': x},  # image
                                                                      {'predicted_mask': y,
                                                                       'processed_mask': y}
                                                                      ))  # mask

        # validation_dataset = validation_dataset.take(count=-1)
        validation_dataset = validation_dataset.cache(filename=self.val_cache) \
            if self.val_cache else validation_dataset
        validation_dataset = validation_dataset.shuffle(
            buffer_size=self.validation_shuffle_size) if self.shuffle_validation_data else validation_dataset
        validation_dataset = validation_dataset.batch(self.batch_size, drop_remainder=True)
        validation_dataset = validation_dataset.repeat() if self.prefetch_data else validation_dataset
        validation_dataset = validation_dataset.prefetch(buffer_size=self.tune) \
            if self.prefetch_data else validation_dataset
        return validation_dataset

    def _get_test_dataset(self):
        """
        Prepares batches of the test set.

        :return: tf Dataset containing the preprocessed test set
        """
        test_dataset = self.test_dataset.map(self._prepare_initial_dataset,
                                             num_parallel_calls=self.tune)
        test_dataset = test_dataset.map(self._step2_add_channels_and_normalize_test_sets, num_parallel_calls=self.tune)
        if self.include_visual_attributes:
            test_dataset = tf.data.Dataset.zip((test_dataset, self.test_visual_props_dataset))

            # a is a tuple containing the rgb image, ground truth mask and sample weight map. While b is a dictionary
            # that contains the corresponding visual attributes for each each rgb image.
            test_dataset = test_dataset.map(
                lambda a, b: ({'image': a[0]},  # image
                              # ground truth data
                              {'predicted_mask': a[1],  # mask for computing loss of predicted mask
                               'processed_mask': a[1]},  # mask for computing loss of morph processed mask
                              {self.visual_attributes[i]: value for i, value in enumerate(b.values())}))

            test_dataset = test_dataset.map(
                lambda a, b, c: (a,
                                 {**b, **c}))  # merge ground true data
        else:
            test_dataset = test_dataset.map(
                lambda x, y: ({'image': x},  # image
                              {'predicted_mask': y,  # to compute loss of the predicted mask
                               'processed_mask': y}))  # to compute loss of the morphologically processed mask

        # test_dataset = test_dataset.take(count=-1)
        test_dataset = test_dataset.cache(filename=self.test_cache) \
            if self.test_cache else test_dataset
        test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)
        test_dataset = test_dataset.repeat() if self.prefetch_data else test_dataset
        return test_dataset

    def _produce_train_val_test_datasets(self):
        """
        Produces the training, validation and test datasets.
        """
        if self.train_directory is not None:
            self.training_dataset = self._get_training_dataset()

        if self.validation_directory is not None:
            self.validation_dataset = self._get_validation_dataset()

        if self.test_directory is not None:
            self.test_dataset = self._get_test_dataset()

    @staticmethod
    def get_images_and_masks_from_dataset(dataset, no_of_images):
        """
        This function produces a list of images, and a list of corresponding masks, from a tf dataset object.

        :param dataset: (tf dataset): Dataset containing images and corresponding mask
        :param no_of_images: Number of images to be in the lis
        :return: List of images and list of masks (label maps)
        """
        images = []
        masks = []

        ds = dataset.unbatch()
        ds = ds.batch(no_of_images)

        for image_batch, mask_batch in ds.take(1):
            images = image_batch
            # The mask has n number of slices equal to the number of classes, get the slice pixel locations with the
            # highest probability.
            masks = np.argmax(mask_batch, axis=3)
        return images, masks


class VisualAttributesDatasetCreator:
    def __init__(self,
                 train_visual_attributes_json_path: str,
                 val_visual_attributes_json_path: str = None,
                 test_visual_attributes_json_path: str = None,
                 save_directory: str = None,
                 visual_properties: tuple = ('L', 'a', 'b', 'contrast', 'correlation', 'energy', 'entropy',
                                             'homogeneity', 'uniformity', 'equivalent_diameter', 'eccentricity',
                                             'feret_diameter_max', 'filled_area', 'perimeter', 'roundness')):

        """
        Produces normalized tensorflow dataset of the training, validation and test visual attributes.

        :param train_visual_attributes_json_path: (str) File path to the json file containing the visual attributes for
            each of the images of the training set.
        :param val_visual_attributes_json_path: (str) File path to the json file containing the visual attributes for
            each of the images of the validation set.
        :param test_visual_attributes_json_path: (str) File path to the json file containing the visual attributes for
            each of the images of the test set.
        :param save_directory: (str) path to a the directory where to save a csv file that contains the statistics
            (mean, standard deviation, median, etc) for of the training set. This file would be used for normalization.
        :param visual_properties: (tuple) a tuple containing the various visual properties of interest. These are the
            ones that would be extracted from the json and used to form the tensorflow dataset as well as the final
            pandas dataframe.
        """
        self.visual_properties = list(visual_properties)

        # Dataframe containing the upper threshold for each of the bins associated with the individual visual properties
        self.upper_thresholds_dict = {}

        # create a dictionary containing individual list of number of samples per bin for each visual parameter
        self.num_of_samples_per_bin = {}

        # dictionary containing weight for each bin associated with the respective visual attributes
        self.bin_weights_dict = {}

        self.train_visual_props_json_path = train_visual_attributes_json_path

        if val_visual_attributes_json_path is not None:
            self.val_visual_props_json_path = val_visual_attributes_json_path
        else:
            self.val_visual_props_json_path = None

        if test_visual_attributes_json_path is not None:
            self.test_visual_props_json_path = test_visual_attributes_json_path
        else:
            self.test_visual_props_json_path = None

        self.train_stats = None
        if save_directory is not None:
            self.save_visual_props_stats = True
            self.save_dir = create_directory(dir_name=save_directory, return_dir=True,
                                             overwrite_if_existing=False)
        else:
            self.save_visual_props_stats = False

        self.train_unnorm_visual_props_dataframe = None
        self.val_unnorm_visual_props_dataframe = None
        self.test_unnorm_visual_props_dataframe = None

        self.train_visual_props_weights_dataframe = None
        self.train_visual_props_weights_dataset = None

        self.train_normalized_visual_props_df = None
        self.val_normalized_visual_props_df = None
        self.test_normalized_visual_props_df = None

        self.train_visual_props_dataset = None
        self.val_visual_props_dataset = None
        self.test_visual_props_dataset = None

        self.mean = None
        self.std = None

    def _read_json_file(self, json_file_path):
        """Read a json file and fills the blank rows with zeros."""
        visual_props = pd.read_json(json_file_path)
        visual_props = visual_props.copy().loc[::, self.visual_properties]
        # visual_props.drop(columns=['image_id'], inplace=True)

        # return a zero value for images with no infocus pea
        visual_props.fillna(value=0, inplace=True)
        return visual_props

    def _compute_bin_weights(self, unnorm_train_dataframe):
        """
        Computes the weights for each bin associated with the individual visual attributes. Before computing the
        weights, this method computes a dataframe of the upper threshold of the bins of all the visual attributes. in
        addition to that, it also computes a dictionary, `self.num_of_samples_per_bin`, which is the number of samples
        within each bin for the individual visual attributes.

        :param unnorm_train_dataframe: pandas dataframe of the un-normalized visual attributes values
        """
        # compute the upper threshold of the individual bins that make up each visual attribute
        for visual_property in self.visual_properties:
            # numpy arrays containing the number of samples per bin, as well as the upper thresholds for each bin
            # associated with a particular visual
            num_of_samples_per_bin, upper_thresholds = np.histogram(a=unnorm_train_dataframe[visual_property],
                                                                    bins='doane')

            # update the dictionary self.num_of_samples_per_bin with array of upper_thresholds for each bin associated
            # with each of the visual attributes
            # change 0s in num_per_bin to 1, so we don't divide by zero when computing weights
            zero_index = (num_of_samples_per_bin == 0)
            num_of_samples_per_bin[zero_index] = 1

            # weights per bin
            weight = num_of_samples_per_bin.max() / num_of_samples_per_bin
            weight = np.log10(weight) + 1
            weight[zero_index] = 0

            self.bin_weights_dict[visual_property] = weight
            self.upper_thresholds_dict[visual_property] = upper_thresholds
            num_of_samples_per_bin[zero_index] = 0
            self.num_of_samples_per_bin[visual_property] = num_of_samples_per_bin

    def _produce_visual_attribute_weights_dataframe(self):
        """
        Produces a dataframe containing sample weights for each of the training example within each column in
        a visual properties dataframe.

        :return: Pandas dataframe containing sample weight for each visual properties value with the
            'visual_props_dataframe'.
        """
        visual_attribute_weights_df = pd.DataFrame(data=None, columns=self.visual_properties)
        #
        # for column in visual_props_dataframe.columns:
        #     bin_threshold = self.upper_thresholds_dict[column]
        #     weights = self.bin_weights_dict[column]
        #
        #     visual_attribute_weights_df[column] = visual_props_dataframe[column].apply(func=self._assign_weight,
        #                                                                                args=(bin_threshold, weights))
        for visual_property in self.visual_properties:
            weight_indices = np.digitize(x=self.train_unnorm_visual_props_dataframe[visual_property],
                                         bins=self.upper_thresholds_dict[visual_property],
                                         right=True) - 1
            visual_attribute_weights_df.loc[::, visual_property] = self.bin_weights_dict[visual_property][
                weight_indices]

        return visual_attribute_weights_df

    def _compute_normalization_statistics(self, dataframe):
        """computer mean and standard deviation of the training set"""
        stats = dataframe.describe().transpose()
        mean = stats['mean']
        std = stats['std']
        self.train_stats = stats
        return mean, std

    def _save_stats(self):
        """
        Saves the stats (mean, std, etc) of the visual attributes dataset, for
        the training, validation and test sets.
        """
        self.train_stats.to_csv(f'{self.save_dir}/training_stats.csv')

    def _normalize_dataframe(self, dataframe):
        # normalize dataframe
        normalized_data = (dataframe - self.mean) / self.std
        return normalized_data

    def _read_json_and_produce_normalized_training_dataframe(self):
        """read and normalizes the values within a training dataframe."""
        self.train_unnorm_visual_props_dataframe = self._read_json_file(
            json_file_path=self.train_visual_props_json_path)

        self._compute_bin_weights(unnorm_train_dataframe=self.train_unnorm_visual_props_dataframe)

        self.train_visual_props_weights_dataframe = self._produce_visual_attribute_weights_dataframe()

        # compute normalization statistics (mean and std)
        self.mean, self.std = self._compute_normalization_statistics(dataframe=self.train_unnorm_visual_props_dataframe)
        self.train_normalized_visual_props_df = self._normalize_dataframe(
            dataframe=self.train_unnorm_visual_props_dataframe)

    def _read_json_and_produce_normalized_val_or_test_dataframe(self):
        """read and normalizes a the values within a validation or test dataframe."""
        if self.val_visual_props_json_path is not None:
            self.val_unnorm_visual_props_dataframe = self._read_json_file(
                json_file_path=self.val_visual_props_json_path)
            self.val_normalized_visual_props_df = self._normalize_dataframe(self.val_unnorm_visual_props_dataframe)

        if self.test_visual_props_json_path is not None:
            self.test_unnorm_visual_props_dataframe = self._read_json_file(
                json_file_path=self.test_visual_props_json_path)
            self.test_normalized_visual_props_df = self._normalize_dataframe(self.test_unnorm_visual_props_dataframe)

    def _produce_train_val_test_datasets(self):
        """Produces the training, validation and test datasets."""
        if self.train_visual_props_json_path is not None:
            self.train_visual_props_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=self.train_normalized_visual_props_df.to_dict('list'))

        if self.val_visual_props_json_path is not None:
            self.val_visual_props_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=self.val_normalized_visual_props_df.to_dict('list'))

        if self.test_visual_props_json_path is not None:
            self.test_visual_props_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=self.test_normalized_visual_props_df.to_dict('list'))

    def _produce_train_visual_props_weight_dataset(self):
        """Produces the training, validation and test datasets."""
        self.train_visual_props_weights_dataset = tf.data.Dataset.from_tensor_slices(
            tensors=self.train_visual_props_weights_dataframe.to_dict('list'))

    def process_data(self):
        self._read_json_and_produce_normalized_training_dataframe()
        self._read_json_and_produce_normalized_val_or_test_dataframe()
        self._produce_train_val_test_datasets()
        self._produce_train_visual_props_weight_dataset()
        if self.save_visual_props_stats:
            self._save_stats()