"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models

from PIL import Image, ImageChops, ImageEnhance, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2


from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None, model_type=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.image_sequences = None
        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.model_type = model_type
        self.image_height = 224
        self.image_width = 224
        self.image_channels = 3
        self.input_image_shape = (self.image_height, self.image_width, self.image_channels)
        self.EfficientNetB0_layer = EfficientNetB0(weights = "imagenet", include_top = True, input_shape = self.input_image_shape)
        self.EfficientNetB0_model = models.Model(self.EfficientNetB0_layer.input, self.EfficientNetB0_layer.layers[-3].output)
        self.model_input_1_data_samples_number = 1280
        self.rgb_image_data = []
        self.ela_image_data = []
        
    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True,
                        text=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        segments = segmentation_fn(image)

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, text, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    text,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                if self.model_type is None:
                    self.image_sequences = np.array(imgs)
                    preds = classifier_fn(self.image_data)
                    #preds = classifier_fn(np.array(imgs))
                    labels.extend(preds)
                    imgs = []
                else:
                    self.image_sequences = np.array(imgs)
                    model_input_1 = self.generate_input_1_data(np.array(imgs))
                    model_input_2 = self.process_model_input_2_data(np.array(imgs), text)
                    preds = classifier_fn([model_input_1, model_input_2]) 
                    #preds = classifier_fn([input_1, input_2])
                    labels.extend(preds)
                    imgs = [] 
        if len(imgs) > 0:
                if self.model_type is None:
                    self.image_sequences = np.array(imgs)
                    preds = classifier_fn(self.image_data)
                    #preds = classifier_fn(np.array(imgs))
                    labels.extend(preds)
                else:
                    self.image_sequences = np.array(imgs)
                    model_input_1 = self.generate_input_1_data(np.array(imgs))
                    model_input_2 = self.process_model_input_2_data(np.array(imgs), text)
                    preds = classifier_fn([model_input_1, model_input_2]) 
                    #preds = classifier_fn([input_1, input_2])
                    labels.extend(preds)
        return data, np.array(labels)
    
    def generate_input_1_data(self, image_collection):
        images_number = image_collection.shape[0]
        model_input_1 = np.empty((images_number, self.model_input_1_data_samples_number))
        for image_data_index in range(images_number):
            rgb_image_url = "/content/drive/MyDrive/" + "image_" + str(image_data_index)
            rgb_image_url = rgb_image_url + ".jpg"
            image_path_to_save_image_data = rgb_image_url
            image_data = image_collection[image_data_index]
            cv2.imwrite(image_path_to_save_image_data, image_data)
            # Resize and save rgb image
            normal_image_rgb_cv2 = cv2.imread(rgb_image_url)
            resized_normal_image_rgb_cv2 = cv2.resize(normal_image_rgb_cv2, (self.image_width, self.image_height))
            cv2.imwrite(rgb_image_url, resized_normal_image_rgb_cv2)
            # Read rgb image using cv2 and plt libraries
            normal_image_rgb_cv2 = cv2.imread(rgb_image_url)
            normal_image_rgb_plt = Image.open(rgb_image_url)
            # Generate and save reduced quality rgb image
            image_quality = 95
            image_path_to_save_reduced_quality_normal_image_rgb = rgb_image_url
            cv2.imwrite(image_path_to_save_reduced_quality_normal_image_rgb, normal_image_rgb_cv2, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])
            # Generate ela image
            reduced_quality_normal_image_rgb = Image.open(rgb_image_url)
            ela_image = ImageChops.difference(normal_image_rgb_plt, reduced_quality_normal_image_rgb)
            minimum_and_maximum_pixel_values_of_each_image_channel = ela_image.getextrema()
            maximum_image_pixel_value = max([maximum_image_channel_pixel_value[1] for maximum_image_channel_pixel_value in minimum_and_maximum_pixel_values_of_each_image_channel])
            if maximum_image_pixel_value == 0:
                maximum_image_pixel_value = 1
            brightness_amplification_factor = 3
            maximum_brightness_value = 255.0
            image_brightness_enhancement_factor = (maximum_brightness_value / maximum_image_pixel_value) * brightness_amplification_factor
            enhanced_ela_image = ImageEnhance.Brightness(ela_image).enhance(image_brightness_enhancement_factor)
            # Save and read ela image
            enhanced_ela_image.save(rgb_image_url)
            enhanced_ela_image = cv2.imread(rgb_image_url)
            model_input_1[image_data_index,:] = self.generate_model_input_1_data(enhanced_ela_image)
            self.rgb_image_data.append(resized_normal_image_rgb_cv2)
            self.ela_image_data.append(enhanced_ela_image)
            return enhanced_ela_image
        
    def generate_model_input_1_data(self, ela_image):
        ela_image = ela_image[np.newaxis, ...]
        EfficientNetB0_model_output = self.EfficientNetB0_model(ela_image)
        EfficientNetB0_model_output = EfficientNetB0_model_output.numpy()
        EfficientNetB0_model_output = np.squeeze(np.asarray(EfficientNetB0_model_output))
        return EfficientNetB0_model_output
    
    def process_model_input_2_data(self, image_collection, text_data):
        images_number = image_collection.shape[0]
        processed_text_data = np.zeros((images_number, text_data.shape[0]))
        for text_data_index in range(images_number):
            processed_text_data[text_data_index, ...] = text_data
        return processed_text_data   
