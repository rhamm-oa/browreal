#!/usr/bin/env python
"""
Eyebrow segmentation and shape analysis module.
This module provides functions to segment eyebrows and analyze their shape.
"""

import cv2
import numpy as np
from skimage import morphology, measure, filters, color
from scipy import ndimage
import matplotlib.pyplot as plt

class EyebrowSegmentation:
    """Class for segmenting and analyzing eyebrow shape."""
    
    def __init__(self):
        """Initialize the eyebrow segmentation module."""
        pass
    
    def segment_eyebrows(self, image, landmarks, use_trimap=False, trimap=None):
        """
        Segment eyebrows from the image using landmarks and color thresholding.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (dict): Landmarks dictionary from EyebrowLandmarkDetector
            use_trimap (bool): Whether to use trimap for segmentation
            trimap (numpy.ndarray): Optional trimap image
            
        Returns:
            dict: Dictionary containing segmentation masks and metadata
        """
        if landmarks is None:
            return None
        
        # Create a copy of the image
        img = image.copy()
        
        # Get face bounding box
        face_rect = landmarks['face_rect']
        face_width = face_rect[2] - face_rect[0]
        face_height = face_rect[3] - face_rect[1]
        
        # Define regions of interest (ROI) for left and right eyebrows
        # Expand the region slightly beyond the landmarks
        left_eyebrow = landmarks['left_eyebrow']
        right_eyebrow = landmarks['right_eyebrow']
        
        # Calculate bounding boxes for eyebrows with padding
        padding_x = int(face_width * 0.03)
        padding_y = int(face_height * 0.03)
        
        left_min_x = max(0, int(np.min(left_eyebrow[:, 0])) - padding_x)
        left_min_y = max(0, int(np.min(left_eyebrow[:, 1])) - padding_y)
        left_max_x = min(img.shape[1], int(np.max(left_eyebrow[:, 0])) + padding_x)
        left_max_y = min(img.shape[0], int(np.max(left_eyebrow[:, 1])) + padding_y)
        
        right_min_x = max(0, int(np.min(right_eyebrow[:, 0])) - padding_x)
        right_min_y = max(0, int(np.min(right_eyebrow[:, 1])) - padding_y)
        right_max_x = min(img.shape[1], int(np.max(right_eyebrow[:, 0])) + padding_x)
        right_max_y = min(img.shape[0], int(np.max(right_eyebrow[:, 1])) + padding_y)
        
        # Extract ROIs
        left_roi = img[left_min_y:left_max_y, left_min_x:left_max_x]
        right_roi = img[right_min_y:right_max_y, right_min_x:right_max_x]
        
        # If using trimap, extract the corresponding regions
        if use_trimap and trimap is not None:
            left_trimap_roi = trimap[left_min_y:left_max_y, left_min_x:left_max_x]
            right_trimap_roi = trimap[right_min_y:right_max_y, right_min_x:right_max_x]
            
            # Use trimap for segmentation (assuming trimap has values: 0=background, 1=unknown, 2=foreground)
            left_mask = (left_trimap_roi == 2).astype(np.uint8) * 255
            right_mask = (right_trimap_roi == 2).astype(np.uint8) * 255
        else:
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            left_mask = cv2.adaptiveThreshold(
                left_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            right_mask = cv2.adaptiveThreshold(
                right_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphological operations to clean up the masks
            kernel = np.ones((3, 3), np.uint8)
            left_mask = cv2.morphologyEx(left_mask, cv2.MORPH_OPEN, kernel)
            right_mask = cv2.morphologyEx(right_mask, cv2.MORPH_OPEN, kernel)
            
            # Use color information to refine the mask
            # Convert to HSV for better color segmentation
            left_hsv = cv2.cvtColor(left_roi, cv2.COLOR_BGR2HSV)
            right_hsv = cv2.cvtColor(right_roi, cv2.COLOR_BGR2HSV)
            
            # Define range for dark colors (typical for eyebrows)
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 80])
            
            left_color_mask = cv2.inRange(left_hsv, lower_dark, upper_dark)
            right_color_mask = cv2.inRange(right_hsv, lower_dark, upper_dark)
            
            # Combine masks
            left_mask = cv2.bitwise_and(left_mask, left_color_mask)
            right_mask = cv2.bitwise_and(right_mask, right_color_mask)
            
            # Further refine with connected components analysis
            # Keep only the largest component
            left_labels = measure.label(left_mask)
            left_props = measure.regionprops(left_labels)
            if left_props:
                left_props.sort(key=lambda x: x.area, reverse=True)
                left_mask = (left_labels == left_props[0].label).astype(np.uint8) * 255
            
            right_labels = measure.label(right_mask)
            right_props = measure.regionprops(right_labels)
            if right_props:
                right_props.sort(key=lambda x: x.area, reverse=True)
                right_mask = (right_labels == right_props[0].label).astype(np.uint8) * 255
        
        # Create full-size masks
        full_left_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        full_right_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        full_left_mask[left_min_y:left_max_y, left_min_x:left_max_x] = left_mask
        full_right_mask[right_min_y:right_max_y, right_min_x:right_max_x] = right_mask
        
        # Combine masks
        combined_mask = cv2.bitwise_or(full_left_mask, full_right_mask)
        
        return {
            'left_mask': full_left_mask,
            'right_mask': full_right_mask,
            'combined_mask': combined_mask,
            'left_roi': (left_min_x, left_min_y, left_max_x, left_max_y),
            'right_roi': (right_min_x, right_min_y, right_max_x, right_max_y)
        }
    
    def analyze_shape(self, segmentation):
        """
        Analyze the shape of segmented eyebrows.
        
        Args:
            segmentation (dict): Segmentation dictionary from segment_eyebrows
            
        Returns:
            dict: Dictionary containing shape metrics
        """
        if segmentation is None:
            return None
        
        left_mask = segmentation['left_mask']
        right_mask = segmentation['right_mask']
        
        # Calculate area
        left_area = np.sum(left_mask > 0)
        right_area = np.sum(right_mask > 0)
        
        # Calculate perimeter
        left_perimeter = measure.perimeter(left_mask > 0)
        right_perimeter = measure.perimeter(right_mask > 0)
        
        # Calculate compactness (circularity)
        # Compactness = 4π * area / perimeter²
        left_compactness = (4 * np.pi * left_area) / (left_perimeter ** 2) if left_perimeter > 0 else 0
        right_compactness = (4 * np.pi * right_area) / (right_perimeter ** 2) if right_perimeter > 0 else 0
        
        # Calculate eccentricity (how elliptical the shape is)
        left_props = measure.regionprops(left_mask.astype(int))
        right_props = measure.regionprops(right_mask.astype(int))
        
        left_eccentricity = left_props[0].eccentricity if left_props else 0
        right_eccentricity = right_props[0].eccentricity if right_props else 0
        
        # Determine shape category based on eccentricity and compactness
        avg_eccentricity = (left_eccentricity + right_eccentricity) / 2
        avg_compactness = (left_compactness + right_compactness) / 2
        
        if avg_eccentricity > 0.9:
            shape_category = "straight"
        elif avg_eccentricity > 0.8:
            shape_category = "slightly curved"
        elif avg_eccentricity > 0.7:
            shape_category = "moderately curved"
        else:
            shape_category = "highly curved"
        
        # Determine density
        avg_area = (left_area + right_area) / 2
        
        # Calculate bounding box areas
        left_roi = segmentation['left_roi']
        right_roi = segmentation['right_roi']
        
        left_roi_area = (left_roi[2] - left_roi[0]) * (left_roi[3] - left_roi[1])
        right_roi_area = (right_roi[2] - right_roi[0]) * (right_roi[3] - right_roi[1])
        
        left_density = left_area / left_roi_area if left_roi_area > 0 else 0
        right_density = right_area / right_roi_area if right_roi_area > 0 else 0
        avg_density = (left_density + right_density) / 2
        
        if avg_density < 0.3:
            density_category = "sparse"
        elif avg_density < 0.5:
            density_category = "medium"
        else:
            density_category = "dense"
        
        return {
            'left_area': left_area,
            'right_area': right_area,
            'avg_area': avg_area,
            'left_perimeter': left_perimeter,
            'right_perimeter': right_perimeter,
            'left_compactness': left_compactness,
            'right_compactness': right_compactness,
            'left_eccentricity': left_eccentricity,
            'right_eccentricity': right_eccentricity,
            'shape_category': shape_category,
            'left_density': left_density,
            'right_density': right_density,
            'avg_density': avg_density,
            'density_category': density_category
        }
    
    def visualize_segmentation(self, image, segmentation):
        """
        Visualize the segmentation results.
        
        Args:
            image (numpy.ndarray): Input image
            segmentation (dict): Segmentation dictionary from segment_eyebrows
            
        Returns:
            numpy.ndarray: Image with visualized segmentation
        """
        if segmentation is None:
            return image
        
        # Create a copy of the image
        vis_image = image.copy()
        
        # Create a colored mask for visualization
        colored_mask = np.zeros_like(vis_image)
        
        # Left eyebrow in red
        colored_mask[segmentation['left_mask'] > 0] = [0, 0, 255]
        
        # Right eyebrow in blue
        colored_mask[segmentation['right_mask'] > 0] = [255, 0, 0]
        
        # Blend the mask with the original image
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        
        # Draw bounding boxes
        left_roi = segmentation['left_roi']
        right_roi = segmentation['right_roi']
        
        cv2.rectangle(vis_image, (left_roi[0], left_roi[1]), (left_roi[2], left_roi[3]), (0, 255, 0), 2)
        cv2.rectangle(vis_image, (right_roi[0], right_roi[1]), (right_roi[2], right_roi[3]), (0, 255, 0), 2)
        
        return vis_image
