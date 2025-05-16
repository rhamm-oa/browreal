#!/usr/bin/env python
"""
Eyebrow landmarks detection and analysis module without dlib dependency.
This module provides functions to detect facial landmarks with a focus on eyebrows using MediaPipe.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import torch
from scipy.spatial import distance

class EyebrowLandmarkDetector:
    """Class for detecting and analyzing eyebrow landmarks without dlib dependency."""
    
    def __init__(self):
        """Initialize the eyebrow landmark detector using MediaPipe."""
        # Initialize MediaPipe face mesh for detailed landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # MediaPipe face detection for face bounding box
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # MediaPipe eyebrow indices
        # Left eyebrow indices
        self.mp_left_eyebrow = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        # Right eyebrow indices
        self.mp_right_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        
        # Define additional eyebrow points for better coverage
        self.mp_left_eyebrow_extended = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 
                                        380, 381, 382, 362, 398, 384, 385, 386, 387, 388]
        self.mp_right_eyebrow_extended = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                                         159, 160, 161, 146, 176, 163, 144, 145, 153, 154]

    def detect_landmarks(self, image):
        """
        Detect facial landmarks in the given image using MediaPipe.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Dictionary containing various landmark points and metadata
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Detect face for bounding box
        face_detection_results = self.face_detection.process(image_rgb)
        if not face_detection_results.detections:
            return None
        
        # Get face bounding box
        detection = face_detection_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_rect = (
            int(bbox.xmin * w),
            int(bbox.ymin * h),
            int((bbox.xmin + bbox.width) * w),
            int((bbox.ymin + bbox.height) * h)
        )
        
        # Get MediaPipe face mesh landmarks
        face_mesh_results = self.face_mesh.process(image_rgb)
        if not face_mesh_results.multi_face_landmarks:
            return None
        
        # Extract all landmarks
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        mp_landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            mp_landmarks.append((x, y))
        
        # Extract eyebrow landmarks
        left_eyebrow = np.array([mp_landmarks[i] for i in self.mp_left_eyebrow if i < len(mp_landmarks)])
        right_eyebrow = np.array([mp_landmarks[i] for i in self.mp_right_eyebrow if i < len(mp_landmarks)])
        
        # Extract extended eyebrow landmarks
        left_eyebrow_extended = np.array([mp_landmarks[i] for i in self.mp_left_eyebrow_extended if i < len(mp_landmarks)])
        right_eyebrow_extended = np.array([mp_landmarks[i] for i in self.mp_right_eyebrow_extended if i < len(mp_landmarks)])
        
        # Create a 68-point compatible format for compatibility with existing code
        # Map MediaPipe landmarks to approximate dlib 68-point model
        # For eyebrows: left eyebrow is points 17-21, right eyebrow is points 22-26
        landmarks_68 = np.zeros((68, 2), dtype=np.int32)
        
        # Map left eyebrow (points 17-21)
        if len(left_eyebrow) >= 5:
            # Use 5 evenly spaced points from the left eyebrow
            indices = np.linspace(0, len(left_eyebrow) - 1, 5).astype(int)
            landmarks_68[17:22] = left_eyebrow[indices]
        
        # Map right eyebrow (points 22-26)
        if len(right_eyebrow) >= 5:
            # Use 5 evenly spaced points from the right eyebrow
            indices = np.linspace(0, len(right_eyebrow) - 1, 5).astype(int)
            landmarks_68[22:27] = right_eyebrow[indices]
        
        return {
            'all_landmarks': landmarks_68,
            'left_eyebrow': landmarks_68[17:22] if len(left_eyebrow) >= 5 else left_eyebrow,
            'right_eyebrow': landmarks_68[22:27] if len(right_eyebrow) >= 5 else right_eyebrow,
            'mp_landmarks': mp_landmarks,
            'mp_left_eyebrow': left_eyebrow,
            'mp_right_eyebrow': right_eyebrow,
            'mp_left_eyebrow_extended': left_eyebrow_extended,
            'mp_right_eyebrow_extended': right_eyebrow_extended,
            'face_rect': face_rect,
            'fa_landmarks': None  # No face-alignment landmarks
        }
    
    def calculate_eyebrow_symmetry(self, landmarks):
        """
        Calculate the symmetry between left and right eyebrows.
        
        Args:
            landmarks (dict): Landmarks dictionary from detect_landmarks
            
        Returns:
            float: Symmetry score (0-1, where 1 is perfect symmetry)
        """
        if landmarks is None or 'mp_left_eyebrow' not in landmarks or 'mp_right_eyebrow' not in landmarks:
            return 0.0
        
        left_eyebrow = landmarks['mp_left_eyebrow']
        right_eyebrow = landmarks['mp_right_eyebrow']
        
        # Ensure we have the same number of points for comparison
        min_points = min(len(left_eyebrow), len(right_eyebrow))
        if min_points < 3:
            return 0.0
            
        # Use the same number of points for both eyebrows
        left_eyebrow = left_eyebrow[:min_points]
        right_eyebrow = right_eyebrow[:min_points]
        
        # Mirror the right eyebrow for comparison
        face_width = landmarks['face_rect'][2] - landmarks['face_rect'][0]
        mirrored_right = np.array([(face_width - x, y) for x, y in right_eyebrow])
        
        # Calculate the average distance between corresponding points
        distances = []
        for i in range(min_points):
            dist = distance.euclidean(left_eyebrow[i], mirrored_right[i])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Normalize by face width to get a score between 0 and 1
        symmetry_score = max(0, 1 - (avg_distance / (face_width * 0.2)))
        
        return symmetry_score
    
    def calculate_eyebrow_arch(self, landmarks):
        """
        Calculate the arch of each eyebrow.
        
        Args:
            landmarks (dict): Landmarks dictionary from detect_landmarks
            
        Returns:
            dict: Dictionary containing arch metrics for both eyebrows
        """
        if landmarks is None:
            return {'left_arch': 0, 'right_arch': 0, 'arch_type': 'unknown'}
        
        left_eyebrow = landmarks['mp_left_eyebrow']
        right_eyebrow = landmarks['mp_right_eyebrow']
        
        # Calculate arch for left eyebrow
        if len(left_eyebrow) >= 5:
            left_start, left_end = left_eyebrow[0], left_eyebrow[-1]
            left_middle = left_eyebrow[len(left_eyebrow) // 2]  # Middle point
            
            # Create a straight line from start to end
            left_line = np.array([left_start, left_end])
            left_line_vector = left_end - left_start
            left_line_length = np.linalg.norm(left_line_vector)
            left_line_unit = left_line_vector / left_line_length if left_line_length > 0 else np.array([0, 0])
            
            # Calculate the perpendicular distance from middle point to the line
            left_middle_vector = left_middle - left_start
            left_projection = np.dot(left_middle_vector, left_line_unit)
            left_projected_point = left_start + left_projection * left_line_unit
            left_distance = np.linalg.norm(left_middle - left_projected_point)
            
            # Normalize by the length of the eyebrow
            left_arch = left_distance / left_line_length if left_line_length > 0 else 0
        else:
            left_arch = 0
        
        # Calculate arch for right eyebrow
        if len(right_eyebrow) >= 5:
            right_start, right_end = right_eyebrow[0], right_eyebrow[-1]
            right_middle = right_eyebrow[len(right_eyebrow) // 2]  # Middle point
            
            # Create a straight line from start to end
            right_line = np.array([right_start, right_end])
            right_line_vector = right_end - right_start
            right_line_length = np.linalg.norm(right_line_vector)
            right_line_unit = right_line_vector / right_line_length if right_line_length > 0 else np.array([0, 0])
            
            # Calculate the perpendicular distance from middle point to the line
            right_middle_vector = right_middle - right_start
            right_projection = np.dot(right_middle_vector, right_line_unit)
            right_projected_point = right_start + right_projection * right_line_unit
            right_distance = np.linalg.norm(right_middle - right_projected_point)
            
            # Normalize by the length of the eyebrow
            right_arch = right_distance / right_line_length if right_line_length > 0 else 0
        else:
            right_arch = 0
        
        # Determine arch type based on the average arch value
        avg_arch = (left_arch + right_arch) / 2
        
        if avg_arch < 0.1:
            arch_type = "straight"
        elif avg_arch < 0.15:
            arch_type = "slightly arched"
        elif avg_arch < 0.2:
            arch_type = "moderately arched"
        else:
            arch_type = "highly arched"
        
        return {
            'left_arch': left_arch,
            'right_arch': right_arch,
            'avg_arch': avg_arch,
            'arch_type': arch_type
        }
    
    def calculate_eyebrow_thickness(self, image, landmarks):
        """
        Calculate the thickness of the eyebrows.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (dict): Landmarks dictionary from detect_landmarks
            
        Returns:
            dict: Dictionary containing thickness metrics for both eyebrows
        """
        if landmarks is None:
            return {'left_thickness': 0, 'right_thickness': 0, 'thickness_category': 'unknown'}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to isolate eyebrows
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        left_eyebrow = landmarks['mp_left_eyebrow_extended']
        right_eyebrow = landmarks['mp_right_eyebrow_extended']
        
        # Function to measure thickness at a point
        def measure_thickness(point, direction):
            x, y = int(point[0]), int(point[1])
            if x < 0 or y < 0 or x >= thresh.shape[1] or y >= thresh.shape[0]:
                return 0
                
            # Search in the direction (up or down)
            thickness = 0
            step = -1 if direction == 'up' else 1
            
            for i in range(1, 30):  # Limit search to 30 pixels
                ny = y + (i * step)
                if ny < 0 or ny >= thresh.shape[0]:
                    break
                if thresh[ny, x] > 0:  # If pixel is part of eyebrow
                    thickness += 1
                else:
                    break
            
            return thickness
        
        # Measure thickness at multiple points along each eyebrow
        left_thicknesses = []
        for point in left_eyebrow:
            up_thickness = measure_thickness(point, 'up')
            down_thickness = measure_thickness(point, 'down')
            left_thicknesses.append(up_thickness + down_thickness)
        
        right_thicknesses = []
        for point in right_eyebrow:
            up_thickness = measure_thickness(point, 'up')
            down_thickness = measure_thickness(point, 'down')
            right_thicknesses.append(up_thickness + down_thickness)
        
        # Calculate average thickness
        left_avg_thickness = np.mean(left_thicknesses) if left_thicknesses else 0
        right_avg_thickness = np.mean(right_thicknesses) if right_thicknesses else 0
        avg_thickness = (left_avg_thickness + right_avg_thickness) / 2
        
        # Categorize thickness
        if avg_thickness < 5:
            thickness_category = "thin"
        elif avg_thickness < 10:
            thickness_category = "medium"
        else:
            thickness_category = "thick"
        
        return {
            'left_thickness': left_avg_thickness,
            'right_thickness': right_avg_thickness,
            'avg_thickness': avg_thickness,
            'thickness_category': thickness_category
        }
    
    def calculate_eyebrow_length(self, landmarks):
        """
        Calculate the length of each eyebrow.
        
        Args:
            landmarks (dict): Landmarks dictionary from detect_landmarks
            
        Returns:
            dict: Dictionary containing length metrics for both eyebrows
        """
        if landmarks is None:
            return {'left_length': 0, 'right_length': 0, 'length_category': 'unknown'}
        
        left_eyebrow = landmarks['mp_left_eyebrow']
        right_eyebrow = landmarks['mp_right_eyebrow']
        
        # Calculate length as the sum of segments between consecutive points
        left_length = 0
        for i in range(len(left_eyebrow) - 1):
            left_length += distance.euclidean(left_eyebrow[i], left_eyebrow[i+1])
        
        right_length = 0
        for i in range(len(right_eyebrow) - 1):
            right_length += distance.euclidean(right_eyebrow[i], right_eyebrow[i+1])
        
        # Calculate average length
        avg_length = (left_length + right_length) / 2
        
        # Normalize by face width
        face_width = landmarks['face_rect'][2] - landmarks['face_rect'][0]
        normalized_length = avg_length / face_width if face_width > 0 else 0
        
        # Categorize length
        if normalized_length < 0.25:
            length_category = "short"
        elif normalized_length < 0.35:
            length_category = "medium"
        else:
            length_category = "long"
        
        return {
            'left_length': left_length,
            'right_length': right_length,
            'avg_length': avg_length,
            'normalized_length': normalized_length,
            'length_category': length_category
        }
    
    def visualize_landmarks(self, image, landmarks, show_metrics=True):
        """
        Visualize the detected landmarks and metrics on the image.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (dict): Landmarks dictionary from detect_landmarks
            show_metrics (bool): Whether to show metrics on the image
            
        Returns:
            numpy.ndarray: Image with visualized landmarks and metrics
        """
        if landmarks is None:
            return image
        
        # Create a copy of the image
        vis_image = image.copy()
        
        # Draw face rectangle
        face_rect = landmarks['face_rect']
        cv2.rectangle(vis_image, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), 2)
        
        # Draw MediaPipe landmarks
        mp_landmarks = landmarks['mp_landmarks']
        for i, (x, y) in enumerate(mp_landmarks):
            # Draw only facial contour landmarks to avoid cluttering
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                cv2.circle(vis_image, (int(x), int(y)), 1, (0, 0, 255), -1)
        
        # Draw eyebrow landmarks with different color
        for x, y in landmarks['mp_left_eyebrow']:
            cv2.circle(vis_image, (int(x), int(y)), 2, (255, 0, 0), -1)
        
        for x, y in landmarks['mp_right_eyebrow']:
            cv2.circle(vis_image, (int(x), int(y)), 2, (255, 0, 0), -1)
        
        # Show metrics if requested
        if show_metrics:
            # Calculate metrics
            symmetry = self.calculate_eyebrow_symmetry(landmarks)
            arch = self.calculate_eyebrow_arch(landmarks)
            thickness = self.calculate_eyebrow_thickness(image, landmarks)
            length = self.calculate_eyebrow_length(landmarks)
            
            # Display metrics on the image
            metrics_text = [
                f"Symmetry: {symmetry:.2f}",
                f"Arch Type: {arch['arch_type']}",
                f"Thickness: {thickness['thickness_category']}",
                f"Length: {length['length_category']}"
            ]
            
            for i, text in enumerate(metrics_text):
                cv2.putText(vis_image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_image
