#!/usr/bin/env python
"""
Main eyebrow analysis module.
This module integrates all eyebrow analysis components.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eyebrow_analysis.landmarks_no_dlib import EyebrowLandmarkDetector
from eyebrow_analysis.segmentation import EyebrowSegmentation
from eyebrow_analysis.color_texture import EyebrowColorTexture

class EyebrowAnalyzer:
    """Main class for comprehensive eyebrow analysis."""
    
    def __init__(self, model_path=None):
        """
        Initialize the eyebrow analyzer.
        
        Args:
            model_path (str, optional): Path to the facial landmark model.
        """
        self.landmark_detector = EyebrowLandmarkDetector(model_path)
        self.segmentation = EyebrowSegmentation()
        self.color_texture = EyebrowColorTexture()
    
    def analyze_image(self, image_path, trimap_path=None, visualize=False, output_dir=None):
        """
        Analyze eyebrows in a single image.
        
        Args:
            image_path (str): Path to the input image
            trimap_path (str, optional): Path to the trimap image
            visualize (bool): Whether to visualize the results
            output_dir (str, optional): Directory to save visualization results
            
        Returns:
            dict: Dictionary containing all analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Load trimap if provided
        trimap = None
        use_trimap = False
        if trimap_path and os.path.exists(trimap_path):
            trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
            use_trimap = True
        
        # Detect landmarks
        landmarks = self.landmark_detector.detect_landmarks(image)
        if landmarks is None:
            print(f"Error: No face detected in {image_path}")
            return None
        
        # Calculate landmark-based metrics
        symmetry = self.landmark_detector.calculate_eyebrow_symmetry(landmarks)
        arch = self.landmark_detector.calculate_eyebrow_arch(landmarks)
        thickness = self.landmark_detector.calculate_eyebrow_thickness(image, landmarks)
        length = self.landmark_detector.calculate_eyebrow_length(landmarks)
        
        # Segment eyebrows
        segmentation_result = self.segmentation.segment_eyebrows(
            image, landmarks, use_trimap=use_trimap, trimap=trimap
        )
        
        # Analyze shape
        shape_analysis = self.segmentation.analyze_shape(segmentation_result)
        
        # Analyze color
        color_analysis = self.color_texture.analyze_color(image, segmentation_result)
        
        # Analyze texture
        texture_analysis = self.color_texture.analyze_texture(image, segmentation_result)
        
        # Combine all results
        results = {
            'image_path': image_path,
            'landmarks': {
                'symmetry': symmetry,
                'arch': arch,
                'thickness': thickness,
                'length': length
            },
            'shape': shape_analysis,
            'color': color_analysis,
            'texture': texture_analysis
        }
        
        # Visualize if requested
        if visualize:
            # Create visualization directory if it doesn't exist
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(image_path), 'visualization')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Visualize landmarks
            landmarks_vis = self.landmark_detector.visualize_landmarks(image, landmarks)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_landmarks.jpg"), landmarks_vis)
            
            # Visualize segmentation
            segmentation_vis = self.segmentation.visualize_segmentation(image, segmentation_result)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmentation.jpg"), segmentation_vis)
            
            # Visualize color analysis
            if color_analysis:
                color_fig = self.color_texture.visualize_color(color_analysis)
                if color_fig:
                    color_fig.savefig(os.path.join(output_dir, f"{base_name}_color.jpg"))
                    plt.close(color_fig)
            
            # Visualize texture analysis
            if texture_analysis:
                texture_fig = self.color_texture.visualize_texture(texture_analysis)
                if texture_fig:
                    texture_fig.savefig(os.path.join(output_dir, f"{base_name}_texture.jpg"))
                    plt.close(texture_fig)
            
            # Create a combined visualization
            combined_vis = self.create_combined_visualization(
                image, landmarks, segmentation_result, 
                color_analysis, texture_analysis, 
                symmetry, arch, thickness, length, shape_analysis
            )
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_combined.jpg"), combined_vis)
        
        return results
    
    def analyze_batch(self, image_dir, trimap_dir=None, output_dir=None, visualize=False):
        """
        Analyze eyebrows in a batch of images.
        
        Args:
            image_dir (str): Directory containing input images
            trimap_dir (str, optional): Directory containing trimap images
            output_dir (str, optional): Directory to save results
            visualize (bool): Whether to visualize the results
            
        Returns:
            pandas.DataFrame: DataFrame containing all analysis results
        """
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(image_dir, 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Initialize results list
        all_results = []
        
        # Process each image
        for image_file in tqdm(image_files, desc="Analyzing images"):
            image_path = os.path.join(image_dir, image_file)
            
            # Find corresponding trimap if available
            trimap_path = None
            if trimap_dir:
                # Try different possible trimap naming conventions
                base_name = os.path.splitext(image_file)[0]
                possible_trimap_names = [
                    f"{base_name}_trimap.jpg",
                    f"{base_name}_trimap.png",
                    f"{base_name}.jpg",
                    f"{base_name}.png"
                ]
                
                for trimap_name in possible_trimap_names:
                    potential_path = os.path.join(trimap_dir, trimap_name)
                    if os.path.exists(potential_path):
                        trimap_path = potential_path
                        break
            
            # Analyze image
            results = self.analyze_image(
                image_path, trimap_path, 
                visualize=visualize, 
                output_dir=os.path.join(output_dir, 'visualization') if visualize else None
            )
            
            if results:
                # Flatten the results dictionary for easier DataFrame creation
                flat_results = self.flatten_dict(results)
                all_results.append(flat_results)
        
        # Create DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Save results to CSV
            df.to_csv(os.path.join(output_dir, 'eyebrow_analysis_results.csv'), index=False)
            
            # Create summary visualizations
            if len(df) > 1:
                self.create_summary_visualizations(df, output_dir)
            
            return df
        else:
            print("No valid results to analyze.")
            return None
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten a nested dictionary.
        
        Args:
            d (dict): Dictionary to flatten
            parent_key (str): Parent key for nested dictionaries
            sep (str): Separator between keys
            
        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                # Skip large arrays and non-scalar values
                if not isinstance(v, (np.ndarray, list)) or (isinstance(v, (np.ndarray, list)) and len(v) <= 10):
                    items.append((new_key, v))
        
        return dict(items)
    
    def create_combined_visualization(self, image, landmarks, segmentation, 
                                     color_analysis, texture_analysis,
                                     symmetry, arch, thickness, length, shape_analysis):
        """
        Create a combined visualization of all analysis results.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (dict): Landmarks dictionary
            segmentation (dict): Segmentation dictionary
            color_analysis (dict): Color analysis dictionary
            texture_analysis (dict): Texture analysis dictionary
            symmetry (float): Symmetry score
            arch (dict): Arch analysis dictionary
            thickness (dict): Thickness analysis dictionary
            length (dict): Length analysis dictionary
            shape_analysis (dict): Shape analysis dictionary
            
        Returns:
            numpy.ndarray: Combined visualization image
        """
        # Create a larger canvas
        h, w = image.shape[:2]
        canvas = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255
        
        # Add original image
        canvas[:h, :w] = image
        
        # Add landmarks visualization
        landmarks_vis = self.landmark_detector.visualize_landmarks(image, landmarks, show_metrics=False)
        canvas[:h, w:w*2] = landmarks_vis
        
        # Add segmentation visualization
        if segmentation:
            segmentation_vis = self.segmentation.visualize_segmentation(image, segmentation)
            canvas[h:h*2, :w] = segmentation_vis
        
        # Create metrics visualization
        metrics_vis = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Add metrics text
        metrics = [
            f"Symmetry: {symmetry:.2f}",
            f"Arch Type: {arch['arch_type']}",
            f"Thickness: {thickness['thickness_category']}",
            f"Length: {length['length_category']}",
            f"Shape: {shape_analysis['shape_category'] if shape_analysis else 'N/A'}",
            f"Density: {shape_analysis['density_category'] if shape_analysis else 'N/A'}",
            f"Color: {color_analysis['color_name'] if color_analysis else 'N/A'}",
            f"Tone: {color_analysis['tone'] if color_analysis else 'N/A'}",
            f"Texture: {texture_analysis['texture_category'] if texture_analysis else 'N/A'}",
            f"Uniformity: {texture_analysis['uniformity_category'] if texture_analysis else 'N/A'}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(metrics_vis, metric, (20, 40 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add color swatch if available
        if color_analysis and 'mean_rgb' in color_analysis:
            mean_rgb = color_analysis['mean_rgb']
            r, g, b = mean_rgb
            color_rect = np.ones((100, 100, 3), dtype=np.uint8)
            color_rect[:, :, 0] = b  # OpenCV uses BGR
            color_rect[:, :, 1] = g
            color_rect[:, :, 2] = r
            
            x_offset = w - 120
            y_offset = h - 120
            metrics_vis[y_offset:y_offset+100, x_offset:x_offset+100] = color_rect
        
        # Add metrics visualization to canvas
        canvas[h:h*2, w:w*2] = metrics_vis
        
        # Add title
        cv2.putText(canvas, "Eyebrow Analysis", (w//2, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return canvas
    
    def create_summary_visualizations(self, df, output_dir):
        """
        Create summary visualizations for batch analysis.
        
        Args:
            df (pandas.DataFrame): DataFrame with analysis results
            output_dir (str): Directory to save visualizations
        """
        # Create visualizations directory
        vis_dir = os.path.join(output_dir, 'summary_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create color distribution visualization
        if 'color_color_name' in df.columns:
            color_counts = df['color_color_name'].value_counts()
            
            # Pie chart
            fig = px.pie(
                values=color_counts.values,
                names=color_counts.index,
                title="Eyebrow Color Distribution"
            )
            fig.write_html(os.path.join(vis_dir, 'color_distribution_pie.html'))
            
            # Bar chart
            fig = px.bar(
                x=color_counts.index,
                y=color_counts.values,
                title="Eyebrow Color Distribution",
                labels={'x': 'Color', 'y': 'Count'}
            )
            fig.write_html(os.path.join(vis_dir, 'color_distribution_bar.html'))
        
        # Create texture distribution visualization
        if 'texture_texture_category' in df.columns:
            texture_counts = df['texture_texture_category'].value_counts()
            
            fig = px.pie(
                values=texture_counts.values,
                names=texture_counts.index,
                title="Eyebrow Texture Distribution"
            )
            fig.write_html(os.path.join(vis_dir, 'texture_distribution_pie.html'))
        
        # Create shape distribution visualization
        if 'shape_shape_category' in df.columns:
            shape_counts = df['shape_shape_category'].value_counts()
            
            fig = px.pie(
                values=shape_counts.values,
                names=shape_counts.index,
                title="Eyebrow Shape Distribution"
            )
            fig.write_html(os.path.join(vis_dir, 'shape_distribution_pie.html'))
        
        # Create symmetry histogram
        if 'landmarks_symmetry' in df.columns:
            fig = px.histogram(
                df, x='landmarks_symmetry',
                title="Eyebrow Symmetry Distribution",
                labels={'landmarks_symmetry': 'Symmetry Score', 'count': 'Count'},
                nbins=20
            )
            fig.write_html(os.path.join(vis_dir, 'symmetry_histogram.html'))
        
        # Create correlation heatmap for numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 1:
            corr = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr,
                title="Correlation Between Eyebrow Features",
                labels=dict(color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.write_html(os.path.join(vis_dir, 'correlation_heatmap.html'))
        
        # Create a dashboard with multiple visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Eyebrow Color Distribution", 
                "Eyebrow Shape Distribution",
                "Eyebrow Texture Distribution", 
                "Eyebrow Symmetry Distribution"
            )
        )
        
        # Add color distribution
        if 'color_color_name' in df.columns:
            color_counts = df['color_color_name'].value_counts()
            fig.add_trace(
                go.Pie(labels=color_counts.index, values=color_counts.values),
                row=1, col=1
            )
        
        # Add shape distribution
        if 'shape_shape_category' in df.columns:
            shape_counts = df['shape_shape_category'].value_counts()
            fig.add_trace(
                go.Pie(labels=shape_counts.index, values=shape_counts.values),
                row=1, col=2
            )
        
        # Add texture distribution
        if 'texture_texture_category' in df.columns:
            texture_counts = df['texture_texture_category'].value_counts()
            fig.add_trace(
                go.Pie(labels=texture_counts.index, values=texture_counts.values),
                row=2, col=1
            )
        
        # Add symmetry histogram
        if 'landmarks_symmetry' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['landmarks_symmetry']),
                row=2, col=2
            )
        
        fig.update_layout(title_text="Eyebrow Analysis Dashboard")
        fig.write_html(os.path.join(vis_dir, 'eyebrow_dashboard.html'))
