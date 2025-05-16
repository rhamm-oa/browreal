#!/usr/bin/env python
"""
Eyebrow color and texture analysis module.
This module provides functions to analyze eyebrow color and texture properties.
"""

import cv2
import numpy as np
from skimage import feature, color
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EyebrowColorTexture:
    """Class for analyzing eyebrow color and texture."""
    
    def __init__(self):
        """Initialize the eyebrow color and texture analyzer."""
        # Define color name mappings
        self.color_names = {
            'black': np.array([0, 0, 0]),
            'dark_brown': np.array([51, 25, 0]),
            'medium_brown': np.array([102, 51, 0]),
            'light_brown': np.array([153, 102, 51]),
            'blonde': np.array([204, 153, 102]),
            'red': np.array([153, 51, 0]),
            'auburn': np.array([102, 51, 0]),
            'gray': np.array([128, 128, 128])
        }
    
    def analyze_color(self, image, segmentation):
        """Analyze the color of the eyebrows.
        
        Args:
            image (numpy.ndarray): Input image
            segmentation (dict): Segmentation dictionary from EyebrowSegmentation
            
        Returns:
            dict: Dictionary containing color metrics
        """
        if segmentation is None:
            return None
            
        # Get masks and ROIs
        combined_mask = segmentation['combined_mask']
        
        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract pixels within the mask
        mask_indices = np.where(combined_mask > 0)
        pixels = rgb_image[mask_indices]
        
        if len(pixels) == 0:
            return None
            
        # Use K-means to find dominant colors
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors and their percentages
        dominant_colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Count occurrences of each label
        label_counts = np.bincount(labels)
        
        # Calculate percentages
        percentages = label_counts / len(labels)
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        dominant_colors = dominant_colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        # Convert to LAB for better color analysis
        lab_colors = []
        for rgb in dominant_colors:
            rgb_normalized = rgb / 255.0
            rgb_reshaped = rgb_normalized.reshape(1, 1, 3)
            lab = color.rgb2lab(rgb_reshaped)[0][0]
            lab_colors.append(lab)
        
        # Get the main color (highest percentage)
        main_color = dominant_colors[0]
        main_lab = lab_colors[0]
        
        # Find closest color name
        color_name = self._get_color_name(main_color)
        
        # Calculate brightness (L in LAB)
        brightness = main_lab[0]
        
        # Categorize brightness
        if brightness < 30:
            brightness_category = "dark"
        elif brightness < 60:
            brightness_category = "medium"
        else:
            brightness_category = "light"
            
        # Calculate tone based on a/b values in LAB
        a, b = main_lab[1], main_lab[2]
        
        # Determine tone
        if abs(a) < 10 and abs(b) < 10:
            tone = "neutral"
        elif a > 0 and b > 0:
            tone = "warm"
        elif a < 0 and b < 0:
            tone = "cool"
        elif a > 0 and b < 0:
            tone = "reddish"
        else:
            tone = "greenish"
            
        # Calculate color consistency (standard deviation of colors)
        color_std = np.std(pixels, axis=0).mean()
        color_consistency = 1.0 - min(color_std / 128.0, 1.0)  # Normalize to 0-1
        
        # Create color analysis dictionary
        color_analysis = {
            'dominant_colors': dominant_colors.tolist(),
            'dominant_percentages': percentages.tolist(),
            'lab_colors': [c.tolist() for c in lab_colors],
            'color_name': color_name,
            'brightness': float(brightness),
            'brightness_category': brightness_category,
            'tone': tone,
            'color_consistency': float(color_consistency)
        }
        
        return color_analysis
    
    def _get_color_name(self, rgb_color):
        """Get the closest color name for an RGB color.
        
        Args:
            rgb_color (numpy.ndarray): RGB color array
            
        Returns:
            str: Color name
        """
        min_distance = float('inf')
        closest_color = None
        
        for name, color_rgb in self.color_names.items():
            distance = np.sqrt(np.sum((rgb_color - color_rgb) ** 2))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
                
        return closest_color
    
    def analyze_texture(self, image, segmentation):
        """Analyze the texture of the eyebrows.
        
        Args:
            image (numpy.ndarray): Input image
            segmentation (dict): Segmentation dictionary from EyebrowSegmentation
            
        Returns:
            dict: Dictionary containing texture metrics
        """
        if segmentation is None:
            return None
            
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get masks and ROIs
        left_mask = segmentation['left_mask'] > 0
        right_mask = segmentation['right_mask'] > 0
        left_roi = segmentation['left_roi']
        right_roi = segmentation['right_roi']
        
        # Extract ROIs
        left_gray_roi = gray[left_roi[1]:left_roi[3], left_roi[0]:left_roi[2]]
        right_gray_roi = gray[right_roi[1]:right_roi[3], right_roi[0]:right_roi[2]]
        left_mask_roi = left_mask[left_roi[1]:left_roi[3], left_roi[0]:left_roi[2]]
        right_mask_roi = right_mask[right_roi[1]:right_roi[3], right_roi[0]:right_roi[2]]
        
        # Calculate texture features for both eyebrows
        left_features = self.calculate_texture_features(left_gray_roi, left_mask_roi)
        right_features = self.calculate_texture_features(right_gray_roi, right_mask_roi)
        
        if left_features is None and right_features is None:
            return None
        
        # Combine features
        if left_features is not None and right_features is not None:
            combined_features = {
                'contrast': (left_features['contrast'] + right_features['contrast']) / 2,
                'dissimilarity': (left_features['dissimilarity'] + right_features['dissimilarity']) / 2,
                'homogeneity': (left_features['homogeneity'] + right_features['homogeneity']) / 2,
                'energy': (left_features['energy'] + right_features['energy']) / 2,
                'correlation': (left_features['correlation'] + right_features['correlation']) / 2,
                'entropy': (left_features['entropy'] + right_features['entropy']) / 2,
                'left_lbp_hist': left_features['lbp_hist'],
                'right_lbp_hist': right_features['lbp_hist']
            }
        elif left_features is not None:
            combined_features = {
                'contrast': left_features['contrast'],
                'dissimilarity': left_features['dissimilarity'],
                'homogeneity': left_features['homogeneity'],
                'energy': left_features['energy'],
                'correlation': left_features['correlation'],
                'entropy': left_features['entropy'],
                'left_lbp_hist': left_features['lbp_hist'],
                'right_lbp_hist': None
            }
        else:
            combined_features = {
                'contrast': right_features['contrast'],
                'dissimilarity': right_features['dissimilarity'],
                'homogeneity': right_features['homogeneity'],
                'energy': right_features['energy'],
                'correlation': right_features['correlation'],
                'entropy': right_features['entropy'],
                'left_lbp_hist': None,
                'right_lbp_hist': right_features['lbp_hist']
            }
        
        # Calculate texture score and categories
        texture_score = (combined_features['contrast'] + combined_features['dissimilarity']) - \
                      (combined_features['homogeneity'] + combined_features['energy'])
        
        # Categorize texture
        if texture_score > 1.5:
            texture_category = "rough"
        elif texture_score > 0.5:
            texture_category = "medium"
        else:
            texture_category = "smooth"
        
        # Calculate uniformity based on correlation
        if combined_features['correlation'] > 0.8:
            uniformity_category = "highly uniform"
        elif combined_features['correlation'] > 0.6:
            uniformity_category = "moderately uniform"
        else:
            uniformity_category = "non-uniform"
        
        combined_features['texture_score'] = texture_score
        combined_features['texture_category'] = texture_category
        combined_features['uniformity_category'] = uniformity_category
        
        return combined_features

    def calculate_texture_features(self, gray_roi, mask_roi):
        """Calculate texture features for a given ROI.
        
        Args:
            gray_roi (numpy.ndarray): Grayscale ROI image
            mask_roi (numpy.ndarray): Binary mask for the ROI
            
        Returns:
            dict: Dictionary containing texture features
        """
        if gray_roi is None or mask_roi is None or gray_roi.size == 0 or mask_roi.size == 0:
            return None
            
        # Apply mask
        masked_roi = gray_roi.copy()
        masked_roi[~mask_roi] = 0
        
        # Calculate GLCM features
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = feature.graycomatrix(masked_roi, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract features from GLCM
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        
        # Calculate LBP histogram
        lbp = feature.local_binary_pattern(masked_roi, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp[mask_roi], bins=np.arange(0, 11), density=True)
        
        # Calculate entropy
        hist, _ = np.histogram(masked_roi[mask_roi], bins=256, density=True)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        return {
            'contrast': float(contrast),
            'dissimilarity': float(dissimilarity),
            'homogeneity': float(homogeneity),
            'energy': float(energy),
            'correlation': float(correlation),
            'entropy': float(entropy),
            'lbp_hist': lbp_hist.tolist()
        }
        
    def visualize_color(self, color_analysis, figsize=(10, 6)):
        """Visualize the color analysis results using interactive 3D LAB visualization.
        
        Args:
            color_analysis (dict): Color analysis dictionary from analyze_color
            figsize (tuple): Figure size (not used in interactive visualization)
            
        Returns:
            plotly.graph_objects.Figure: Interactive color visualization
        """
        if color_analysis is None:
            return None
            
        # Convert RGB colors to LAB
        from skimage import color as skcolor
        lab_colors = [skcolor.rgb2lab(np.array([rgb/255.0 for rgb in c]).reshape(1, 1, 3))[0][0] 
                     for c in color_analysis['dominant_colors']]
        
        # Create figure with subplots with correct specs for table
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}],
                   [None, {'type': 'table'}]],
            subplot_titles=('3D LAB Color Space', 'Dominant Colors', 'Color Information')
        )
        
        # Add 3D scatter plot of LAB colors
        L, a, b = zip(*lab_colors)
        fig.add_trace(
            go.Scatter3d(
                x=a, y=b, z=L,
                mode='markers',
                marker=dict(
                    size=8,
                    color=[f'rgb({int(r)},{int(g)},{int(b)})' 
                           for r,g,b in color_analysis['dominant_colors']],
                    opacity=0.8
                ),
                text=[f'L:{l:.1f}, a:{a:.1f}, b:{b:.1f}' for l,a,b in zip(L,a,b)],
                hoverinfo='text',
                name='LAB Colors'
            ),
            row=1, col=1
        )
        
        # Update 3D layout
        fig.update_scenes(
            xaxis_title='a* (Green to Red)',
            yaxis_title='b* (Blue to Yellow)',
            zaxis_title='L* (Black to White)',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        # Add dominant colors bar
        y_positions = np.cumsum([0] + [p*100 for p in color_analysis['dominant_percentages'][:-1]])
        for color, height, y_pos in zip(color_analysis['dominant_colors'],
                                      [p*100 for p in color_analysis['dominant_percentages']],
                                      y_positions):
            fig.add_trace(
                go.Bar(
                    x=[100],
                    y=[height],
                    marker_color=f'rgb({int(color[0])},{int(color[1])},{int(color[2])})',
                    name=f'{height:.1f}%',
                    orientation='v',
                    base=y_pos
                ),
                row=1, col=2
            )
        
        # Add color information
        color_info = [
            f"Color Name: {color_analysis['color_name']}",
            f"Tone: {color_analysis['tone']}",
            f"Brightness: {color_analysis['brightness_category']}",
            f"Consistency: {color_analysis['color_consistency']:.2f}"
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Color Information']),
                cells=dict(values=[color_info]),
                columnwidth=[400]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title=dict(
                text='Color Analysis',
                x=0.5,
                font=dict(size=24)
            ),
            template='plotly_white'
        )
        
        return fig

    def visualize_texture(self, texture_analysis):
        """Visualize the texture analysis results using interactive visualization.
        
        Args:
            texture_analysis (dict): Texture analysis dictionary from analyze_texture
            
        Returns:
            plotly.graph_objects.Figure: Interactive texture visualization
        """
        if texture_analysis is None:
            return None
            
        # Create figure with subplots with correct specs for table
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'table'}]],
            subplot_titles=('Texture Metrics', 'LBP Histogram', 'GLCM Matrix', 'Information')
        )
        
        # Texture Metrics Bar Chart
        metrics = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        values = [texture_analysis[metric] for metric in metrics]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker=dict(
                    color=values,
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=1, color='rgb(50, 50, 50)')
                ),
                hovertemplate='%{x}: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # LBP Histogram
        if texture_analysis['left_lbp_hist'] is not None:
            lbp_hist = texture_analysis['left_lbp_hist']
        elif texture_analysis['right_lbp_hist'] is not None:
            lbp_hist = texture_analysis['right_lbp_hist']
        else:
            lbp_hist = None
            
        if lbp_hist is not None:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(lbp_hist))),
                    y=lbp_hist,
                    marker=dict(
                        color=lbp_hist,
                        colorscale='Viridis',
                        opacity=0.8,
                        line=dict(width=1, color='rgb(50, 50, 50)')
                    ),
                    name='LBP Histogram'
                ),
                row=1, col=2
            )
        
        # GLCM Matrix Heatmap
        glcm_matrix = np.array([
            [texture_analysis['contrast'], texture_analysis['dissimilarity']],
            [texture_analysis['homogeneity'], texture_analysis['energy']]
        ])
        
        fig.add_trace(
            go.Heatmap(
                z=glcm_matrix,
                x=['Contrast', 'Dissimilarity'],
                y=['Homogeneity', 'Energy'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Value')
            ),
            row=2, col=1
        )
        
        # Add texture information table
        texture_info = [
            f"Texture Category: {texture_analysis['texture_category']}",
            f"Texture Score: {texture_analysis['texture_score']:.2f}",
            f"Uniformity: {texture_analysis['uniformity_category']}",
            f"Correlation: {texture_analysis['correlation']:.2f}"
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Texture Information']),
                cells=dict(values=[texture_info]),
                columnwidth=[400]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title=dict(
                text='Texture Analysis',
                x=0.5,
                font=dict(size=24)
            ),
            template='plotly_white'
        )
        
        return fig
