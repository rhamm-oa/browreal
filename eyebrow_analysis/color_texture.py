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
        
    def visualize_color(self, color_analysis, figsize=(10, 6), image_name="Eyebrow"):
        """Visualize the color analysis results using interactive 3D LAB visualization.
        
        Args:
            color_analysis (dict): Color analysis dictionary from analyze_color
            figsize (tuple): Figure size (not used in interactive visualization)
            image_name (str): Name of the image for the title
            
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
        
        # Add LAB color space boundaries for reference (gamut)
        # Create a mesh grid for the LAB color space boundaries
        a_range = np.linspace(-100, 100, 10)
        b_range = np.linspace(-100, 100, 10)
        L_range = np.linspace(0, 100, 10)
        
        # Create boundary points for the LAB color space
        # Bottom face (L=0)
        for a_val in a_range:
            for b_val in b_range:
                fig.add_trace(
                    go.Scatter3d(
                        x=[a_val], y=[b_val], z=[0],
                        mode='markers',
                        marker=dict(size=2, color='lightgray', opacity=0.3),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Top face (L=100)
        for a_val in a_range:
            for b_val in b_range:
                fig.add_trace(
                    go.Scatter3d(
                        x=[a_val], y=[b_val], z=[100],
                        mode='markers',
                        marker=dict(size=2, color='lightgray', opacity=0.3),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Side faces
        for L_val in L_range:
            for a_val in [-100, 100]:
                for b_val in b_range:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[a_val], y=[b_val], z=[L_val],
                            mode='markers',
                            marker=dict(size=2, color='lightgray', opacity=0.3),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            
            for b_val in [-100, 100]:
                for a_val in a_range:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[a_val], y=[b_val], z=[L_val],
                            mode='markers',
                            marker=dict(size=2, color='lightgray', opacity=0.3),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        # Add 3D scatter plot of LAB colors
        L, a, b = zip(*lab_colors)
        fig.add_trace(
            go.Scatter3d(
                x=a, y=b, z=L,
                mode='markers',
                marker=dict(
                    size=12,
                    color=[f'rgb({int(r)},{int(g)},{int(b)})' 
                           for r,g,b in color_analysis['dominant_colors']],
                    opacity=1.0
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
        
        # Create a better visualization of dominant colors - color swatches
        fig.add_trace(
            go.Heatmap(
                z=[[i for i in range(len(color_analysis['dominant_colors']))]],
                colorscale=[
                    [i/(len(color_analysis['dominant_colors'])-1), 
                     f'rgb({int(c[0])},{int(c[1])},{int(c[2])})'] 
                    for i, c in enumerate(color_analysis['dominant_colors'])
                ],
                showscale=False,
                hoverinfo='text',
                text=[[f"RGB: {c[0]},{c[1]},{c[2]}\nPercentage: {p*100:.1f}%"] 
                      for c, p in zip(color_analysis['dominant_colors'], 
                                      color_analysis['dominant_percentages'])]
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
                text=f'{image_name} - Color Analysis',
                x=0.5,
                font=dict(size=24)
            ),
            template='plotly_white'
        )
        
        # Update axes for the dominant colors plot
        fig.update_xaxes(
            title_text="Color Distribution",
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Dominant Colors",
            row=1, col=2
        )
        
        return fig

    def visualize_texture(self, texture_analysis, image_name="Eyebrow"):
        """Visualize the texture analysis results using interactive visualization.
        
        Args:
            texture_analysis (dict): Texture analysis dictionary from analyze_texture
            image_name (str): Name of the image for the title
            
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
        
        # Texture Metrics Bar Chart with explanations
        metrics = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        values = [texture_analysis[metric] for metric in metrics]
        
        # Prepare hover text with explanations
        explanations = {
            'contrast': 'Measures local variations in the image. Higher values indicate more texture.',
            'dissimilarity': 'Measures how different each pixel is from its neighbor. Higher values indicate more texture.',
            'homogeneity': 'Measures the closeness of elements distribution. Higher values indicate smoother texture.',
            'energy': 'Measures the uniformity of the texture. Higher values indicate more uniform texture.',
            'correlation': 'Measures how correlated a pixel is to its neighbor. Higher values indicate more structure.'
        }
        
        hover_texts = [f"{metric}: {values[i]:.3f}<br>{explanations[metric]}" for i, metric in enumerate(metrics)]
        
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
                hovertext=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Update axes for texture metrics
        fig.update_xaxes(
            title_text="Texture Features",
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Feature Value",
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
                    name='LBP Histogram',
                    hovertemplate='Pattern %{x}: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Update axes for LBP histogram
            fig.update_xaxes(
                title_text="LBP Pattern ID",
                row=1, col=2
            )
            fig.update_yaxes(
                title_text="Frequency",
                row=1, col=2
            )
        
        # GLCM Matrix Heatmap with explanations
        glcm_matrix = np.array([
            [texture_analysis['contrast'], texture_analysis['dissimilarity']],
            [texture_analysis['homogeneity'], texture_analysis['energy']]
        ])
        
        # Add annotations to explain GLCM matrix
        glcm_explanations = [
            "The GLCM Matrix shows relationships between texture features:<br>" +
            "- <b>Contrast</b>: Measures local variations<br>" +
            "- <b>Dissimilarity</b>: Measures how different neighboring pixels are<br>" +
            "- <b>Homogeneity</b>: Measures smoothness of texture<br>" +
            "- <b>Energy</b>: Measures uniformity of texture"
        ]
        
        fig.add_trace(
            go.Heatmap(
                z=glcm_matrix,
                x=['Contrast', 'Dissimilarity'],
                y=['Homogeneity', 'Energy'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Value'),
                hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add annotation for GLCM explanation
        fig.add_annotation(
            text=glcm_explanations[0],
            xref="x3", yref="y3",
            x=0.5, y=0.1,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
        
        # Add texture information table with expanded explanations
        texture_info = [
            f"Texture Category: {texture_analysis['texture_category']}",
            f"Texture Score: {texture_analysis['texture_score']:.2f}",
            f"Uniformity: {texture_analysis['uniformity_category']}",
            f"Correlation: {texture_analysis['correlation']:.2f}",
            f"Entropy: {texture_analysis['entropy']:.2f}"
        ]
        
        # Add explanations for the metrics
        texture_explanations = [
            "Describes the overall texture feel (smooth, medium, rough)",
            "Numerical score combining contrast, dissimilarity, homogeneity and energy",
            "Describes how consistent the texture is across the eyebrow",
            "Measures the linear dependency of gray levels (0-1)",
            "Measures randomness in the texture (higher = more random)"
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Texture Information', 'Explanation'],
                    fill_color='royalblue',
                    font=dict(color='white')
                ),
                cells=dict(
                    values=[texture_info, texture_explanations],
                    fill_color=['lightgray', 'white'],
                    align='left'
                ),
                columnwidth=[200, 300]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title=dict(
                text=f'{image_name} - Texture Analysis',
                x=0.5,
                font=dict(size=24)
            ),
            template='plotly_white'
        )
        
        return fig
