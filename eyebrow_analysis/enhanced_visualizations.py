"""
Enhanced visualization module for eyebrow analysis.
Provides improved visualizations including 3D color spaces and detailed texture analysis.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from skimage import color as skcolor
import colorsys

class EnhancedVisualizations:
    def create_modern_texture_visualization(self, texture_analysis):
        """
        Create a modern, interactive visualization of texture analysis results.
        
        Args:
            texture_analysis (dict): Texture analysis results
            
        Returns:
            plotly.graph_objects.Figure: Interactive texture visualization
        """
        if texture_analysis is None:
            return None
            
        self.texture_analysis = texture_analysis
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Texture Metrics',
                'GLCM Matrix',
                'LBP Distribution',
                'Key Properties'
            ),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                  [{"type": "bar"}, {"type": "table"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Texture Metrics Bar Chart with modern styling
        metrics = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        values = [self.texture_analysis[metric] for metric in metrics]
        
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
            row=1, col=2
        )
        
        # LBP Histogram with modern styling
        if 'lbp_hist' in texture_analysis and texture_analysis['lbp_hist'] is not None:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(texture_analysis['lbp_hist']))),
                    y=texture_analysis['lbp_hist'],
                    marker=dict(
                        color='rgb(26, 118, 255)',
                        opacity=0.7,
                        line=dict(width=1, color='rgb(50, 50, 50)')
                    ),
                    hovertemplate='Pattern %{x}: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Properties Table with modern styling
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Property', 'Value'],
                    font=dict(size=12, color='white'),
                    fill_color='rgb(55, 83, 109)',
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Category', 'Uniformity', 'Score', 'Entropy'],
                        [
                            texture_analysis['texture_category'],
                            texture_analysis['uniformity_category'],
                            f"{texture_analysis['texture_score']:.2f}",
                            f"{texture_analysis['entropy']:.2f}"
                        ]
                    ],
                    font=dict(size=11),
                    align='left',
                    fill_color=['rgb(245, 247, 249)', 'white']
                )
            ),
            row=2, col=2
        )
        
        # Update layout with modern styling
        fig.update_layout(
            height=800,
            template='plotly_white',
            font_family=self.font_family,
            showlegend=False,
            title=dict(
                text='Texture Analysis',
                x=0.5,
                font=dict(size=24)
            )
        )
        
        # Update axes styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig

    def __init__(self):
        """Initialize the enhanced visualization module."""
        plt.style.use('default')  # Using default matplotlib style
        self.font_family = "Arial"
        # Set some nice default parameters for plots
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def create_3d_color_space(self, pixels, color_space='lab'):
        """
        Create a 3D visualization of colors in different color spaces.
        
        Args:
            pixels (numpy.ndarray): Array of pixel values in BGR format
            color_space (str): Color space to visualize ('lab', 'rgb', or 'hsv')
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D color space visualization
        """
        # Convert BGR to RGB
        rgb_pixels = pixels[...,::-1]
        
        if color_space.lower() == 'lab':
            # Convert RGB to LAB
            lab_pixels = skcolor.rgb2lab(rgb_pixels / 255.0)
            fig = go.Figure(data=[go.Scatter3d(
                x=lab_pixels[:,0], y=lab_pixels[:,1], z=lab_pixels[:,2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rgb_pixels,
                    opacity=0.7
                )
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title='L* (Lightness)',
                    yaxis_title='a* (Green-Red)',
                    zaxis_title='b* (Blue-Yellow)'
                ),
                title='Color Distribution in LAB Space'
            )
            
        elif color_space.lower() == 'rgb':
            fig = go.Figure(data=[go.Scatter3d(
                x=rgb_pixels[:,0], y=rgb_pixels[:,1], z=rgb_pixels[:,2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rgb_pixels,
                    opacity=0.7
                )
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title='Red',
                    yaxis_title='Green',
                    zaxis_title='Blue'
                ),
                title='Color Distribution in RGB Space'
            )
            
        elif color_space.lower() == 'hsv':
            # Convert RGB to HSV
            hsv_pixels = skcolor.rgb2hsv(rgb_pixels)
            fig = go.Figure(data=[go.Scatter3d(
                x=hsv_pixels[:,0]*360, y=hsv_pixels[:,1]*100, z=hsv_pixels[:,2]*100,
                mode='markers',
                marker=dict(
                    size=3,
                    color=rgb_pixels,
                    opacity=0.7
                )
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title='Hue (degrees)',
                    yaxis_title='Saturation (%)',
                    zaxis_title='Value (%)'
                ),
                title='Color Distribution in HSV Space'
            )
            
        fig.update_layout(
            font_family=self.font_family,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def create_color_analysis_dashboard(self, color_analysis):
        """
        Create a comprehensive color analysis dashboard.
        
        Args:
            color_analysis (dict): Color analysis results
            
        Returns:
            plotly.graph_objects.Figure: Interactive color analysis dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Dominant Colors',
                'Color Values in Different Spaces',
                'Color Distribution',
                'Color Properties'
            ),
            specs=[[{"type": "bar"}, {"type": "table"}],
                  [{"type": "pie"}, {"type": "table"}]]
        )
        
        # Dominant Colors Bar Chart
        dominant_colors = color_analysis['dominant_colors']
        percentages = color_analysis['dominant_percentages']
        
        fig.add_trace(
            go.Bar(
                x=[f'Color {i+1}' for i in range(len(dominant_colors))],
                y=percentages * 100,
                marker_color=[f'rgb({r},{g},{b})' for r,g,b in dominant_colors],
                name='Dominant Colors'
            ),
            row=1, col=1
        )
        
        # Color Values Table
        mean_rgb = color_analysis['mean_rgb']
        mean_hsv = color_analysis['mean_hsv']
        mean_lab = color_analysis['mean_lab']
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Color Space', 'Values'],
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=[
                        ['RGB', 'HSV', 'LAB'],
                        [
                            f"({mean_rgb[0]}, {mean_rgb[1]}, {mean_rgb[2]})",
                            f"({mean_hsv[0]:.0f}°, {mean_hsv[1]:.0f}%, {mean_hsv[2]:.0f}%)",
                            f"({mean_lab[0]:.0f}, {mean_lab[1]:.0f}, {mean_lab[2]:.0f})"
                        ]
                    ],
                    font=dict(size=11),
                    align="left"
                )
            ),
            row=1, col=2
        )
        
        # Color Distribution Pie Chart
        fig.add_trace(
            go.Pie(
                labels=[f'Color {i+1}' for i in range(len(dominant_colors))],
                values=percentages * 100,
                marker=dict(colors=[f'rgb({r},{g},{b})' for r,g,b in dominant_colors])
            ),
            row=2, col=1
        )
        
        # Color Properties Table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Property', 'Value'],
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=[
                        ['Color Name', 'Tone', 'Brightness', 'Consistency'],
                        [
                            color_analysis['color_name'],
                            color_analysis['tone'],
                            color_analysis['brightness_category'],
                            f"{color_analysis['color_consistency']:.2f}"
                        ]
                    ],
                    font=dict(size=11),
                    align="left"
                )
            ),
            row=2, col=2
        )
        
        # Create an enhanced GLCM visualization
        glcm_matrix = np.array([
            [self.texture_analysis['contrast'], self.texture_analysis['dissimilarity']],
            [self.texture_analysis['homogeneity'], self.texture_analysis['energy']]
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
            row=1, col=2
        )
        
        # Add LBP histograms if available
        if 'left_lbp_hist' in self.texture_analysis and self.texture_analysis['left_lbp_hist'] is not None:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(self.texture_analysis['left_lbp_hist']))),
                    y=self.texture_analysis['left_lbp_hist'],
                    name='Left Eyebrow',
                    marker=dict(
                        color='rgb(26, 118, 255)',
                        opacity=0.7,
                        line=dict(width=1, color='rgb(50, 50, 50)')
                    ),
                    hovertemplate='Pattern %{x}: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        if 'right_lbp_hist' in self.texture_analysis and self.texture_analysis['right_lbp_hist'] is not None:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(self.texture_analysis['right_lbp_hist']))),
                    y=self.texture_analysis['right_lbp_hist'],
                    name='Right Eyebrow',
                    marker=dict(
                        color='rgb(55, 83, 109)',
                        opacity=0.7,
                        line=dict(width=1, color='rgb(50, 50, 50)')
                    ),
                    hovertemplate='Pattern %{x}: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Properties Table with modern styling
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Property', 'Value'],
                    font=dict(size=12, color='white'),
                    fill_color='rgb(55, 83, 109)',
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Category', 'Uniformity', 'Score', 'Entropy'],
                        [
                            self.texture_analysis['texture_category'],
                            self.texture_analysis['uniformity_category'],
                            f"{self.texture_analysis['texture_score']:.2f}",
                            f"{self.texture_analysis['entropy']:.2f}"
                        ]
                    ],
                    font=dict(size=11),
                    align='left',
                    fill_color=['rgb(245, 247, 249)', 'white']
                )
            ),
            row=2, col=2
        )
        
        # Update layout with modern styling
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=True,
            title=dict(
                text='Texture Analysis',
                x=0.5,
                font=dict(size=24)
            )
        )
        
        # Update axes styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Update axes labels and styling
        fig.update_xaxes(title_text='Texture Metric', row=1, col=1)
        fig.update_yaxes(title_text='Value', row=1, col=1)
        fig.update_xaxes(title_text='LBP Pattern', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        
        return fig
