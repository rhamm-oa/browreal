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
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from eyebrow_analysis.landmarks_no_dlib import EyebrowLandmarkDetector
from eyebrow_analysis.segmentation import EyebrowSegmentation
from eyebrow_analysis.color_texture import EyebrowColorTexture
from eyebrow_analysis.enhanced_visualizations import EnhancedVisualizations

class EyebrowAnalyzer:
    """Main class for comprehensive eyebrow analysis."""
    
    def __init__(self, model_path=None):
        """
        Initialize the eyebrow analyzer.
        
        Args:
            model_path (str, optional): Path to the facial landmark model (not used in dlib-free version).
        """
        # Initialize components
        self.landmark_detector = EyebrowLandmarkDetector()
        self.segmentation = EyebrowSegmentation()
        self.color_texture = EyebrowColorTexture()
        self.visualizer = EnhancedVisualizations()
        
        # PDF report styles
        self.styles = getSampleStyleSheet()
        self.custom_style = ParagraphStyle(
            'CustomStyle',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            spaceAfter=20,
            leading=16
        )
        self.title_style = ParagraphStyle(
            'TitleStyle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
    
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
            
            # Create a specific directory for PDF reports
            pdf_dir = "/home/user/vbrow/visualizations"
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Visualize landmarks
            landmarks_vis = self.landmark_detector.visualize_landmarks(image, landmarks)
            landmarks_path = os.path.join(output_dir, f"{base_name}_landmarks.jpg")
            cv2.imwrite(landmarks_path, landmarks_vis)
            
            # Visualize segmentation
            segmentation_vis = self.segmentation.visualize_segmentation(image, segmentation_result)
            segmentation_path = os.path.join(output_dir, f"{base_name}_segmentation.jpg")
            cv2.imwrite(segmentation_path, segmentation_vis)
            
            # Visualize color analysis
            color_path = None
            if color_analysis:
                color_fig = self.color_texture.visualize_color(color_analysis)
                if color_fig:
                    # Save interactive HTML version
                    color_path = os.path.join(output_dir, f"{base_name}_color.html")
                    color_fig.write_html(color_path)
                    
                    # Save static image for PDF report
                    color_img_path = os.path.join(output_dir, f"{base_name}_color.jpg")
                    color_fig.write_image(color_img_path)
                    
                    # Update color_path to point to the image for PDF report
                    color_path = color_img_path
            
            # Visualize texture analysis
            texture_path = None
            if texture_analysis:
                texture_fig = self.color_texture.visualize_texture(texture_analysis)
                if texture_fig:
                    # Save interactive HTML version
                    texture_path = os.path.join(output_dir, f"{base_name}_texture.html")
                    texture_fig.write_html(texture_path)
                    
                    # Save static image for PDF report
                    texture_img_path = os.path.join(output_dir, f"{base_name}_texture.jpg")
                    texture_fig.write_image(texture_img_path)
                    
                    # Update texture_path to point to the image for PDF report
                    texture_path = texture_img_path
            
            # Create combined visualization
            combined_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
            combined_vis = self.create_combined_visualization(
                image, landmarks, segmentation_result, 
                color_analysis, texture_analysis, 
                symmetry, arch, thickness, length, shape_analysis
            )
            cv2.imwrite(combined_path, combined_vis)
            
            # Create enhanced PDF report with all visualizations
            self._create_enhanced_pdf_report(
                image_path=image_path,
                results=results,
                landmarks_path=landmarks_path,
                segmentation_path=segmentation_path,
                color_path=color_path,
                texture_path=texture_img_path,
                combined_path=combined_path,
                output_dir=pdf_dir
            )
        
        return results
    
    def _create_combined_visualization(self, image_path, results, output_dir):
        # Load image
        image = cv2.imread(image_path)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create a larger canvas
        h, w = image.shape[:2]
        canvas = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255
        
        # Add original image
        canvas[:h, :w] = image
        
        # Add landmarks visualization
        landmarks_vis = self.landmark_detector.visualize_landmarks(image, results['landmarks'], show_metrics=False)
        canvas[:h, w:w*2] = landmarks_vis
        
        # Add segmentation visualization
        if results['shape']:
            segmentation_vis = self.segmentation.visualize_segmentation(image, results['shape'])
            canvas[h:h*2, :w] = segmentation_vis
        
        # Create metrics visualization
        metrics_vis = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Add metrics text
        metrics = [
            f"Symmetry: {results['landmarks']['symmetry']:.2f}",
            f"Arch Type: {results['landmarks']['arch']['arch_type']}",
            f"Thickness: {results['landmarks']['thickness']['thickness_category']}",
            f"Length: {results['landmarks']['length']['length_category']}",
            f"Shape: {results['shape']['shape_category'] if results['shape'] else 'N/A'}",
            f"Density: {results['shape']['density_category'] if results['shape'] else 'N/A'}",
            f"Color: {results['color']['color_name'] if results['color'] else 'N/A'}",
            f"Tone: {results['color']['tone'] if results['color'] else 'N/A'}",
            f"Texture: {results['texture']['texture_category'] if results['texture'] else 'N/A'}",
            f"Uniformity: {results['texture']['uniformity_category'] if results['texture'] else 'N/A'}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(metrics_vis, metric, (20, 40 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add color swatch if available
        if results['color'] and 'mean_rgb' in results['color']:
            mean_rgb = results['color']['mean_rgb']
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
        
        # Save combined visualization
        combined_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
        cv2.imwrite(combined_path, canvas)
        
        return combined_path
    
    def _create_enhanced_pdf_report(self, image_path, results, landmarks_path, segmentation_path,
                                color_path, texture_path, combined_path, output_dir):
        """
        Create an enhanced PDF report with detailed visualizations.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the report filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        report_path = os.path.join(output_dir, f"{base_name}_enhanced_report.pdf")
        
        # Create the PDF document
        doc = SimpleDocTemplate(report_path, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("Eyebrow Analysis Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Original Image
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.thumbnail((400, 400))  # Resize for PDF
            img_path = os.path.join(output_dir, "temp_original.png")
            img.save(img_path)
            story.append(Paragraph("Original Image", self.styles['Heading2']))
            story.append(RLImage(img_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 20))
            
        # Add landmarks visualization
        if os.path.exists(landmarks_path):
            story.append(Paragraph("Facial Landmarks", self.styles['Heading2']))
            story.append(RLImage(landmarks_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 20))
        
        # Add segmentation visualization
        if os.path.exists(segmentation_path):
            story.append(Paragraph("Eyebrow Segmentation", self.styles['Heading2']))
            story.append(RLImage(segmentation_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 20))
        
        # Add color analysis
        if color_path and os.path.exists(color_path):
            story.append(Paragraph("Color Analysis", self.styles['Heading2']))
            story.append(RLImage(color_path, width=7*inch, height=5*inch))
            story.append(Spacer(1, 20))
            
            # Add color space visualizations if they exist
            color_spaces = ['lab', 'rgb', 'hsv']
            for space in color_spaces:
                space_path = os.path.join(output_dir, f"{base_name}_color_space_{space}.png")
                if os.path.exists(space_path):
                    story.append(Paragraph(f"Color Distribution in {space.upper()} Space", self.styles['Heading3']))
                    story.append(RLImage(space_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 15))
        
        # Add texture analysis
        if texture_path and os.path.exists(texture_path):
            story.append(Paragraph("Texture Analysis", self.styles['Heading2']))
            story.append(RLImage(texture_path, width=7*inch, height=5*inch))
            story.append(Spacer(1, 20))
        
        # Add combined visualization
        if os.path.exists(combined_path):
            story.append(Paragraph("Combined Analysis", self.styles['Heading2']))
            story.append(RLImage(combined_path, width=7*inch, height=5*inch))
            story.append(Spacer(1, 20))
        
        # Add metrics summary
        story.append(Paragraph("Analysis Summary", self.styles['Heading2']))
        
        # Landmark Analysis
        story.append(Paragraph("Landmark Analysis", self.styles['Heading3']))
        metrics = [
            f"Symmetry: {results['landmarks']['symmetry']:.2f}",
            f"Arch Type: {results['landmarks']['arch']}",
            f"Thickness: {results['landmarks']['thickness']}",
            f"Length: {float(results['landmarks']['length']):.2f} pixels" if isinstance(results['landmarks']['length'], (int, float)) else f"Length: {results['landmarks']['length']}"
        ]
        for metric in metrics:
            story.append(Paragraph(metric, self.custom_style))
        story.append(Spacer(1, 10))
        
        # Shape Analysis
        if results['shape']:
            story.append(Paragraph("Shape Analysis", self.styles['Heading3']))
            shape_metrics = []
            if 'shape_category' in results['shape']:
                shape_metrics.append(f"Shape Category: {results['shape']['shape_category']}")
            if 'area' in results['shape']:
                shape_metrics.append(f"Area: {results['shape']['area']:.2f}")
            if 'perimeter' in results['shape']:
                shape_metrics.append(f"Perimeter: {results['shape']['perimeter']:.2f}")
            if 'compactness' in results['shape']:
                shape_metrics.append(f"Compactness: {results['shape']['compactness']:.2f}")
            for metric in shape_metrics:
                story.append(Paragraph(metric, self.custom_style))
            story.append(Spacer(1, 10))
        # Color Analysis
        if results['color']:
            story.append(Paragraph("Color Analysis", self.styles['Heading3']))
            
            # Add color space explanation
            color_explanations = [
                "The color analysis uses the LAB color space, which represents colors in three dimensions:",
                "L: Lightness (0 = black, 100 = white)",
                "A: Green to Red (-128 = green, +127 = red)",
                "B: Blue to Yellow (-128 = blue, +127 = yellow)"
            ]
            
            for explanation in color_explanations:
                story.append(Paragraph(explanation, self.custom_style))
            
            # Add 3D LAB visualization
            if color_path and color_path.endswith('.html'):
                story.append(Paragraph("3D Color Space Visualization", self.styles['Heading4']))
                story.append(Image(color_path.replace('.html', '.jpg'), width=400, height=300))
            
            color_metrics = [
                f"Color Name: {results['color']['color_name']}",
                f"Tone: {results['color']['tone']}",
                f"Brightness: {results['color']['brightness_category']}",
                f"Color Consistency: {results['color']['color_consistency']:.2f}"
            ]
            for metric in color_metrics:
                story.append(Paragraph(metric, self.custom_style))
            story.append(Spacer(1, 10))
        
        # Texture Analysis
        if results['texture']:
            story.append(Paragraph("Texture Analysis", self.styles['Heading3']))
            
            # Add explanations for metrics
            texture_explanations = [
                "Contrast: How different neighboring areas are - higher values mean more dramatic changes in texture",
                "Dissimilarity: How different the overall texture patterns are - lower values indicate more uniform patterns",
                "Homogeneity: How similar nearby areas are - higher values mean more consistent texture",
                "Energy: Overall uniformity measure - higher values indicate more uniform, less complex textures",
                "Correlation: How related neighboring areas are - higher values show more predictable patterns"
            ]
            
            for explanation in texture_explanations:
                story.append(Paragraph(explanation, self.custom_style))
            texture_metrics = [
                f"Texture Category: {results['texture']['texture_category']}",
                f"Uniformity: {results['texture']['uniformity_category']}",
                f"Texture Score: {results['texture']['texture_score']:.2f}",
                f"Entropy: {results['texture']['entropy']:.2f}"
            ]
            for metric in texture_metrics:
                story.append(Paragraph(metric, self.custom_style))
            story.append(Spacer(1, 10))
        
        # Build the PDF
        doc.build(story)
        
        # Clean up temporary files
        for temp_file in os.listdir(output_dir):
            if temp_file.startswith('temp_'):
                os.remove(os.path.join(output_dir, temp_file))
        
        return report_path

    def analyze_batch(self, image_dir, trimap_dir=None, output_dir=None, visualize=False):
        """Analyze eyebrows in a batch of images.
        
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
