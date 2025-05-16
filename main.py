#!/usr/bin/env python
"""
Main script to run eyebrow analysis on images.
"""

import os
import argparse
import cv2
import matplotlib.pyplot as plt
from eyebrow_analysis.analyzer_no_dlib import EyebrowAnalyzer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Eyebrow Analysis Tool')
    
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='Analysis mode: single image or batch processing')
    
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image for single mode')
    
    parser.add_argument('--trimap', type=str, default=None,
                        help='Path to trimap image for single mode')
    
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images for batch mode')
    
    parser.add_argument('--trimap_dir', type=str, default=None,
                        help='Directory containing trimap images for batch mode')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to facial landmark model')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize analyzer
    analyzer = EyebrowAnalyzer(model_path=args.model_path)
    
    if args.mode == 'single':
        if args.image is None:
            print("Error: Image path is required for single mode.")
            return
        
        # Analyze single image
        results = analyzer.analyze_image(
            args.image, 
            trimap_path=args.trimap, 
            visualize=args.visualize, 
            output_dir=args.output_dir
        )
        
        if results:
            print("\nEyebrow Analysis Results:")
            print("-------------------------")
            print(f"Symmetry: {results['landmarks']['symmetry']:.2f}")
            print(f"Arch Type: {results['landmarks']['arch']['arch_type']}")
            print(f"Thickness: {results['landmarks']['thickness']['thickness_category']}")
            print(f"Length: {results['landmarks']['length']['length_category']}")
            
            if results['shape']:
                print(f"Shape: {results['shape']['shape_category']}")
                print(f"Density: {results['shape']['density_category']}")
            
            if results['color']:
                print(f"Color: {results['color']['color_name']}")
                print(f"Tone: {results['color']['tone']}")
                print(f"Brightness: {results['color']['brightness_category']}")
            
            if results['texture']:
                print(f"Texture: {results['texture']['texture_category']}")
                print(f"Uniformity: {results['texture']['uniformity_category']}")
            
            print("\nCheck the output directory for detailed visualizations.")
    
    elif args.mode == 'batch':
        if args.image_dir is None:
            print("Error: Image directory is required for batch mode.")
            return
        
        # Analyze batch of images
        results_df = analyzer.analyze_batch(
            args.image_dir,
            trimap_dir=args.trimap_dir,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        
        if results_df is not None:
            print(f"\nAnalyzed {len(results_df)} images.")
            print(f"Results saved to {args.output_dir}")
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print("-----------------")
            
            if 'landmarks_symmetry' in results_df.columns:
                print(f"Average Symmetry: {results_df['landmarks_symmetry'].mean():.2f}")
            
            for col in ['landmarks_arch_arch_type', 'landmarks_thickness_thickness_category', 
                        'landmarks_length_length_category', 'shape_shape_category',
                        'shape_density_category', 'color_color_name', 'color_tone',
                        'texture_texture_category', 'texture_uniformity_category']:
                if col in results_df.columns:
                    print(f"\n{col.split('_')[-1].title()} Distribution:")
                    print(results_df[col].value_counts())

if __name__ == "__main__":
    main()
