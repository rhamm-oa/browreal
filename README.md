# Eyebrow Analysis System

A comprehensive tool for analyzing eyebrow characteristics from facial images, including landmarks, segmentation, color, texture, shape, and more.

## Features

This system provides detailed analysis of eyebrows, including:

- **Landmark Detection**: Identifies key eyebrow points using multiple detection methods (dlib, MediaPipe, face-alignment)
- **Segmentation**: Isolates eyebrow regions from facial images
- **Shape Analysis**: Analyzes eyebrow shape, arch, thickness, length, and density
- **Color Analysis**: Determines eyebrow color, tone, brightness, and color consistency
- **Texture Analysis**: Evaluates texture characteristics such as coarseness, uniformity, and pattern
- **Symmetry Analysis**: Measures the symmetry between left and right eyebrows
- **Visualization**: Generates detailed visualizations of all analysis aspects
- **Batch Processing**: Supports analysis of multiple images with summary statistics

## Project Structure

```
vbrow/
├── eyebrow_analysis/           # Main analysis modules
│   ├── __init__.py
│   ├── landmarks.py            # Eyebrow landmark detection
│   ├── segmentation.py         # Eyebrow segmentation
│   ├── color_texture.py        # Color and texture analysis
│   └── analyzer.py             # Main analyzer integrating all components
├── models/                     # Pre-trained models
│   └── shape_predictor_68_face_landmarks.dat
├── main.py                     # Main script to run analysis
├── setup_venv.py               # Script to set up virtual environment
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Installation

1. Clone or download this repository.

2. Set up a virtual environment:
   ```
   python setup_venv.py
   ```

3. Activate the virtual environment:
   - Windows: `eyebrow_venv\Scripts\activate`
   - Linux/Mac: `source eyebrow_venv/bin/activate`

4. Ensure you have the required model file:
   - Download the dlib facial landmark predictor model (`shape_predictor_68_face_landmarks.dat`) if not already present in the `models` directory
   - You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## Usage

### Single Image Analysis

```bash
python main.py --mode single --image path/to/image.jpg --visualize --output_dir results
```

Optional parameters:
- `--trimap path/to/trimap.jpg`: Provide a trimap for better segmentation
- `--model_path path/to/model.dat`: Specify a custom facial landmark model

### Batch Processing

```bash
python main.py --mode batch --image_dir path/to/images --visualize --output_dir results
```

Optional parameters:
- `--trimap_dir path/to/trimaps`: Directory containing trimap images
- `--model_path path/to/model.dat`: Specify a custom facial landmark model

## Output

The system generates:

1. **CSV file** with detailed metrics for each analyzed image
2. **Visualization images** showing:
   - Detected landmarks
   - Segmentation results
   - Color analysis
   - Texture analysis
   - Combined visualization with all metrics
3. **Interactive HTML visualizations** for batch analysis:
   - Color distribution
   - Shape distribution
   - Texture distribution
   - Symmetry distribution
   - Correlation heatmap
   - Dashboard with multiple visualizations

## Dependencies

- OpenCV
- dlib
- NumPy
- scikit-image
- scikit-learn
- TensorFlow
- MediaPipe
- face-alignment
- matplotlib
- pandas
- plotly
- and more (see requirements.txt)

## Trimap Format

If using trimaps for improved segmentation:
- Trimaps should be grayscale images where:
  - 0 = background
  - 1 = unknown region
  - 2 = foreground (eyebrow)

## Extending the System

The modular design allows for easy extension:

- Add new analysis modules in the `eyebrow_analysis` directory
- Integrate new modules in the `analyzer.py` file
- Add new visualization methods as needed

## Troubleshooting

- **No face detected**: Ensure the image contains a clear, front-facing face
- **Poor segmentation**: Try providing a trimap for better results
- **Missing dependencies**: Ensure all requirements are installed via `pip install -r requirements.txt`
- **Model not found**: Download the required facial landmark model and place it in the `models` directory
