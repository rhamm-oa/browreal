#!/usr/bin/env python
import os
import subprocess
import sys
import platform

def create_virtual_environment():
    """Create a virtual environment for the eyebrow analysis project."""
    venv_dir = "eyebrow_venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print(f"Virtual environment '{venv_dir}' already exists.")
        return venv_dir
    
    print(f"Creating virtual environment in '{venv_dir}'...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    print("Virtual environment created successfully.")
    
    return venv_dir

def install_dependencies(venv_dir):
    """Install required dependencies in the virtual environment."""
    # Determine the pip executable path based on the operating system
    if sys.platform == "win32":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
        python_path = os.path.join(venv_dir, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")
    
    print("Upgrading pip...")
    subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
    
    # Install dependencies
    print("Installing dependencies...")
    try:
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Warning: Some packages failed to install. Continuing with installation...")

def update_code_for_gpu():
    """Update the code to use GPU where applicable."""
    print("\nUpdating code to use dlib-free implementation...")
    
    # Copy landmarks_no_dlib.py to landmarks.py
    landmarks_no_dlib = os.path.join("eyebrow_analysis", "landmarks_no_dlib.py")
    landmarks_file = os.path.join("eyebrow_analysis", "landmarks.py")
    
    if os.path.exists(landmarks_no_dlib):
        print(f"Using dlib-free implementation from {landmarks_no_dlib}")
        with open(landmarks_no_dlib, 'r') as src:
            content = src.read()
            
        with open(landmarks_file, 'w') as dst:
            dst.write(content)
            
        print(f"Updated {landmarks_file} to use MediaPipe instead of dlib.")
    
    # Copy analyzer_no_dlib.py to analyzer.py
    analyzer_no_dlib = os.path.join("eyebrow_analysis", "analyzer_no_dlib.py")
    analyzer_file = os.path.join("eyebrow_analysis", "analyzer.py")
    
    if os.path.exists(analyzer_no_dlib):
        print(f"Using dlib-free analyzer from {analyzer_no_dlib}")
        with open(analyzer_no_dlib, 'r') as src:
            content = src.read()
            
        with open(analyzer_file, 'w') as dst:
            dst.write(content)
            
        print(f"Updated {analyzer_file} to use dlib-free implementation.")
        
    # Update main.py to use the standard analyzer path
    main_file = "main.py"
    if os.path.exists(main_file):
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Replace import from analyzer_no_dlib to analyzer
        if "from eyebrow_analysis.analyzer_no_dlib import EyebrowAnalyzer" in content:
            content = content.replace(
                "from eyebrow_analysis.analyzer_no_dlib import EyebrowAnalyzer",
                "from eyebrow_analysis.analyzer import EyebrowAnalyzer"
            )
            
            with open(main_file, 'w') as f:
                f.write(content)
                
            print(f"Updated {main_file} to use standard analyzer path.")
        else:
            print(f"No changes needed in {main_file}.")

def main():
    venv_dir = create_virtual_environment()
    install_dependencies(venv_dir)
    update_code_for_gpu()
    
    print("\nSetup completed. To activate the virtual environment:")
    
    if sys.platform == "win32":
        print(f"Run: {venv_dir}\\Scripts\\activate")
    else:
        print(f"Run: source {venv_dir}/bin/activate")
    
    print("\nNOTE: This setup uses MediaPipe for facial landmark detection instead of dlib.")
    print("MediaPipe provides good accuracy for eyebrow analysis without requiring complex dependencies.")
    print("\nTo run the eyebrow analysis:")
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print(f"   {venv_dir}\\Scripts\\activate")
    else:
        print(f"   source {venv_dir}/bin/activate")
    print("2. Run the analysis on a single image:")
    print("   python main.py --mode single --image path/to/image.jpg --visualize")
    print("3. Or run batch analysis on a directory of images:")
    print("   python main.py --mode batch --image_dir path/to/images --visualize")

if __name__ == "__main__":
    main()
