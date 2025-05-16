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
    
    # Install main dependencies first (excluding dlib and face-alignment)
    print("Installing main dependencies...")
    try:
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("Main dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Warning: Some packages failed to install. Continuing with installation...")
    
    # Install dlib separately with different methods depending on the platform
    print("\nInstalling dlib...")
    try:
        # Try the simple pip install first
        subprocess.check_call([pip_path, "install", "dlib>=19.22.0"])
        print("Dlib installed successfully.")
    except subprocess.CalledProcessError:
        print("Standard dlib installation failed. Trying alternative methods...")
        
        if sys.platform == "win32":
            # For Windows, try pre-built wheel
            try:
                print("Trying pre-built dlib wheel...")
                subprocess.check_call([pip_path, "install", "https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl"])
                print("Dlib installed from pre-built wheel.")
            except subprocess.CalledProcessError:
                print("Pre-built wheel failed. Please install dlib manually.")
                print("Instructions: https://github.com/davisking/dlib#installation")
        else:
            # For Linux/Mac, try with specific CMake flags
            try:
                print("Installing dlib with specific CMake flags...")
                # Install required system dependencies first
                if platform.system() == "Linux":
                    print("You may need to install the following system packages:")
                    print("sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev")
                    print("sudo apt-get install libx11-dev libgtk-3-dev")
                    print("sudo apt-get install python3-dev")
                
                # Install dlib with specific flags
                env = os.environ.copy()
                env["DLIB_USE_CUDA"] = "1"  # Enable CUDA
                subprocess.check_call([pip_path, "install", "dlib>=19.22.0", "--no-cache-dir", "--verbose"], env=env)
                print("Dlib installed with CUDA support.")
            except subprocess.CalledProcessError:
                print("Alternative dlib installation failed.")
                print("Please install dlib manually following instructions at:")
                print("https://github.com/davisking/dlib#installation")
    
    # Install face-alignment after dlib
    print("\nInstalling face-alignment...")
    try:
        subprocess.check_call([pip_path, "install", "face-alignment>=1.3.5"])
        print("Face-alignment installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install face-alignment. You may need to install it manually.")

def update_code_for_gpu():
    """Update the code to use GPU where applicable."""
    print("\nUpdating code to use GPU acceleration...")
    
    # Update landmarks.py to use GPU for face-alignment
    landmarks_file = os.path.join("eyebrow_analysis", "landmarks.py")
    if os.path.exists(landmarks_file):
        with open(landmarks_file, 'r') as f:
            content = f.read()
        
        # Replace CPU device with CUDA if available
        if "device='cpu'" in content:
            content = content.replace("device='cpu'", "device='cuda' if torch.cuda.is_available() else 'cpu'")
            
            # Add torch import if not already there
            if "import torch" not in content:
                import_section_end = content.find("class EyebrowLandmarkDetector:")
                content = content[:import_section_end] + "import torch\n" + content[import_section_end:]
            
            with open(landmarks_file, 'w') as f:
                f.write(content)
            
            print(f"Updated {landmarks_file} to use GPU if available.")
        else:
            print(f"No changes needed in {landmarks_file}.")

def main():
    venv_dir = create_virtual_environment()
    install_dependencies(venv_dir)
    update_code_for_gpu()
    
    print("\nSetup completed. To activate the virtual environment:")
    
    if sys.platform == "win32":
        print(f"Run: {venv_dir}\\Scripts\\activate")
    else:
        print(f"Run: source {venv_dir}/bin/activate")
    
    print("\nNOTE: If you encounter issues with dlib or other packages:")
    print("1. You may need to install system dependencies for dlib:")
    print("   - On Ubuntu/Debian: sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python3-dev")
    print("   - On Windows: Install Visual Studio with C++ development tools")
    print("2. For GPU support, ensure you have CUDA and cuDNN installed")
    print("3. You can try installing pre-built wheels for dlib from:")
    print("   https://github.com/jloh02/dlib/releases")

if __name__ == "__main__":
    main()
