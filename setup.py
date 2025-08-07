import subprocess
import sys
import pkg_resources
import importlib

def check_and_install_requirements():
    """Check if all requirements are installed and install missing ones"""
    
    # Read requirements from requirements.txt
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print("Checking required packages...")
    
    # Check which packages are missing
    missing_packages = []
    for requirement in requirements:
        try:
            # Try to import the package
            package_name = requirement.split('>=')[0].split('==')[0]
            importlib.import_module(package_name.replace('-', '_'))
            print(f"✓ {requirement}")
        except ImportError:
            print(f"✗ {requirement} - MISSING")
            missing_packages.append(requirement)
    
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✓ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing packages: {e}")
            return False
    else:
        print("\n✓ All required packages are already installed!")
    
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    import os
    
    directories = [
        'data',
        'models',
        'utils',
        'training',
        'inference',
        'configs',
        'notebooks',
        'runs'
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def main():
    print("Setting up MNIST Document Processing Environment")
    print("=" * 50)
    
    # Check and install requirements
    if check_and_install_requirements():
        # Create directories
        create_directories()
        
        print("\n" + "=" * 50)
        print("✓ Setup completed successfully!")
        print("\nYou can now run the MNIST training:")
        print("python main.py --mode train --model cnn --epochs 10")
        print("\nOr evaluate a trained model:")
        print("python main.py --mode evaluate --model cnn")
    else:
        print("\n✗ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
