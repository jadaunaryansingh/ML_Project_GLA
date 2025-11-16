#!/bin/bash
# RDKit installation script for Render
# RDKit doesn't have a reliable pip package, so we'll try alternative methods

echo "Installing RDKit..."

# Method 1: Try rdkit-pypi (works for Python 3.8-3.11)
pip install rdkit-pypi==2022.9.5 || {
    echo "rdkit-pypi failed, trying alternative..."
    # Method 2: Try installing from conda-forge via pip (if available)
    pip install rdkit || {
        echo "Standard rdkit failed, trying rdkit-pypi with specific version..."
        # Method 3: Try without version constraint
        pip install rdkit-pypi || {
            echo "All RDKit installation methods failed!"
            echo "Please check Python version compatibility"
            exit 1
        }
    }
}

echo "RDKit installation completed"



