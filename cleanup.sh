#!/bin/bash

# Exit on error
set -e

echo "Starting project cleanup..."

# 1. Move app directory to root if it exists in project/
if [ -d "project/app" ]; then
    echo "Moving app directory to root..."
    mv project/app .
fi

# 2. Move config directory to root if it exists in project/
if [ -d "project/config" ]; then
    echo "Moving config directory to root..."
    mv project/config .
fi

# 3. Move README.md to root if it exists in project/
if [ -f "project/README.md" ]; then
    echo "Moving README.md to root..."
    mv project/README.md .
fi

# 4. Move .env to root if it exists in project/
if [ -f "project/.env" ]; then
    echo "Moving .env to root..."
    mv project/.env .
fi

# 5. Merge test directories
if [ -d "project/tests" ]; then
    echo "Merging test directories..."
    # Create tests directory if it doesn't exist
    mkdir -p tests
    # Move all test files from project/tests to tests
    cp -r project/tests/* tests/ 2>/dev/null || true
fi

# 6. Clean up duplicate files
echo "Cleaning up duplicate files..."
rm -f project/pyproject.toml
rm -f project/.gitignore
rm -f project/setup.py
rm -f project/requirements.txt
rm -f project/test_requirements.txt

# 7. Clean up duplicate directories
echo "Cleaning up duplicate directories..."
rm -rf project/tests
rm -rf project/htmlcov
rm -rf project/.benchmarks
rm -rf project/logs
rm -rf project/.coverage

# 8. Clean up build artifacts
echo "Cleaning up build artifacts..."
rm -rf project/gaffer.egg-info

# 9. Remove empty project directory if it exists
if [ -d "project" ]; then
    echo "Removing empty project directory..."
    rm -rf project
fi

echo "Cleanup complete!" 