from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
def read_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()
    
setup(
    name='analysis',  # Replace with your package name
    version='0.1.0',  # Define the package version
    author='Manuel',  # Replace with your name
    description='A brief description of your package',
    #long_description=open('README.md').read(),  # Make sure to have a README.md file
    #long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/analysis',  # Replace with your repository URL
    package_dir={'': 'src'},  # Indicate that packages are under src
    packages=find_packages(where='src'),  # Automatically find packages in src
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update based on your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # Define the minimum Python version
    install_requires=read_requirements('requirements.txt'),  # Read dependencies from requirements.txt
)