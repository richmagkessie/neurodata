from setuptools import setup, find_packages

setup(
    name='neurodata',
    version='0.1.0',
    description='Neural dataset encoding for privacy-preserving health data sharing',
    author='Richmond Kessie',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12',
        'pandas>=1.3',
        'numpy>=1.21',
        'scipy>=1.7',
        'scikit-learn>=1.0',
        'matplotlib>=3.4',
        'cryptography>=38.0',
    ],
)
