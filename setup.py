##### Sensemore Communication Protocol : SMCom  #######
#	

#Required packages
# - numpy


import os,sys
import subprocess
import pathlib
from distutils import spawn
from setuptools import setup, find_packages, Extension

try:
	import numpy
except ImportError:
	if subprocess.call([sys.executable, '-m', 'pip', 'install', 'numpy']):
		raise RuntimeError('numpy install failed.')
	
try:
	import scipy
except ImportError:
	if subprocess.call([sys.executable, '-m', 'pip', 'install', 'scipy']):
		raise RuntimeError('scipy install failed.')
	
try:
	import matplotlib
except ImportError:
	if subprocess.call([sys.executable, '-m', 'pip', 'install', 'matplotlib']):
		raise RuntimeError('matplotlib install failed.')

try:
	import sklearn
except ImportError:
    if subprocess.call([sys.executable, '-m', 'pip', 'install', 'scikit-learn']):
        raise RuntimeError('scikit-learn install failed.')

setup(
    name='pyflow',
    version='0.1',  # Change this to your desired version
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn'],
    author='M. Samil Arinc',
    author_email='msarinc@gmail.com',
    description='PyFlow is a deep learning framework on Python. It is a framework for building and training deep neural networks.',
    url='https://github.com/samilarinc/DL_Framework_Python',  # Update with your repository URL
)
