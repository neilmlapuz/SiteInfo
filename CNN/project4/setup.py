from setuptools import find_packages
from setuptools import setup

setup(name='trainer',
  version='0.1',
  packages=find_packages(),
  description='Run my keras model on gcloud ml-engine',
  author='Neil Lapuz',
  author_email='naruto_neil03@yahoo.com',
  license='MIT',
  install_requires=[
      'keras==2.13.1',
      'h5py'
  ],
  zip_safe=False)