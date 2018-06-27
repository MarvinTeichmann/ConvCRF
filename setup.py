#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='ConvCRF',
      version='1.0',
      description='Reference Implementation of ConvCRF.',
      author='Marvin Teichmann',
      author_email=('marvin.teichmann@googlemail.com'),
      packages=find_packages(),
      package_data={'': ['*.lst']}
      )
