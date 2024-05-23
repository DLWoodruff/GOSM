#!/bin/usr/env python
import glob
import sys
import os

# We raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

from setuptools import find_packages
from distutils.core import setup

packages = find_packages()

setup(name='prescient_gosm',
      version='2.0',
      description='Power Generation Scenario creation utilities',
      url='https://github.com/jwatsonnm/prescient_gosm',
      author='Jean-Paul Watson, David Woodruff, Andrea Staid, Dominic Yang',
      author_email='jwatson@sandia.gov',
      packages=packages,
      entry_points={
            'console_scripts': [
                'runner.py = prescient_gosm.runner:main',
                'preprocessor.py = prescient_gosm.preprocessor:main',
                'populator.py = prescient_gosm.populator:main',
                'scenario_creator.py = prescient_gosm.scenario_creator:main',
                'markov_populator.py = prescient_gosm.markov_populator:main',
                'change_time_step.py = prescient_gosm.change_time_step:main'
            ]
        },
      install_requires=['numpy','matplotlib','pandas','scipy','pyomo','six',
                        'pyutilib', 'python-dateutil', 'networkx']
     )
