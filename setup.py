# Copyright (C) 2018 Arno Onken
#
# This file is part of the mmae package.
#
# The mmae package is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# The mmae package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
'''
Setup for the mmae package.

'''
from setuptools import setup

setup(
    name="mmae",
    version="0.1.0",
    description=("Package for multimodal autoencoders with Bregman"
                 " divergences."),
    long_description=open('README.rst').read(),
    keywords="autoencoder multimodal deep learning bregman",
    url="https://github.com/asnelt/mmae/",
    author="Arno Onken",
    author_email="asnelt@asnelt.org",
    license="GPLv3+",
    packages=["mmae"],
    install_requires=[
        "numpy",
        "six",
        "keras"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        ("License :: OSI Approved :: GNU General Public License v3 or later"
         " (GPLv3+)"),
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"]
)
