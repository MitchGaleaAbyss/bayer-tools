#!/usr/bin/env python3


from distutils.core import setup
from typing import List, Optional, Tuple


setup(
    name="bayer_tools",
    version="1.0.0",
    description="tools for processing bayer images",
    author="Mitch Galea",
    author_email="m.galea@abysssolutions.com.au",
    url="http://github.com/abyss-solutions/abyss-robotics",
    packages=[
        "bayer_tools.dead_pixels",
    ],
    install_requires=[
        "opencv-python",
    ],
)
