"""
Setup script for Intersection environment package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="intersection-env",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-agent reinforcement learning environment for traffic intersection simulation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ShamG1/marl-traffic-intersection",
    packages=find_packages(exclude=["MAPPO", "MAPPO.*", "tests", "tests.*"]),
    package_data={
        'Intersection': ['assets/*.png', 'assets/*.ico', 'assets/*.jpg', 'assets/*.jpeg'],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "pygame>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "intersection-manual-test=Intersection.manual_test:main",
            "intersection-traffic-test=Intersection.traffic_test:main",
        ],
    },
)
