from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.3'
DESCRIPTION = 'Data Preprocessing library for Data Science and Machine Learning.'
LONG_DESCRIPTION = 'A package that provides all the data preprocessing tools important to Data scienece and ML tasks in one place'

# Setting up
setup(
    name="preprocessing_tools",
    version=VERSION,
    author="Antriksh Arya",
    author_email="<antriksh0704@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['scikit-learn', 'numpy', 'pandas', 'seaborn', 'matplotlib'],
    keywords=['python', 'data', 'preprocessing', 'data science', 'data analysis', 'machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)