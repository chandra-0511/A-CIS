from setuptools import setup, find_packages

setup(
    name="acis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'PyQt5',
        'pyqtgraph'
    ],
)