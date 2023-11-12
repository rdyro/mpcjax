from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="mpcjax",
    version="0.3.1",
    packages=find_packages(),
    long_description=Path("README.md").read_text(),
    install_requires=["jaxfi"],
)
