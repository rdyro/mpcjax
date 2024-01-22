from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="mpcjax",
    version="0.4.0",
    packages=find_packages(),
    long_description=Path("README.md").read_text(),
    install_requires=["jaxfi", "redis", "numpy"],
)
