from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="mpcjax",
    version="0.2.0",
    packages=find_packages(),
    long_description=Path("README.md").read_text(),
    install_requires=[
        "jfi @ git+https://github.com/rdyro/jfi-JAXFriendlyInterface",
    ],
)