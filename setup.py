from pathlib import Path
from setuptools import setup, find_packages

jfi_path = (Path(__file__).parent / "jfi-JAXFriendlyInterface").absolute()

setup(
    name="mpcjax",
    version="0.1.0",
    packages=find_packages(),
    long_description=Path("README.md").read_text(),
    install_requires=[
        f"jfi @ file://{jfi_path}",
    ],
)
