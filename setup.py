"""Package setup for the AI_Supermarket_BI project.

Allows the project to be installed in editable mode (``pip install -e .``)
so that the ``src`` package can be imported from anywhere.
"""
from pathlib import Path

from setuptools import find_packages, setup

_HERE = Path(__file__).resolve().parent


def _load_requirements():
    """Read pinned dependencies from requirements.txt, skipping comments."""
    req_file = _HERE / "requirements.txt"
    if not req_file.exists():
        return []
    lines = req_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _read_long_description():
    """Use the README as the package long description when available."""
    readme = _HERE / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "AI-Powered Supermarket Analytics & Business Intelligence System"


setup(
    name="ai_supermarket_bi",
    version="1.0.0",
    description="AI-Powered Supermarket Analytics & Business Intelligence System",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="AI Supermarket BI Team",
    license="MIT",
    packages=find_packages(include=["src", "src.*", "tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=_load_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="supermarket analytics business-intelligence forecasting segmentation streamlit",
)
