from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="verifiable-instructions",
    version="0.1.0",
    description="Lightweight verifiable instructions for training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abukharin-nv/verifiable-instructions",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "nltk>=3.8",
        "langdetect>=1.0.9",
        "absl-py>=1.0.0",
        "immutabledict>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
)
