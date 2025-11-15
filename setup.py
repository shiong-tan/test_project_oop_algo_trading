from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantml-trader",
    version="2.0.0",
    author="Shiong Tan",
    author_email="shiong.tan@example.com",
    description="Production-ready quantitative ML trading system with comprehensive risk management and backtesting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shiong-tan/quantml-trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=0.990",
        ],
        "advanced": [
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantml-trader=algo_trading.cli:main",
        ],
    },
)
