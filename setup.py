"""
setup.py — Gold Grid Trading Bot Package Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="gold-grid-trading-bot",
    version="1.0.0",
    author="Gold Grid Trading Bot Team",
    author_email="dev@goldgridbot.example.com",
    description="AI Adaptive Grid Trading System for Gold (GLD ETF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gold-grid-trading-bot",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="trading gold grid-trading algorithmic-trading machine-learning",
    entry_points={
        "console_scripts": [
            "gold-collect=scripts.data_collection:main",
            "gold-train=scripts.train_models:main",
            "gold-backtest=scripts.backtest:main",
            "gold-optimize=scripts.optimize:main",
        ]
    },
)
