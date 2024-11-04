from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="atlas_hfmodels",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        # "huggingface-hub",
        # "transformers",
        # "torch"
    ],
    author="Haopeng Yu",
    author_email="atlasbioin4@gmail.com",
    description="Manage your models on the Hugging Face Hub",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/atlas_hfmodels",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'atlas_hfmodels=atlas_hfmodels.atlas_hfmodels:main',
        ],
    },
)