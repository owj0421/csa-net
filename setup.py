import setuptools
from setuptools import setup
from setuptools import setup, find_packages

setup_requires = [
]

install_requires = [
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
    "pillow>=11.0.0",
    "transformers>=4.46.1",
    "wandb",
    "tqdm",
    "scikit-learn",
    "gradio",
    "faiss-gpu",
],

# long description을 README.md로 대체하기 위한 작업
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    version='0.0.16',
    
    name='fashion_recommenders', 
    description='An easy-to-use PyTorch-based package for fashion recommendation models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author='Wonjun, Oh',
    author_email='owj0421@naver.com',
    packages=find_packages(where='src'),
    
    url='https://github.com/owj0421/fashion_recommenders',
    license='MIT',

    python_requires='>=3.10',
    setup_requires=setup_requires,
    install_requires = install_requires,
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)