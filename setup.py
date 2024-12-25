import setuptools
from setuptools import setup

# long description을 README.md로 대체하기 위한 작업
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(

	# module 이름
    name='', 
    
    # version 명시
    version='0.0.1',
    
    # package에 대한 짧은 description
    description='',
    
    # package에 대한 자세한 description
    # README.md로 대체한다
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # package 저자 이름
    author='Wonjun, Oh',
    
    # package 저자 이메일 
    author_email='owj0421@naver.com',
    
    # package url (ex: github)
    url='',
    license='MIT',
    packages=setuptools.find_packages(),
    
    # 파이썬 버전
    python_requires='>=3.10',  
    
    install_requires = [
        "pytorch==2.5.0",
        "torchvision==0.20.0",
        "torchaudio==2.5.0",
        "pillow==11.0.0",
        "transformers==4.46.1",
        "wandb",
        "tqdm",
        "scikit-learn",
        "gradio",
        "faiss-gpu",
    ],
    classifiers = [
                      "Programming Language :: Python :: 3",
                      "License :: OSI Approved :: MIT License",
                      "Operating System :: OS Independent"
    ]
)