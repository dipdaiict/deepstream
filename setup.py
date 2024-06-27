from typing import List
from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="DeepStream",
    version="0.0.1",
    author="Dip Patel",
    author_email="dippatel256@gmail.com",
    description="An end-to-end deep learning project",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/dipdaiict/deepstream",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=get_requirements(r'C:\Self-Learning\deepstream\requirements_dev.txt'),
    python_requires='>=3.12',
)
