from setuptools import setup, find_packages

setup(
    name="qiskit_nature_cp2k",
    version="0.0.1",
    description="An integration plugin for Qiskit Nature and CP2K",
    author="Max Rossmannek, Stefano Battaglia",
    author_email="oss@zurich.ibm.com, stefano.battaglia@chem.uzh.ch",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "qiskit-nature~=0.7.0",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)