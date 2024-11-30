from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

REQUIRED = ["numpy==1.26.4", 
            "scipy==1.12.0",
            "partitura==1.5.0", # tmp version, wait for partitura release 1.6.0
            "matplotlib==3.7.0",
            "fastdtw==0.3.4",
]

setup(
    name="mpteval",
    version="0.1.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    include_package_data=True,
    description="Musically informed piano transcription metrics",
    author="Patricia Hu, Lukáš Samuel Marták, Carlos Cancino-Chacón",
    author_email="patricia.hu@jku.at",
    url="https://github.com/CPJKU/mpteval",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)