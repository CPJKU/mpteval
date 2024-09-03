from setuptools import setup, find_packages

setup(
    name='tri24',
    version='1.0',
    description='Musically informed piano transcription metrics',
    packages=find_packages('src', exclude=['tri24']),
)
