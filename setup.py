from setuptools import setup, find_packages

setup(
    name='mpteval',
    version='1.0',
    description='Musically informed piano transcription metrics',
    packages=find_packages('src', exclude=['mpteval']),
)
