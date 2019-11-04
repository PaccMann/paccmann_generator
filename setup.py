"""Install package."""
from setuptools import setup, find_packages

setup(
    name='paccmann_generator',
    version='0.0.1',
    description='Multimodal generative models for PaccMann^RL.',
    long_description=open('README.md').read(),
    url='https://github.com/PaccMann/paccmann_generator',
    author='Jannis Born, Matteo Manica, Ali Oskooei, Joris Cadow',
    author_email=(
        'jab@zurich.ibm.com, drugilsberg@gmail.com, '
        'ali.oskooei@gmail.com, joriscadow@gmail.com'
    ),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pytoda>=0.0.1',
        'torch>=1.0.0'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)
