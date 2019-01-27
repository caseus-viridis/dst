from setuptools import find_packages, setup

setup(
    name="dst",
    version="0.0.1",
    description="Dynamic sparse training",
    author="Xin Wang",
    author_email="xin@cerebras.net",
    license="MIT",

    packages=find_packages('src', exclude=('tests', 'docs', 'experiments')),
    package_dir={'': 'src'},

    install_requires=(
        'torch',
        'torchvision',
        'numpy',
        'tqdm',
        'PyYAML',
    ),

    python_requires='>= 3'
)
