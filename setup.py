from setuptools import find_packages, setup

setup(
    name="dst",
    version="0.0.1",
    description="Dynamic sparse training tools for PyTorch",
    author="Xin Wang",
    author_email="caseus.viridis@gmail.com",
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

    python_requires='>= 3.4'
)
