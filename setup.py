from setuptools import find_packages, setup

setup(
    name="dst",
    version="0.0.3.dev",
    description="Dynamic sparse training tools in PyTorch",
    author="Xin Wang",
    author_email="caseus_viridis@gmail.com",
    license="MIT",
    packages=find_packages('src', exclude=('tests', 'docs', 'experiments')),
    package_dir={'': 'src'},
    install_requires=(
        'torch',
        'torchvision',
        'torchtext',
        'numpy',
        'tqdm',
        'python-dotenv',
        'pytorch-ignite',
        'pytorch-monitor',
        'PTable',
    ),
    python_requires='>=3.4')
