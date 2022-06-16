from io import open
from setuptools import find_packages, setup

block_list = {
    'bloomberg.ds.katiecli',
    'bloomberg.ds.platformsdk',
    'ipython',
    'bloomberg.ml.bagels',
    'tensorflow'
}
with open('requirements.txt') as requirements:
    INSTALL_REQUIRES = [s.strip() for s in requirements if s.split('=')[0] not in block_list]

setup(
    name="source-exploration",
    version="0.0.1",
    author="",
    author_email="aspangher@bloomberg.net",
    description="Models for learning controlled generation.",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    # long_description_content_type="text/markdown",
    url="https://bbgithub.dev.bloomberg.com/aspangher/controlled-sequence-gen",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            "transformers=transformers.__main__:main",
        ]
    },
    # python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)