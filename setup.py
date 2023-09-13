from pathlib import Path
from setuptools import setup

# resolve README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

reqs = [
    "opencv-python",
    "numpy",
    "scipy",
    "matplotlib",
    "imageio",
    "imageio[ffmpeg]",
    "pyyaml",
    "xmltodict",
    "jupyter",
    "requests",
    "tqdm",
]

test_reqs = ["pytest"]


setup(
    name="afdme",
    version="1.0",
    author="Theodore Nowak",
    author_email="theodore.nowak@pnnl.gov",
    description="Acoustic Fish Detection - Marine Energy: Towards the automated detection of targets around marine energy sites.",
    package_dir={"": "src"},
    packages=["afdme"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsnowak/afd-me",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    install_requires=reqs,
    tests_require=test_reqs,
)
