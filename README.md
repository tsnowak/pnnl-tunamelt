
# Acoustic Fish Detection - Marine Energy

This code accompanies the paper `On the automated detection of targets in acoustic camera video around an experimental tidal turbine: a labeled data set of targets and automated detection approach` which proposes a novel, labeled data set and automated detection pipeline for detecting fish around operating tidal turbines.

Instructions for downloading the data set and setting up the development environment used in this work are described below.

# Getting Started

## Downloading the Data Set

The AFD-ME data set proposed in this work is hosted on pcloud. A [web interface link](https://u.pcloud.link/publink/show?code=k76italK) is provided, but we need to create a direct download link if we want to download the data set via terminal.

We will use pcloud's web API to generate a direct download link. To do so, we will take the `code` parameter in web interface link, and pass it into the pcloud web API as shown below. **Note, this link will always remain the same, so feel free to copy and paste!**
```
https://api.pcloud.com/getpublinkdownload?code=k76italK&forcedownload=0
```
This will return a page like the below. **This will change for each request, as the direct download link always expires in one day:**
```
{
	"result": 0,
	"expires": "Wed, 03 May 2023 04:06:08 +0000",
	"dwltag": "ywtWXDIDIP8bkcqlT1KG35",
	"path": "\/cBZnt4E1dZHhHGlcZZZj05Mo7Zg5ZZGiFZkZnR6fMJZR4ZszZ7zZPHZuHZSpZ3HZVFZlpZY5ZbLZ04ZTRZC4Ztd8tVZhaYWhc86DPXLdkfDV0Q8PQaGYeqk\/AFD-ME.tar.gz",
	"hosts": [
		"p-def4.pcloud.com",
		"c383.pcloud.com"
	]
}
```
We can then choose a host to download from - in this case either `p-def4.pcloud.com` or `c383.pcloud.com` - and append the `path` with backslashes removed - `/cBZnt4E1dZHhHGlcZZZj05Mo7Zg5ZZGiFZkZnR6fMJZR4ZszZ7zZPHZuHZSpZ3HZVFZlpZY5ZbLZ04ZTRZC4Ztd8tVZhaYWhc86DPXLdkfDV0Q8PQaGYeqk/AFD-ME.tar.gz` - to create the direct download link. The URL for this generated (and likely now expired) direct download path is given below. This can then be used with `wget` to download the data set in a terminal session.
```
https://p-def4.pcloud.com/cBZnt4E1dZHhHGlcZZZMG0Mo7Zg5ZZGiFZkZnR6fMJZR4ZszZ7zZPHZuHZSpZ3HZVFZlpZY5ZbLZ04ZTRZC4Ztd8tVZ6wcSuxqImyBvBT1so6vuBLEV23UX/AFD-ME.tar.gz
```

## Downloading conda

All package dependencies for this code base are handled by `conda` and `pip`. To use this repository, these managers will need to be installed:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Miniconda setup script downloads. Will also contain pip/PyPi.

## Conda environment setup

Once `conda` is setup on your machine, you can go ahead and setup the environment.

``` bash
# creates a conda environment for this code
conda create -n turbx python=3.9

# activates the new environment
conda activate turbx

# installs the software for this project into the environment
conda env update --file env.yml
```

## Data directories symlink

The code as its written relies on a certain directory structure: 

``` bash
ln -s <path/to/my/data> $REPO_PATH/data
ln -s <path/to/my/data> $REPO_PATH/notebooks/data
```

``` bash
ls $REPO_PATH/data
- mp4
- labels/cvat-video-1.1
```
Currently `cvat-video-1.1` is the only format of labels supported

# Data Set

## Data Label Structure

During data set creation video labels are converted from `xml` files into python objects. Below is the structure of the python objects returned by the data loader:

``` python
{'test': [
    {
        'filename': '2010-09-08_074500_HF_S002_S001.mp4',
        'tracks': [
            {
                'frames': [
                    {'box': ((486, 1011), (542, 1062)),
                    'frame': 16,
                    'keyframe': 1,
                    'occluded': 0,
                    'outside': 0},
                    {'box': ((455, 1016), (511, 1067)), ...
                ]
                'label': 'target',
                'track_id': 0
            },
                'frames': ...
        ],
        'video_id': 12,
        'video_length': 160,
        'video_shape': {'height': 1792, 'width': 1032}
    }
]
} 
```

# CVAT

CVAT is an open-source image and video labeling tool that can be downloaded and stood up. We used CVAT's docker image to create a video labeling platform for our expert-annotators to identify targets of interest in video.

## Exporting Video Annotations from CVAT

Exporting all project annotations does not preserve per-video frame information. We therefore have to export each task manually. To do so using the CVAT CLI create a python environment [via the linked](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/cli/#usage) then use the below to export annotations from a specific task (or just download annotations for each video by hand):

``` bash
# format = CVAT for images 1.1
# task = 103
cli.py dump --format "CVAT for images 1.1" 103 output.zip
```
