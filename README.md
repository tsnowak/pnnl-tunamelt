# Acoustic Fish Detection - Marine Energy

This code accompanies the paper `Towards automating the detection of targets in acoustic camera video around tidal turbines` which proposes a novel, labeled data set and automated detection pipeline for detecting fish around operating tidal turbines.

# Getting Started

### Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### Install dependencies

```bash
pip install .
```

### Download the data set

```bash
# download and unpack the dataset into {REPO_PATH}/data
python scripts/download_dataset.py
```

### Running AFD-ME

Executables used to run our method and visualize results are at `scripts/` and `scripts/experiments`. Analysis of experimental results were often done inside jupyters notebooks which are located at `notebooks/`. Some examples of these scripts are:

```bash
# visualize AFD-ME
python /scripts/TODO/inference...
```

```bash
# rerun AFD-ME on the entire dataset
python /scripts/experiments/multirun.py
```

# More Information

## Creating symlinks for the data directories

The code as its written relies on a certain directory structure:

```bash
ln -s <path/to/my/data> $REPO_PATH/data
ln -s <path/to/my/data> $REPO_PATH/notebooks/data
```

```bash
ls $REPO_PATH/data
- mp4
- labels/cvat-video-1.1
```

Currently `cvat-video-1.1` is the only format of labels supported

## Data set label structure

During data set creation video labels are converted from `xml` files into python objects. Below is the structure of the python objects returned by the data loader:

```python
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

## About pCloud Data Downloads

### Manually downloading the data set

The AFD-ME data set proposed in this work is hosted on pcloud. A [web interface link](https://u.pcloud.link/publink/show?code=k76italK) is provided. For those who want a direct download link in order to download the data in a headless environment (such as a terminal session), a few steps must be taken.

### Creating a direct download link with pcloud

We will first use pcloud's web API to generate a direct download link. To do so, we will take the `code` parameter in web interface link, and pass it into the pcloud web API as shown below. **This link will always remain the same and can be copy/pasted.**

```
https://api.pcloud.com/getpublinkdownload?code=k76italK&forcedownload=0
```

This will return a page like the below. **This return will change for each request, as the direct download link always expires in one day!**

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

We can then choose a host to download from either `p-def4.pcloud.com` or `c383.pcloud.com` and append the `path` with backslashes removed - `/cBZnt4E1dZHhHGlcZZZj05Mo7Zg5ZZGiFZkZnR6fMJZR4ZszZ7zZPHZuHZSpZ3HZVFZlpZY5ZbLZ04ZTRZC4Ztd8tVZhaYWhc86DPXLdkfDV0Q8PQaGYeqk/AFD-ME.tar.gz` - to create the direct download link. The URL for this generated (and likely now expired) direct download path is given below. This can then be used with `wget` to download the data set in a terminal session.

```
https://p-def4.pcloud.com/cBZnt4E1dZHhHGlcZZZMG0Mo7Zg5ZZGiFZkZnR6fMJZR4ZszZ7zZPHZuHZSpZ3HZVFZlpZY5ZbLZ04ZTRZC4Ztd8tVZ6wcSuxqImyBvBT1so6vuBLEV23UX/AFD-ME.tar.gz
```

## About the Computer Vision Annotation Tool (CVAT)

CVAT is an open-source image and video labeling tool that can be downloaded and stood up. We used CVAT's docker image to create a video labeling platform for our expert-annotators to identify targets of interest in video.

### Exporting Video Annotations from CVAT

Exporting all project annotations does not preserve per-video frame information. We therefore have to export each task manually. To do so using the CVAT CLI create a python environment [via the linked](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/cli/#usage) then use the below to export annotations from a specific task (or just download annotations for each video by hand):

```bash
# format = CVAT for images 1.1
# task = 103
cli.py dump --format "CVAT for images 1.1" 103 output.zip
```
