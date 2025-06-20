import os
import argparse
from pathlib import Path

from tunamelt import REPO_PATH

"""
Used to unzip and rename the per-video labels downloaded
individually from CVAT.

python unzip_all.py -path /full/path/to/data/PNNL-TUNAMELT/labels
"""

# path to label files
parser = argparse.ArgumentParser()
parser.add_argument(
    "-path",
    "--path",
    dest="path",
    default=f"{REPO_PATH}/data/PNNL-TUNAMELT/labels",
    type=str,
    help="path to files to unzip",
)

args = parser.parse_args()

# unzip and rename annotations.xml as video.xml
path = Path(args.path)
for f in path.glob("**/*.zip"):
    unzip_cmd = f"unzip {f} -d {str(f.parent)}"
    rename_cmd = f"mv {str(f.parent)}/annotations.xml {str(f.parent)}/{f.stem}.xml"
    os.system(unzip_cmd)
    os.system(rename_cmd)
