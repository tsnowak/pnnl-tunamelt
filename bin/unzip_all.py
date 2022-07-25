import os
import argparse
from pathlib import Path

from turbx import REPO_PATH

"""
Used to unzip and rename the per-video labels downloaded
individually from CVAT

python unzip_all.py -path /full/path/to/data/labels
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-path",
    "--path",
    dest="path",
    default=f"{REPO_PATH}/data/labels",
    type=str,
    help="path to files to unzip",
)

args = parser.parse_args()

path = Path(args.path)
for f in path.glob("**/*.zip"):
    unzip_cmd = f"unzip {f} -d {str(f.parent)}"
    rename_cmd = f"mv {str(f.parent)}/annotations.xml {str(f.parent)}/{f.stem}.xml"
    os.system(unzip_cmd)
    os.system(rename_cmd)
