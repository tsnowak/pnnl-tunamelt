import os
import requests
import json
import tarfile
from tqdm import tqdm
from pathlib import Path

from tunamelt import REPO_PATH, log

"""
Used to download and extract the data set to {REPO_PATH}/data
"""


def download(url, out):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(out, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


log.warning(
    f"Code in this repository expects PNNL-TUNAMELT to be under {REPO_PATH}/data. Create a symbolic link at {REPO_PATH}/data to store data on other drives."
)

url = "https://api.pcloud.com/getpublinkdownload?code=k76italK&forcedownload=0"

# get a url to directly download the dataset
log.info("Fetching download URL...")
resp = requests.get(url)
resp = json.loads(resp.content)
dlurl = "https://" + resp["hosts"][0] + resp["path"]

# create the data directory
data_path = Path(f"{REPO_PATH}/data")
data_path.mkdir(exist_ok=True)
file_path = Path(str(data_path) + "/" + "PNNL-TUNAMELT.tar.gz")
# download the dataset if tar file doesn't exist or tar file size is less than 6.38GB
if (not file_path.exists()) or (not (os.path.getsize(str(file_path)) > (6.38e9))):
    log.info("Downloading dataset...")
    download(dlurl, f"{str(file_path)}")
else:
    log.info("Dataset already downloaded, skipping.")

log.info("Extracting dataset...")
tar = tarfile.open(f"{str(file_path)}")
tar.extractall(path=str(data_path))
tar.close()
