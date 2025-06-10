import os
import yaml
import logging
import logging.config
from pathlib import Path

# create some helpful global vars
path_to_file = Path(__file__).absolute()
if "site-packages" in str(path_to_file):
    # if pip install .
    # WARNING: possible errors with pathing
    REPO_PATH = str(path_to_file.parent)
else:
    # PREFERRED: if pip install -e .
    REPO_PATH = str(path_to_file.parents[2])

logging_config = os.path.join(str(path_to_file.parent), "config/logging.yaml")
print(logging_config)
with open(logging_config, "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

log = logging.getLogger("std_logger")  # gets the root logger by default
log.debug("Loaded logging config file: %s" % (logging_config))
