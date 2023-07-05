import os
import yaml
import logging
import logging.config
from pathlib import Path

# create some helpful global vars
REPO_PATH = str(Path(Path(__file__).parents[2]).absolute())

logging_config = os.path.join(REPO_PATH, "src/afdme/logging.yaml")
with open(logging_config, "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

log = logging.getLogger("std_logger")  # gets the root logger by default
log.debug("Loaded logging config file: %s" % (logging_config))
