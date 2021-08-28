
import os
import yaml
import logging
import logging.config
from pathlib import Path

# create some helpful global vars
REPO_PATH=str(Path(Path(__file__).parents[2]).absolute())

logging_config = os.path.join(REPO_PATH, 'src/fish/logging.yaml')
with open(logging_config, 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger("std_logger") # gets the root logger by default
logger.debug("Loaded logging config file: %s" % (logging_config))