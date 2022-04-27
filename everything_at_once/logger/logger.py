import logging
import logging.config
import os.path
from pathlib import Path
from everything_at_once.utils import read_json


def setup_logging(save_dir, log_config=None, default_level=logging.INFO):
    """
    Setup logging configuration
    """
    if log_config is None:
        log_config = os.path.join(os.path.dirname(__file__), 'logger_config.json')

    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
