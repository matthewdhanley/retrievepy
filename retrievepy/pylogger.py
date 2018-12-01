import logging
import logging.config
import os

CONFIG_FILE = 'logger.cfg'

def get_logger():
    """
    Create a logging object for easy logging
    :return: logging object
    """
    # set up LOGGER from config file
    if os.path.isfile(CONFIG_FILE):
        logging.config.fileConfig(CONFIG_FILE)
        logger = logging.getLogger('cube_ds')
    else:
        # use defaults if no config file
        format = '%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s'
        logging.basicConfig(format=format)
        logger = logging.getLogger('cube_ds')
        logger.warning(CONFIG_FILE+' not found. Using defaults for logging.')
    return logger