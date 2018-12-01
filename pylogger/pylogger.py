import logging
import logging.config

CONFIG_FILE = 'share/cfg/logger.cfg'


def get_logger():
    """
    Create a logging object for easy logging
    :return: logging object
    """
    # set up LOGGER from config file
    # try
    logging.config.fileConfig(CONFIG_FILE)
    logger = logging.getLogger('retrievepy')
    # else:
    #     # use defaults if no config file
    #     format = '%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s'
    #     logging.basicConfig(format=format)
    #     logger = logging.getLogger('retrievepy')
    #     logger.warning(CONFIG_FILE+' not found. Using defaults for logging.')
    return logger
