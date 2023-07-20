
import logging


def get_logger(name, logging_level=logging.INFO):
    logger = logging.getLogger(name)

    if not logger.handlers:  # check if the logger has any handlers
        formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.setLevel(logging_level)
        logger.addHandler(handler)
    
    return logger