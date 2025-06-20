import logging
import sys


def get_logger(name, workdir, level=logging.INFO, mode="w"):
    """
    Set up logger
    Args: 
        mode: "w" to overwrite existing file, "a": append existing file. 
    """
    logger = logging.getLogger(name)  # noqa: F821
    logger.setLevel(level)
    file_handler = logging.FileHandler(f"{workdir}/{name}.log", mode=mode)
    console_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def convert_sec_hms(seconds):
    """
    convert time from secone to hour - minutes - seconds
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return int(hours), int(minutes), int(remaining_seconds)