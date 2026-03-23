import logging
from colorlog import ColoredFormatter
from datetime import datetime
import os

class NavLogger:

    @staticmethod
    def create(
        name: str = "app",
        log_dir: str = None,        # directory to place log file
        level: int = logging.DEBUG,
    ) -> logging.Logger:

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # ---------------------------------------------------------
        # 1. Build the log file path with timestamp
        # ---------------------------------------------------------
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = os.path.join(log_dir, f"{timestamp}.log")
        else:
            log_path = None   # no file logging if directory missing

        # ---------------------------------------------------------
        # 2. Colored formatter
        # ---------------------------------------------------------
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        # ---------------------------------------------------------
        # 3. Console handler (color)
        # ---------------------------------------------------------
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # ---------------------------------------------------------
        # 4. File handler (ANSI colors saved)
        # ---------------------------------------------------------
        if log_path is not None:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

        return logger
