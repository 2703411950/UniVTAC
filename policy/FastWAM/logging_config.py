"""Minimal logger shim — replaces fastwam.utils.logging_config."""
import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(log_level: int = logging.INFO, is_main_process: bool = True):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
