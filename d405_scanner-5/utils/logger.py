"""
utils/logger.py
Strukturiertes Logging für den gesamten Scan-Workflow.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "d405_scanner", log_dir: Path = Path("output")) -> logging.Logger:
    """
    Erstellt einen Logger mit Console- und File-Handler.
    Gibt farbige Ausgabe in der Console (falls unterstützt).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"scan_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Verhindert doppelte Handler bei mehrfachem Aufruf
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console Handler (INFO+)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColorFormatter())
    logger.addHandler(console)

    # File Handler (DEBUG+)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logger gestartet → {log_file}")
    return logger


class _ColorFormatter(logging.Formatter):
    """ANSI-Farben für Console-Output."""
    COLORS = {
        logging.DEBUG:    "\033[37m",   # Grau
        logging.INFO:     "\033[36m",   # Cyan
        logging.WARNING:  "\033[33m",   # Gelb
        logging.ERROR:    "\033[31m",   # Rot
        logging.CRITICAL: "\033[1;31m", # Fett Rot
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)s │ %(message)s",
            datefmt="%H:%M:%S"
        )
        return fmt.format(record)


# Modul-weiter Standard-Logger
log = setup_logger()
