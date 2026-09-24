"""Application-wide logging built on loguru.

Usage in any project module::

    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("something happened")

Features
--------
* Coloured console output (level taken from ``LOG_LEVEL``, default INFO).
* File output to ``logs/app.log`` with 10 MB rotation and 30-day retention.
* Uniform record format ``timestamp | LEVEL | module | message``.
* Sinks never raise into application code (``catch=True``), honouring the
  project rule that the pipeline must not crash because of logging.
"""
import sys
from typing import Any

from loguru import logger as _loguru_logger

from src import config

#: Single format string used by both the console and the file sink.
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"

#: Guard so repeated imports / Streamlit script reruns never duplicate sinks.
_configured = False


def _configure_logger() -> None:
    """Install the console and rotating-file sinks exactly once."""
    global _configured
    if _configured:
        return

    try:
        config.LOG_PATH.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - disk/permission failure
        _loguru_logger.warning("Could not create log directory {}: {}", config.LOG_PATH, exc)

    _loguru_logger.remove()  # drop loguru's default stderr sink

    # --- console sink (coloured) ------------------------------------------
    _loguru_logger.add(
        sys.stderr,
        level=config.LOG_LEVEL.upper(),
        format=LOG_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        catch=True,
    )

    # --- rotating file sink ------------------------------------------------
    _loguru_logger.add(
        str(config.LOG_PATH / "app.log"),
        level="DEBUG",
        format=LOG_FORMAT,
        rotation="10 MB",
        retention="30 days",
        compression=None,
        enqueue=False,
        backtrace=False,
        diagnose=False,
        catch=True,
    )

    _configured = True


def get_logger(module_name: str = "src") -> Any:
    """Return a loguru logger tagged with ``module_name``.

    Parameters
    ----------
    module_name:
        Typically ``__name__`` of the calling module. It is bound to the
        record's extra fields and surfaces automatically in the ``{name}``
        slot of the log format.

    Returns
    -------
    loguru.Logger
        A bound logger instance; call ``.debug/.info/.warning/.error`` on it.
    """
    _configure_logger()
    return _loguru_logger.bind(module_name=module_name)


if __name__ == "__main__":
    # Manual smoke check: writes one record per level to console and file.
    demo = get_logger("src.logger")
    demo.debug("DEBUG level message (file sink only unless LOG_LEVEL=DEBUG)")
    demo.info("INFO level message")
    demo.success("SUCCESS level message")
    demo.warning("WARNING level message")
    demo.error("ERROR level message")
    print(f"Log file: {config.LOG_PATH / 'app.log'}")
