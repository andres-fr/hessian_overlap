#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module features a convenient color logger (wrapping the built-in
``loging.`` class), plus some helper functionality.
"""


import datetime
import json
import logging
import os
import socket
import sys
from typing import Optional

import pytz

#
# ignore "mypy" import error for coloredlogs.
# see https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
import coloredlogs  # type: ignore


# ##############################################################################
# # COLOR LOGGER
# ##############################################################################
def make_timestamp(timezone="Europe/Berlin", with_tz_output=False):
    """
    Output example: day, month, year, hour, min, sec, milisecs:
    10_Feb_2018_20:10:16.151
    """
    ts = datetime.datetime.now(tz=pytz.timezone(timezone)).strftime(
        "%Y_%m_%d_%H_%M_%S.%f"
    )[:-3]
    if with_tz_output:
        return "%s(%s)" % (ts, timezone)
    else:
        return ts


class HostnameFilter(logging.Filter):
    """
    Needed to include hostname into the logger. See::
      https://stackoverflow.com/a/55584223/4511978
    """

    def filter(self, record) -> bool:
        record.hostname = socket.gethostname()
        return True


class ColorLogger:
    """
    This class:
    1. Creates a ``logging.Logger`` with a convenient configuration.
    2. Attaches ``coloredlogs.install`` to it for colored terminal output
    3. Provides some wrapper methods for convenience
    Usage example::
      # create 2 loggers
      cl1 = ColorLogger("term.and.file.logger", "/tmp/test.txt")
      cl2 = ColorLogger("JustTermLogger")
      # use them at wish
      cl1.logger.debug("this is a debugging message")
      cl2.logger.info("this is an informational message")
      cl1.logger.warning("this is a warning message")
      cl2.logger.error("this is an error message")
      cl1.logger.critical("this is a critical message")
    """

    FORMAT_STR = (
        "%(asctime)s.%(msecs)03d %(hostname)s: %(name)s"
        + "[%(process)d] %(levelname)s %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def get_logger(
        self,
        logger_name,
        logfile_dir: Optional[str],
        filemode: str = "a",
        logging_level: int = logging.DEBUG,
    ) -> logging.Logger:
        """
        :param filemode: In case ``logfile_dir`` is given, this specifies the
          output mode (e.g. 'a' for append).
        :returns: a ``logging.Logger`` configured to output all events at level
          ``self.logging_level`` or above into ``sys.stdout`` and (optionally)
          the given ``logfile_dir``, if not None.
        """
        # create logger, formatter and filter, and set desired logger level
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(self.FORMAT_STR, datefmt=self.DATE_FORMAT)
        hostname_filter = HostnameFilter()
        logger.setLevel(logging_level)
        # create and wire stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(hostname_filter)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        # optionally, create and wire file handler
        if logfile_dir is not None:
            basename = (
                make_timestamp(with_tz_output=False) + logger_name + ".log"
            )
            logfile_path = os.path.join(logfile_dir, basename)
            # create one handler for print and one for export
            file_handler = logging.FileHandler(logfile_path, filemode)
            file_handler.addFilter(hostname_filter)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        #
        return logger

    def __init__(
        self,
        logger_name: str,
        logfile_path: Optional[str] = None,
        filemode: str = "a",
        logging_level: int = logging.DEBUG,
    ):
        """
        :param logger_name: A process may have several loggers. This parameter
          distinguishes them.
        :param logfile_path: Where to write out.
        """
        self.logger: logging.Logger = self.get_logger(
            logger_name, logfile_path, filemode, logging_level
        )
        #
        coloredlogs.install(
            logger=self.logger, fmt=self.FORMAT_STR, level=logging_level
        )

    # a few convenience wrappers:
    def debug(self, *args, **kwargs) -> None:
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs) -> None:
        self.logger.critical(*args, **kwargs)


class JsonColorLogger(ColorLogger):
    """
    Less human-friendly, more machine-friendly prompts
    """

    FORMAT_STR = """["%(asctime)s.%(msecs)03d", %(message)s]"""
    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"

    def loj(self, header, body):
        """ """
        self.info(json.dumps((header, body)))

    @staticmethod
    def read_file(
        path,
        body_fn=lambda x: x,
        filter_headers=None,
        ts_colname="TIMESTAMP",
        header_colname="HEADER",
    ):
        """
        Load a log file into a pandas table. The file is assumed to be created
        by repeatedly calling the ``loj`` method, hence having lines in the
        form ``[timestamp, [header, body]]``.

        :param filter_headers: Collection of allowed headers. If given, only
          entries with these headers will be gathered.
        :param body_fn: The ``body`` entry from each line will be passed to
          this function for processing. The function must return a dictionary,
          such that all the returned key-value pairs will be aggregated into
          the output pandas table. If the body is already in the form of a
          dictionary, and you want to preserve all key-value pairs, this
          function is simply the identity. It can be used to filter out
          entries, rename keys, etc.
        :returns: A pandas dataframe, with ``ts_colname, header_colname``
          columns, as well as one column per key returned by ``body_fn``.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            print("This method requires pandas to be installed!")
            return
        # first gather a list of [timestamp, [header, body]] entries
        entries = pd.read_json(path, lines=True).values.tolist()
        result = []
        for i, (ts, (header, body)) in enumerate(entries):
            if (filter_headers is not None) and (header not in filter_headers):
                continue
            body = body_fn(body)
            assert (ts_colname not in body) and (
                header_colname not in body
            ), "Colliding timestamp/header column names! {body.keys()}"
            result.append({ts_colname: ts, header_colname: header, **body})
        #
        result = pd.DataFrame(result)
        return result
