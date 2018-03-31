import logging
import time

#from appmetrics import metrics

# https://docs.python.org/3/library/logging.handlers.html
# https://docs.python.org/3/howto/logging-cookbook.html#using-file-rotation
# http://docs.python-guide.org/en/latest/writing/logging/

import json
import logging.handlers
import platform

# globals
metrics_logger = None
HOSTNAME = platform.node()

'''
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp


def unix_time(function, args=tuple(), kwargs={}):
    Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    You can calculate the amount of used CPU-time used by your
    function/callable by summing `user` and `sys`. `real` is just like the wall
    clock.
    Note that `sys` and `user`'s resolutions are limited by the resolution of
    the operating system's software clock (check `man 7 time` for more
    details).

    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    function(*args, **kwargs)
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()

    return {'real': end_time - start_time,
            'sys': end_resources.ru_stime - start_resources.ru_stime,
            'user': end_resources.ru_utime - start_resources.ru_utime}

'''
DATEFMT = '%Y-%m-%dT%H:%M:%SZ'
FMT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'

class json_formatter(logging.Formatter):
    def __init__(self, task_name=None):
        self.task_name = task_name

        super(json_formatter, self).__init__()

    def format(self, record):
        data = {'message': record.msg,
                'time': record.created,
                'timestring': time.strftime(DATEFMT, time.gmtime(record.created)),
                'filename': record.filename,
                'level': record.levelname,
                'function': record.funcName
                }
        #'asctime': time.asctime(time.gmtime(record.created)),
        return json.dumps(data)

def initLogger(logger_level = logging.DEBUG,
               console_level = logging.DEBUG,
               console_logging = 'text',
               file_level = logging.DEBUG,
               file_logging = 'json',
               file_out = 'metrics.log'):

    # create logger
    metrics_logger = logging.getLogger(__name__)
    metrics_logger.setLevel(logger_level)

    if console_logging != 'none':
        # create handler for console output
        console_handler = logging.StreamHandler()
        if console_logging == 'json':
            console_handler.setFormatter(json_formatter())
        else:
            # use basic formatter
            console_handler.setFormatter(logging.Formatter(fmt = FMT,
                                                           datefmt = DATEFMT))
        console_handler.setLevel(console_level)
        metrics_logger.addHandler(console_handler)

    if file_logging != 'none' and not (file_out == '' or file_out is None):
        # create handler for file output
        file_handler = logging.FileHandler(filename=file_out)
        if file_logging == 'json':
            file_handler.setFormatter(json_formatter())
        else:
            # use basic formatter
            file_handler.setFormatter(logging.Formatter(fmt = FMT,
                                                        datefmt = DATEFMT))
        file_handler.setLevel(file_level)
        metrics_logger.addHandler(file_handler)

    return metrics_logger

