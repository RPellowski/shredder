import logging, logging.handlers
import time
import json
import platform

# globals
metrics_logger = None
HOSTNAME = platform.node()

import resource

def unix_time():
    return (time.time(), resource.getrusage(resource.RUSAGE_SELF))

def unix_time_elapsed(t0, t1):
    return {'unix_time_elapsed' : {
               'real': t1[0] - t0[0],
               'sys': t1[1].ru_stime - t0[1].ru_stime,
               'user': t1[1].ru_utime - t0[1].ru_utime
               }
           }

# ISO 8601 same as '%Y-%m-%dT%H:%M:%SZ'
DATEFMT = '%FT%TZ'
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
                'function': record.funcName,
                'hostname': HOSTNAME
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

    # console handler
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

    # file handler
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

