import logging
import time

#from appmetrics import metrics

# https://docs.python.org/3/library/logging.handlers.html
# https://docs.python.org/3/howto/logging-cookbook.html#using-file-rotation
# http://docs.python-guide.org/en/latest/writing/logging/

import json
import logging.handlers
import platform

HOSTNAME = platform.node()

class MyFormatter(logging.Formatter):
    def __init__(self, task_name=None):
        self.task_name = task_name

        super(MyFormatter, self).__init__()

    def format(self, record):
        data = {'message': record.msg,
                'time': record.created,
                'timestring': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(record.created)),
                'filename': record.filename,
                'level': record.levelname,
                'function': record.funcName
                }
        #'asctime': time.asctime(time.gmtime(record.created)),
        return json.dumps(data)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = MyFormatter()

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(filename='metrics.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def hello():
    return "Hello World!\n"


def log_debug_message():
    logger.debug('debug message')
    return "Logged debug message.\n"


def log_info_message():
    logger.info('info message')
    return "Logged info message.\n"


def log_warning_message():
    logger.warning('warning message')
    return "Logged warning message\n"


if __name__ == '__main__':
    hello()
    log_debug_message()
    log_info_message()
    log_warning_message()

