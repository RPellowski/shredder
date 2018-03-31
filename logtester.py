import logging, logmetrics

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
    global logger
    logger = logmetrics.initLogger()
    log_debug_message()
    log_info_message()
    log_warning_message()

