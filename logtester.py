import logging, logmetrics

def log_debug_message():
    logger.debug('debug message')

def log_info_message():
    logger.info('info message')

def log_warning_message():
    logger.warning('warning message')

from appmetrics import metrics
import time
import random

def test_histogram():
    @metrics.with_histogram("test1")
    def my_worker():
        time.sleep(random.random())

    # test timing histogram
    my_worker()
    my_worker()
    my_worker()
    logger.debug(metrics.get("test1"))

    # test data histogram
    histogram = metrics.new_histogram("test2")
    histogram.notify(1.0)
    histogram.notify(2.0)
    histogram.notify(3.0)
    logger.debug(histogram.get())

def test_counter():
    counter = metrics.new_counter("test")
    counter.notify(10)
    logger.debug(counter.get())

def test_gauge():
    gauge = metrics.new_gauge("gauge_test")
    gauge.notify("version 1.0")
    logger.debug(gauge.get())

def test_meter():
    meter = metrics.new_meter("meter_test")
    meter.notify(1)
    meter.notify(1)
    meter.notify(3)
    logger.debug(meter.get())


if __name__ == '__main__':
    global logger
    logger = logmetrics.initLogger() #console_logging='json', file_logging='none')
    log_debug_message()
    log_info_message()
    log_warning_message()

    t0 = logmetrics.unix_time()
    for i in range(100000):
        pass
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))

    test_histogram()
    test_gauge()
    test_counter()
    test_meter()

# testing strategy for automation
# logging.disable() default is CRITICAL for 3.7
# logging.disable(logging.NOTSET)

