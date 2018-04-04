import sys
import time
import logging
import logmetrics
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from appmetrics import metrics

FS_BUCKETS = ['code', 'doc', 'other']
EXT = {
      '.py'  : 'code',
      '.md'  : 'doc',
      '.txt' : 'doc',
      '.log' : 'skip',
      '.pyc' : 'skip',
      }

class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        e = [v for k,v in EXT.items() if event.src_path.endswith(k)]
        if event.is_directory or len(e) == 0:
            metrics.metric('other').notify(1)
        elif not (e[0] is 'skip'):
            metrics.metric(e[0]).notify(1)
        #logger.info({"event" : [event.event_type, event.is_directory, event.src_path]})


if __name__ == "__main__":

    global logger
    logger = logmetrics.initLogger(console_logging='json', file_out = 'activity.log')

    global counters
    counters = [metrics.new_counter(f) for f in FS_BUCKETS]

    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    observer.start()
    try:
        while True:
            time.sleep(60)
            for f in FS_BUCKETS:
                if metrics.metric(f).raw_data() > 0:
                    logger.info({f: metrics.get(f)})
                    metrics.metric(f).notify(0 - metrics.metric(f).raw_data())
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

