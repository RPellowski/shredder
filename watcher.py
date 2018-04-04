import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
from appmetrics import metrics

class Event():
    def __init__(self, event_type, src_path, is_directory):
        self.event_type = event_type
        self.src_path = src_path
        self.is_directory = is_directory

#class Handler(FileSystemEventHandler):
def test_events():
    ext = {
          '.py'  : 'code',
          '.md'  : 'doc',
          '.txt' : 'doc'
          }
    def on_any_event(self, event):
        e = [v for k,v in ext.items() if event.src_path.endswith(k)]
        if event.is_directory :
            metrics.metric('other').notify(1)
        elif len(e) > 0:
            print(e[0])
            metrics.metric(e[0]).notify(1)
        else:
            metrics.metric('other').notify(1)
    for f in ['my.py', 'my.txt', '4914', '.']:
        for t in ['modified', 'deleted', 'moved', 'created']:
            event = Event(t, f, True if f is '.' else False)
            on_any_event(None, event)
    #logging.info({"event":[event.event_type, event.is_directory, event.src_path]})

if __name__ == "__xmain__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    #event_handler = LoggingEventHandler()
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    #counter
    observer.start()
    try:
        while True:
            time.sleep(1)
            for f in ['code', 'doc', 'other']:
            #logging.info({"event":[event.event_type, event.is_directory, event.src_path]})
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

#every minute, if bucket > 0, log all event counts
# bucket based on filetype: documentation/code/other
# any change to that bucket increments one of the counters: created/deleted/modified/moved

else:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    global counters
    counters = [metrics.new_counter(f) for f in ['code', 'doc', 'other']]
    #counters = [metrics.new_counter(" ".join((f, t)))
    #    for f in ['code', 'doc', 'other']
    #        for t in ['modified', 'deleted', 'moved', 'created']
    #    ]
    test_events()
    for f in ['code', 'doc', 'other']:
        print(f, metrics.get(f))
    for f in ['code', 'doc', 'other']:
        metrics.metric(f).notify(0 - metrics.metric(f).raw_data())
    for f in ['code', 'doc', 'other']:
        print(f, metrics.get(f))
