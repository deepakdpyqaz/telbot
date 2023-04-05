import logging
from multiprocessing import Process, Queue
import requests

class AdvLogger:
    def __init__(self, filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(process)s:%(message)s', chatID=[], api_key=""):
        self.filename = filename
        self.level = level
        self.format = format
        self.chatID = chatID
        self.api_key = api_key
        logging.basicConfig(filename=self.filename, level=self.level, format=self.format)
        self.log_mapper = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        self.log_worker = None
        self.log_queue = None
    def log_message(self, message, level="info"):
        if self.log_queue:
            self.log_queue.put((message, level))
        else:
            print("Logger not started")

    def _worker(self):
        self.log_queue = Queue()
        while True:
            message, level = self.log_queue.get()
            if level == "stop":
                break
            logging.log(self.log_mapper[level], message)
            if self.chatID:
                for chat in self.chatID:
                    requests.get("https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(self.api_key, chat, message))
    def start(self):
        p = Process(target=self._worker)
        self.log_worker = p
        p.start()
        print("Logger started")
    def stop(self):
        self.log_queue.put(("hii", "info"))
        self.log_queue.put(("", "stop"))
        