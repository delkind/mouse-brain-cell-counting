import logging
import os
import shutil
import sys
import time
from abc import abstractmethod, ABC


class DirWatcher(ABC):
    def __init__(self, input_dir, intermediate_dir, results_dir, name):
        self.__results_dir__ = results_dir
        self.__intermediate_dir__ = intermediate_dir
        self.__input_dir__ = input_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel('INFO')
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%a %Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        for d in [intermediate_dir, results_dir]:
            os.makedirs(d, exist_ok=True)
        self.initial_items = sorted([item for item in os.listdir(self.__input_dir__)
                                     if os.path.isdir(os.path.join(self.__input_dir__, item))])

    def extract_item(self):
        while True:
            items = [item for item in os.listdir(self.__input_dir__)
                     if os.path.isdir(os.path.join(self.__input_dir__, item))]
            items = [i for i in self.initial_items if i in items] + [i for i in items if i not in self.initial_items]

            if not items:
                return None, None

            try:
                item = items[0]
                directory = os.path.join(self.__intermediate_dir__, item)
                os.replace(os.path.join(self.__input_dir__, item), directory)
                return item, directory
            except OSError as e:
                continue

    def handle_item(self, item, directory):
        try:
            retval = self.process_item(item, directory)
            if retval is not None and not retval:
                shutil.rmtree(os.path.join(self.__intermediate_dir__, item), ignore_errors=True)
            else:
                os.replace(os.path.join(self.__intermediate_dir__, item), os.path.join(self.__results_dir__, item))
        except Exception as e:
            if self.on_process_error(item, e):
                self.logger.error("Irrecoverable error occurred. Exitting...")
                raise e
            else:
                self.logger.info("Recovered from the error. Continuing...")

    def run_until_empty(self):
        while True:
            item, directory = self.extract_item()
            if item is None:
                break
            self.handle_item(item, directory)

    def run_until_count(self, results_count):
        while True:
            results = [result for result in os.listdir(self.__results_dir__)
                       if os.path.isdir(os.path.join(self.__results_dir__, result))]
            if len(results) >= results_count:
                break

            item, directory = self.extract_item()
            if item is not None:
                self.handle_item(item, directory)
            else:
                time.sleep(1)

    def run_until_stopped(self):
        while True:
            item, directory = self.extract_item()
            if item is not None:
                self.handle_item(item, directory)
            else:
                time.sleep(1)

    def on_process_error(self, item, exception):
        os.replace(os.path.join(self.__intermediate_dir__, item), os.path.join(self.__input_dir__, item))
        return True

    @abstractmethod
    def process_item(self, item, directory):
        pass
