import argparse
import subprocess
import sys
from abc import ABC, abstractmethod


class TaskManager(ABC):
    def __init__(self, description):
        self.description = description
        self.process_number = -1
        self.args = None

    def run(self):
        parser = argparse.ArgumentParser(description=self.description)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--_processor_number', action='store', type=int, default=None, help=argparse.SUPPRESS)
        group.add_argument('--processors', '-p', action='store', type=int, help='Number of processors')
        self.add_args(parser)
        args = parser.parse_args()
        self.args = vars(args)

        print(self.args)
        args = {k: v for k, v in self.args.items() if k != 'processors'}
        if 'processors' in self.args and self.args['processors'] is not None and self.args['processors'] != 1:
            num_processors = self.args['processors']
            self.prepare_input(**args)
            self.spawn_processors(num_processors)
        else:
            self.process_number = self.args['_processor_number'] if self.args['_processor_number'] is not None else 0
            self.execute_task(**args)

    @abstractmethod
    def prepare_input(self, **kwargs):
        pass

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def execute_task(self, **kwargs):
        pass

    def spawn_processors(self, num_processors):
        args = [a for a in sys.argv if not a.startswith('-p') and not a.startswith('--_processor_number')]
        processes = [subprocess.Popen(["python"] + args + ["--_processor_number", f"{i}"]) for i in range(num_processors)]
        exit_codes = [p.wait() for p in processes]


if __name__ == '__main__':
    for arg in sys.argv:
        print(arg)
