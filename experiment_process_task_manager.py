import argparse
from abc import ABC

from task_manager import TaskManager


class ExperimentProcessTaskManager(TaskManager, ABC):
    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument('--input-dir', '-i', action='store', required=True, help='Input directory')
        parser.add_argument('--process_dir', '-d', action='store', required=True, help='Processing directory')
        parser.add_argument('--output_dir', '-o', action='store', required=True,
                            help='Results output directory')
        parser.add_argument('--connectivity_dir', '-c', action='store', required=True,
                            help='Connectivity cache directory')
        parser.add_argument('--structure_map_dir', '-m', action='store', required=True,
                            help='Brain structure map directory')
        parser.add_argument('--structs', '-s', action='store', required=True,
                            help='List of structures to process')