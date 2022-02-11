import ast
import itertools
import math
import os
import pickle
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from localize_brain import detect_brain

mcc = MouseConnectivityCache(manifest_file='./.mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)

unique_colors = [
    (255, 0, 0)[::-1],
    (255, 255, 0)[::-1],
    (0, 234, 255)[::-1],
    (170, 0, 255)[::-1],
    (255, 127, 0)[::-1],
    (191, 255, 0)[::-1],
    (0, 149, 255)[::-1],
    (255, 0, 170)[::-1],
    (255, 212, 0)[::-1],
    (106, 255, 0)[::-1],
    (0, 64, 255)[::-1],
    (237, 185, 185)[::-1],
    (185, 215, 237)[::-1],
    (231, 233, 185)[::-1],
    (220, 185, 237)[::-1],
    (185, 237, 224)[::-1],
    (143, 35, 35)[::-1],
    (35, 98, 143)[::-1],
    (143, 106, 35)[::-1],
    (107, 35, 143)[::-1],
    (79, 143, 35)[::-1],
    (0, 0, 0)[::-1],
    (115, 115, 115)[::-1],
    (204, 204, 204)[::-1],
]


def get_brain_bbox_and_image(bboxes, directory, experiment_id, section, image_needed, scale=4):
    thumb = cv2.imread(f"{directory}/thumbnail-{experiment_id}-{section}.jpg", cv2.IMREAD_GRAYSCALE)
    _, brain_bbox, _ = detect_brain(thumb)
    thumb = cv2.resize(thumb, (0, 0), fx=64 // scale, fy=64 // scale)
    if image_needed:
        for bbox in bboxes[section]:
            x, y, w, h = bbox.scale(64)
            image = cv2.imread(f'{directory}/full-{experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg',
                               cv2.IMREAD_GRAYSCALE)
            x, y, w, h = bbox.scale(64 // scale)
            thumb[y: y + h, x: x + w] = cv2.resize(image, (0, 0), fx=1.0 / scale, fy=1.0 / scale)
    return thumb, brain_bbox.pad(5, 5).scale(64)


def get_contours(bboxes, directory, experiment_id, section, brain_seg_data):
    unique_numbers = list(set(np.unique(brain_seg_data[:, :, section]).tolist()) - {0})
    colors = {**{k: v for k, v in zip(unique_numbers, [r['rgb_triplet'] for r in mcc.get_structure_tree().
                                      get_structures_by_id(unique_numbers)])}, 0: [255, 255, 255]}
    cts_dict = defaultdict(list)
    for bbox in bboxes[section]:
        x, y, w, h = bbox.scale(64)
        mask = cv2.imread(f'{directory}/cellmask-{experiment_id}-{section}-{x}_{y}_{w}_{h}.png',
                          cv2.IMREAD_GRAYSCALE)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_mask = np.zeros_like(mask)
        new_mask = cv2.fillPoly(new_mask, cnts, color=255)
        cnts, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [cnt for cnt in cnts if cnt.shape[0] > 2]
        polygons = [Polygon((cnt.squeeze() + np.array([x, y])) // 64).centroid for cnt in cnts]
        polygons = [(int(p.x), int(p.y)) for p in polygons]
        for i, cnt in enumerate(cnts):
            cts_dict[brain_seg_data[polygons[i][1], polygons[i][0], section]].append(cnt.squeeze() + np.array([x, y]))

    return [(colors[k], v) for k, v in cts_dict.items()]


def create_section_image(section, experiment_id, directory, celldata, bboxes, brain_seg_data):
    thumb, brain_bbox = get_brain_bbox_and_image(bboxes, directory, experiment_id, section, True)
    cell_contours = get_contours(bboxes, directory, experiment_id, section, brain_seg_data)

    thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

    for color, contours in cell_contours:
        cv2.polylines(thumb, [c // 4 for c in contours], color=tuple(color)[::-1], thickness=1, isClosed=True)

    x, y, w, h = brain_bbox.scale(0.25)
    return thumb[y: y + h, x: x + w]


def create_section_contours(section, experiment_id, directory, bboxes, path, brain_seg_data):
    thumb, brain_bbox = get_brain_bbox_and_image(bboxes, directory, experiment_id, section, False, scale=2)
    cell_contours = get_contours(bboxes, directory, experiment_id, section, brain_seg_data)

    x, y, w, h = brain_bbox.scale(0.5)
    mask = np.zeros((h, w, 4), dtype=thumb.dtype)

    cv2.rectangle(mask, (0, 0), (w, h), color=(0, 0, 0, 255), thickness=2)

    # fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(brain_bbox.w // 100, brain_bbox.h // 100), dpi=25)
    # ax.set_xlim(0, brain_bbox.w)
    # ax.set_ylim(brain_bbox.h, 0)
    # ax.axis('off')
    # ax.add_patch(plt.Rectangle((0, 0), w, h, color=(0, 0, 0), fill=False))

    for color, contours in cell_contours:
        cv2.polylines(mask, [(c // 2) - np.array([x, y]) for c in contours], color=(tuple(color)[::-1]) + (255,),
                      thickness=2, isClosed=True)
        # for poly in contours:
        #     ax.add_patch(plt.Polygon(poly - np.array([brain_bbox.x, brain_bbox.y]),
        #                              closed=True, fill=False, color=np.array(color) / 255))

    # plt.savefig(path, dpi=25)
    cv2.imwrite(path, mask)


def create_section_contours_pdf(section, experiment_id, directory, bboxes, path, brain_seg_data):
    brain_bbox, patches = produce_patch_collection(bboxes, brain_seg_data, directory, experiment_id, section)
    fig, ax = plt.subplots(figsize=(brain_bbox.w // 100, brain_bbox.h // 100), dpi=25)
    plot_patch_collection(ax, brain_bbox, patches)
    plt.savefig(path, dpi=25)
    plt.close()


def produce_patch_collection(bboxes, brain_seg_data, directory, experiment_id, section):
    _, brain_bbox = get_brain_bbox_and_image(bboxes, directory, experiment_id, section, False, scale=2)
    cell_contours = get_contours(bboxes, directory, experiment_id, section, brain_seg_data)
    patches = [plt.Polygon(poly - np.array([brain_bbox.x, brain_bbox.y]),
                           closed=True, fill=False, color=np.array(color) / 255) for color, contours in cell_contours
               for poly in contours]
    return brain_bbox, PatchCollection(patches, match_original=True)


def plot_patch_collection(ax, brain_bbox, patches):
    ax.set_xlim(0, brain_bbox.w)
    ax.set_ylim(brain_bbox.h, 0)
    ax.axis('off')
    ax.add_collection(patches)


class ExperimentDataAnnotator(object):
    def __init__(self, experiment_id, directory, brain_seg_data_dir, logger):
        self.logger = logger
        self.directory = directory
        self.experiment_id = experiment_id
        self.brain_seg_data_dir = brain_seg_data_dir
        with open(f'{self.directory}/bboxes.pickle', "rb") as f:
            bboxes = pickle.load(f)
        self.bboxes = {k: v for k, v in bboxes.items() if v}
        self.celldata = pd.read_parquet(f'{self.directory}/celldata-{self.experiment_id}.parquet')
        self.tile_dim = int(math.ceil(math.sqrt(len(self.bboxes))))
        self.seg_data = np.load(f'{self.brain_seg_data_dir}/{self.experiment_id}/'
                                f'{self.experiment_id}-sections.npz')['arr_0']

    @staticmethod
    def generate_colormap(N):
        if N < 2:
            return np.array([0.9, 0, 0, 1])
        arr = np.arange(N) / N
        arr = arr.reshape(N, 1).T.reshape(-1)
        ret = matplotlib.cm.hsv(arr)
        n = ret[:, 3].size
        a = n // 2
        b = n - a
        for i in range(3):
            ret[0:n // 2, i] *= np.arange(0.2, 1, 0.8 / a)
        ret[n // 2:, 3] *= np.arange(1, 0.1, -0.9 / b)
        #     print(ret)
        return ret

    def transparent_cmap(cmap, N=255):
        mycmap = cmap
        mycmap._init()
        mycmap._lut[0, -1] = 0
        return mycmap

    def process(self):
        self.create_images()
        self.create_tiles(placer=self.place_heatmap, name='heatmaps', zoom=1, binsize=5)
        self.create_tiles(placer=self.place_patches, name='patches', zoom=4, gridsize=5)

    def create_images(self):
        self.logger.info(f"Experiment {self.experiment_id}: Creating annotated images...")
        for section in self.bboxes.keys():
            self.create_section_image(section)

    def create_section_image(self, section):
        thumb = create_section_image(section, self.experiment_id, self.directory, self.celldata, self.bboxes,
                                     self.seg_data)
        cv2.imwrite(f"{self.directory}/annotated-{self.experiment_id}-{section}.jpg", thumb)

    def create_tiles(self, placer, name, zoom, **kwargs):
        fig, axs = plt.subplots(self.tile_dim, self.tile_dim, constrained_layout=True)
        fig.suptitle(self.experiment_id, fontsize=8)
        for ax in axs.flatten().tolist():
            ax.set_axis_off()

        for num, section in enumerate(sorted(list(self.bboxes.keys()))):
            self.logger.debug(f"Experiment {self.experiment_id}: creating {name} for section {section} "
                              f"({num + 1}/{len(self.bboxes.keys())})")
            ax = axs[num // self.tile_dim, num % self.tile_dim]
            ax.set_title(section, fontsize=6)
            labels, p = self.process_section(ax, kwargs, placer, section, zoom)
            self.decorate_section(ax, fig, labels, p)

        self.logger.info(f"Experiment {self.experiment_id}: Saving {name}...")
        plt.savefig(f"{self.directory}/{name}-{self.experiment_id}.pdf", dpi=2400)
        plt.close()

    def process_section(self, ax, kwargs, placer, section, zoom):
        section_celldata = self.celldata[self.celldata.section == section]
        thumb = Image.open(f'{self.directory}/thumbnail-{self.experiment_id}-{section}.jpg').convert('LA')
        thumb = thumb.resize((thumb.size[0] * zoom, thumb.size[1] * zoom))
        labels, p = placer(ax=ax, thumb=thumb, section_celldata=section_celldata, zoom=zoom, **kwargs)
        return labels, p

    @staticmethod
    def decorate_section(ax, fig, labels, p):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        bar = fig.colorbar(p, cax=cax, ax=ax)
        bar.ax.tick_params(length=1, pad=0, labelsize=2)
        if labels:
            bar.set_ticks(sorted(list(labels.values())))
            bar.ax.set_yticklabels(labels)

    @staticmethod
    def place_patches(ax, thumb, gridsize, section_celldata, zoom, radius=3, colname='dense'):
        ax.imshow(thumb)
        structs = np.unique(section_celldata.structure_id.to_numpy() + section_celldata[colname].to_numpy()).tolist()
        struct_counts = {s: len(section_celldata[(section_celldata.structure_id + section_celldata[colname]) == s]) for
                         s in structs}
        cmap = ListedColormap(ExperimentDataAnnotator.generate_colormap(len(structs)))
        coords = np.stack((section_celldata.centroid_x.to_numpy() / (64 / zoom * gridsize),
                           section_celldata.centroid_y.to_numpy() / (64 / zoom * gridsize),
                           section_celldata.structure_id.to_numpy() + section_celldata[colname].to_numpy())).astype(
            int).tolist()
        coords = sorted(list({(x, y, s) for x, y, s in zip(*coords)}), key=lambda t: struct_counts[t[2]], reverse=True)
        patches = [Circle((x * gridsize, y * gridsize), radius) for x, y, _ in coords]
        colors = [c for _, _, c in coords]
        p = PatchCollection(patches, cmap=cmap, alpha=1.0)
        p.set_array((np.array([structs.index(s) for s in colors]) + 1))
        ax.add_collection(p)
        labels = {str(s): (i + 1) for i, s in enumerate(structs)}
        return labels, p

    @staticmethod
    def place_heatmap(ax, thumb, section_celldata, zoom, binsize):
        heatmap = np.zeros((thumb.size[1], thumb.size[0]), dtype=float)
        x = (section_celldata.centroid_x.to_numpy() // 64).astype(int)
        y = (section_celldata.centroid_y.to_numpy() // 64).astype(int)
        heatmap[y, x] = section_celldata.coverage.to_numpy()
        ax.imshow(thumb)
        ax.imshow(heatmap, cmap=ExperimentDataAnnotator.transparent_cmap(plt.get_cmap('hot')))
        return {}, matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(heatmap.min(), heatmap.max()),
            cmap=plt.get_cmap('hot'))


class CellProcessor(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'cell-processor-{_processor_number}')
        self.structure_ids = structs
        self.brain_seg_data_dir = structure_map_dir
        self.source_dir = input_dir
        self.output_dir = output_dir

    def process_item(self, item, directory):
        experiment = ExperimentDataAnnotator(item, directory, self.logger)
        experiment.process()


class ExperimentCellAnalyzerTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment cell data analyzer")

    def prepare_input(self, connectivity_dir, **kwargs):
        pass

    def execute_task(self, structs, structure_map_dir, **kwargs):
        analyzer = CellProcessor(structs=ast.literal_eval(structs), structure_map_dir=structure_map_dir, **kwargs)
        experiments = os.listdir(structure_map_dir)
        analyzer.run_until_count(len(experiments))


if __name__ == '__main__':
    task_mgr = ExperimentCellAnalyzerTaskManager()
    task_mgr.run()
