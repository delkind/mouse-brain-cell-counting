import argparse
import ast
import bz2
import functools
import math
import operator as op
import os
import pickle
import socket
import time
import urllib.error
from itertools import groupby

import cv2
import numpy as np
import pandas
import scipy.ndimage as ndi
import simplejson
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from annotate_cell_data import ExperimentDataAnnotator
from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from localize_brain import detect_brain
from util import infinite_dict

SLAB_SIZE = 6


def get_center_x(section_seg_data):
    _, bbox, _ = detect_brain((section_seg_data != 0).astype(np.uint8) * 255)
    center_x = bbox.x + bbox.w // 2
    return center_x


class ExperimentCellsProcessor(object):
    def __init__(self, mcc, experiment_id, directory, brain_seg_data_dir, parent_struct_id,
                 experiment_fields_to_save, details, logger, default_struct_id=997):
        self.experiment_fields_to_save = experiment_fields_to_save
        self.default_struct_id = default_struct_id
        self.parent_struct_id = parent_struct_id
        self.brain_seg_data_dir = brain_seg_data_dir
        self.directory = directory
        self.mcc = mcc
        self.id = experiment_id
        mapi = MouseConnectivityApi()
        while True:
            try:
                self.details = {**details, **(mapi.get_experiment_detail(self.id)[0])}
                break
            except simplejson.errors.JSONDecodeError or urllib.error.URLError or urllib.error.URLError:
                time.sleep(1.0)
        self.logger = logger
        self.subimages = {i['section_number']: i for i in self.details['sub_images']}
        self.seg_data = np.load(f'{self.brain_seg_data_dir}/{self.id}/{self.id}-sections.npz')['arr_0']
        self.structure_tree = self.mcc.get_structure_tree()
        self.structure_ids = self.get_requested_structure_children()
        with open(f'{self.directory}/bboxes.pickle', "rb") as f:
            bboxes = pickle.load(f)
        self.bboxes = {k: v for k, v in bboxes.items() if v}

    def get_requested_structure_children(self):
        return self.get_structure_children(self.parent_struct_id)

    def get_structure_children(self, structure_id):
        structure_ids = self.structure_tree.descendant_ids(structure_id)
        structure_ids = list(set(functools.reduce(op.add, structure_ids)))
        return structure_ids

    def get_structure_mask(self, section):
        section_seg_data = self.seg_data[:, :, section]
        mask = np.isin(section_seg_data, self.structure_ids)
        mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int8)).astype(np.int8)
        return mask

    def calculate_coverages(self, csv):
        csv['coverage'] = 0
        for section in sorted(np.unique(csv.section)):
            csv_section = csv[csv.section == section]
            self.logger.debug(f"Creating heatmap experiment {self.id} section {section}")
            image = np.zeros((self.seg_data.shape[0] * 64, self.seg_data.shape[1] * 64), dtype=np.int16)
            for bbox in self.bboxes.get(section, []):
                x, y, w, h = bbox.scale(64)
                cellmask = cv2.imread(f'{self.directory}/cellmask-{self.id}-{section}-{x}_{y}_{w}_{h}.png',
                                      cv2.IMREAD_GRAYSCALE)
                cnts, _ = cv2.findContours(cellmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = [c for c in cnts if c.shape[0] > 2]
                cnts = [c for c in cnts if math.pi < Polygon(c.squeeze()).area *
                        (self.subimages[section]['resolution'] ** 2) < math.pi * 36]
                new_img = np.zeros_like(cellmask)
                new_img = cv2.fillPoly(new_img, cnts, color=1)
                image[y: y + cellmask.shape[0], x: x + cellmask.shape[1]] = new_img

            centroids_y = csv_section.centroid_y.to_numpy().astype(int)
            centroids_x = csv_section.centroid_x.to_numpy().astype(int)
            coverages = np.array([image[centroids_y[i] - 32: centroids_y[i] + 32,
                                  centroids_x[i] - 32: centroids_x[i] + 32].sum() for i in range(len(centroids_y))])
            csv.at[csv.section == section, 'coverage'] = coverages / 4096

    def build_dense_masks(self, celldata_struct, dense_masks, relevant_sections):
        scale = 64 // (dense_masks.shape[0] // self.seg_data.shape[0])
        for section in relevant_sections:
            self.logger.debug(f"Analyzing section {section}")
            celldata_section = celldata_struct[celldata_struct.section == section]
            centroids_x = (celldata_section.centroid_x.to_numpy() // scale).astype(int)
            centroids_y = (celldata_section.centroid_y.to_numpy() // scale).astype(int)
            coverages = celldata_section.coverage.to_numpy()

            if coverages.shape[0] > 2:
                model = KMeans(n_clusters=2)
                yhat = model.fit_predict(coverages.reshape(-1, 1))
                clusters = np.unique(yhat).tolist()
                if len(clusters) == 1:
                    yhat = np.zeros_like(coverages)
                    dense = 1
                else:
                    dense = np.argmax([coverages[yhat == 0].mean(), coverages[yhat == 1].mean()])
                if scale <= 8:
                    dense_masks[:, :, section - min(relevant_sections)] = self.produce_precise_dense_mask(
                        celldata_section, centroids_x, centroids_y, dense, dense_masks, scale, yhat)
                else:
                    dense_masks[:, :, section - min(relevant_sections)] = \
                        self.produce_coarse_dense_mask(centroids_x, centroids_y, dense, yhat)

    def produce_precise_dense_mask(self, celldata_section, centroids_x, centroids_y, dense, dense_masks, scale, yhat):
        dense_mask = np.zeros_like(dense_masks[:, :, 0], dtype=np.uint8)
        radius = (math.sqrt(celldata_section.area.max() / math.pi) + 0.5) * 8 / scale
        centroids_y, centroids_x = centroids_y[yhat == dense], centroids_x[yhat == dense]
        radii = np.maximum((celldata_section.coverage.to_numpy() * radius + 0.5), 1).astype(int)
        for i in range(centroids_x.shape[0]):
            cv2.circle(dense_mask, (centroids_x[i], centroids_y[i]), radii[i], 1, cv2.FILLED)
        dense_mask = ndi.binary_dilation(dense_mask, ndi.generate_binary_structure(2, 64 // scale), iterations=6)
        return self.remove_small_components(dense_mask)

    def produce_coarse_dense_mask(self, centroids_x, centroids_y, dense, yhat):
        dense_mask = np.zeros_like(self.seg_data[:, :, 0])
        dense_mask[centroids_y[yhat == dense], centroids_x[yhat == dense]] = 1
        dense_mask = ndi.binary_closing(dense_mask, ndi.generate_binary_structure(2, 1), iterations=4)
        return self.remove_small_components(dense_mask)

    def remove_small_components(self, dense_mask):
        dense_mask, comps = ndi.measurements.label(dense_mask)
        if comps > 0:
            dm_nonzero = dense_mask[dense_mask != 0]
            sums = np.array([(dm_nonzero == i + 1).sum() for i in range(comps)])
            comps = np.argwhere((sums > 10) & ((sums.max() / sums) < 5)).flatten()
            dense_mask = np.isin(dense_mask, comps + 1)
        else:
            dense_mask = np.zeros_like(dense_mask)
        return dense_mask

    def plot_coverage_masks(self, data_frame, dense_masks, relevant_sections):
        heatmaps = dict()
        for section in relevant_sections:
            heatmaps[section] = np.zeros_like(self.seg_data[:, :, section], dtype=float)
            heatmaps[section][data_frame[data_frame.section == section].centroid_y.to_numpy().astype(int) // 64,
                              data_frame[data_frame.section == section].centroid_x.to_numpy().astype(int) // 64] = \
                data_frame[data_frame.section == section].coverage
        import matplotlib.pyplot as plt
        for section in range(dense_masks.shape[2]):
            fig, axs = plt.subplots(1, 2)
            fig.suptitle(f"Section {section + min(relevant_sections)}")
            axs[0].imshow(dense_masks[:, :, section], cmap='gray')
            axs[1].imshow(heatmaps.get(section + min(relevant_sections), np.zeros_like(dense_masks[:, :, section])),
                          cmap='hot')
            plt.show()

    def detect_dense_dg(self, data_frame):
        dg_structs = [10703, 10704, 632]
        scale = 4
        scaling_factor = 64 // scale
        celldata_indices = data_frame.index[data_frame.structure_id.isin(dg_structs)]
        celldata_struct = data_frame.iloc[celldata_indices]
        relevant_sections = sorted(np.unique(celldata_struct.section.to_numpy()).tolist())
        if not relevant_sections:
            return dict()

        dense_masks = np.zeros((self.seg_data.shape[0] * scaling_factor, self.seg_data.shape[1] * scaling_factor,
                                max(relevant_sections) - min(relevant_sections) + 1), dtype=np.uint8)

        self.build_dense_masks(celldata_struct, dense_masks, relevant_sections)
        # self.plot_coverage_masks(data_frame, dense_masks, relevant_sections)

        dense_masks = dense_masks * 632

        for section in sorted(relevant_sections):
            self.logger.debug(f"Processing DG dense areas for section {section}")
            sparse_seg_data_section = np.zeros_like(dense_masks[:, :, section - min(relevant_sections)])
            for r in dg_structs:
                x, y = np.where(self.seg_data[:, :, section] == r)
                locs = [[x * scaling_factor + i, y * scaling_factor + j] for i in
                        range(scaling_factor) for j in range(scaling_factor)]
                x, y = list(zip(*locs))
                x = np.concatenate(x)
                y = np.concatenate(y)
                sparse_seg_data_section[x, y] = r

            dense_locs = np.where(sparse_seg_data_section[:, :] == 632)
            if dense_locs[0].shape[0] > 0:
                sparse_locs = np.where(np.logical_and(sparse_seg_data_section != 632, sparse_seg_data_section != 0))
                x = np.stack(sparse_locs).swapaxes(0, 1)
                y = sparse_seg_data_section[sparse_locs]

                neigh = KNeighborsClassifier(n_neighbors=3)
                neigh.fit(x, y)

                x = np.stack(sparse_locs).swapaxes(0, 1)
                sparse_seg_data_section[sparse_locs] = neigh.predict(x)

            sparse_locs = dense_masks[:, :, section - min(relevant_sections)] == 0
            dense_masks[sparse_locs, section - min(relevant_sections)] = sparse_seg_data_section[sparse_locs]

        centroids_y = data_frame.centroid_y.to_numpy().astype(int) // scale
        centroids_x = data_frame.centroid_x.to_numpy().astype(int) // scale
        sections = data_frame.section.to_numpy().astype(int) - min(relevant_sections)

        cells_values = dense_masks[centroids_y[celldata_indices], centroids_x[celldata_indices],
                                   sections[celldata_indices]]
        nonzero_indices = celldata_indices[cells_values != 0]

        data_frame.iloc[nonzero_indices, data_frame.columns.get_loc('structure_id')] = cells_values[cells_values != 0]

        data_frame.at[data_frame.structure_id == 632, 'dense'] = True
        data_frame.at[data_frame.structure_id.isin([10703, 10704]), 'dense'] = False

        return {r: (min(relevant_sections), max(relevant_sections), dense_masks.shape, np.where(dense_masks == r))
                for r in dg_structs}

    def detect_dense_ca(self, csv):
        scale = 8
        scaling_factor = 64 // scale

        structures_including_dense = [f'Field CA{i}' for i in range(1, 4)]
        dense_struct_ids = [r['id'] for r in
                            self.structure_tree.get_structures_by_name([s + ', pyramidal layer'
                                                                        for s in structures_including_dense])]
        structures_including_dense = [r['id'] for r in
                                      self.structure_tree.get_structures_by_name(structures_including_dense)]
        celldata_structs = csv[csv.structure_id.isin(structures_including_dense)]
        relevant_sections = sorted(np.unique(celldata_structs.section.to_numpy()).tolist())
        if not relevant_sections:
            return dict()

        dense_masks = np.zeros((self.seg_data.shape[0] * scaling_factor, self.seg_data.shape[1] * scaling_factor,
                                max(relevant_sections) - min(relevant_sections) + 1,
                                len(structures_including_dense)), dtype=int)

        dense_masks_dict = dict()

        for ofs, structure in enumerate(structures_including_dense):
            celldata_struct = celldata_structs[celldata_structs.structure_id.isin([structure])]
            self.build_dense_masks(celldata_struct, dense_masks[:, :, :, ofs], relevant_sections)
            dense_masks_dict[dense_struct_ids[ofs]] = \
                (min(relevant_sections), max(relevant_sections), dense_masks[:, :, :, ofs].shape,
                 np.where(dense_masks[:, :, :, ofs] != 0))

        # slc = self.seg_data[:, :, min(relevant_sections): max(relevant_sections) + 1]
        # for i, s in enumerate(dense_struct_ids):
        #     locs_mask = (dense_masks[:, :, :, i] != 0)
        #     slc[locs_mask] = s

        dense_masks = dense_masks.sum(axis=3) != 0
        # self.plot_coverage_masks(csv, dense_masks, relevant_sections)

        centroids_y = celldata_structs.centroid_y.to_numpy().astype(int) // scale
        centroids_x = celldata_structs.centroid_x.to_numpy().astype(int) // scale
        sections = celldata_structs.section.to_numpy().astype(int)
        csv.at[csv.structure_id.isin(structures_including_dense), 'dense'] = \
            dense_masks[centroids_y, centroids_x, sections - min(relevant_sections)].astype(bool)

        for i, s in enumerate(dense_struct_ids):
            csv.at[(csv.structure_id == structures_including_dense[i]) & csv.dense, 'structure_id'] = s

        return dense_masks_dict

    def calculate_global_parameters(self, dense_masks, cells):
        relevant_sections = cells.section.unique()
        relevant_sections.sort()
        globs_per_section = infinite_dict()

        for section in relevant_sections:
            self.logger.debug(f"Calculating globals for section {section} of {self.id}...")
            section_seg_data = self.seg_data[:, :, section]
            for region, (start, end, shape, (mask_y, mask_x, mask_section)) in dense_masks.items():
                if start <= section <= end:
                    if shape[0] < section_seg_data.shape[0]:
                        ratio = section_seg_data.shape[0] // shape[0]
                        deltas_x = np.array(([i for i in range(ratio)] * ratio) * len(mask_x))
                        deltas_y = np.array([[i] * ratio for i in range(ratio)] * len(mask_x))
                        mask_y = np.kron(mask_y * ratio, np.ones((1, ratio * ratio))).flatten() + deltas_y
                        mask_x = np.kron(mask_x * ratio, np.ones((1, ratio * ratio))).flatten() + deltas_x
                    elif shape[0] > section_seg_data.shape[0]:
                        ratio = shape[0] // section_seg_data.shape[0]
                        section_seg_data = np.kron(section_seg_data,
                                                   np.ones((ratio, ratio), dtype=section_seg_data.dtype))

                    relevant_cells = mask_section == (section - start)
                    section_seg_data[mask_y[relevant_cells], mask_x[relevant_cells]] = region

            center_x = get_center_x(section_seg_data)
            scale_factor = (0.35 * 64) / (section_seg_data.shape[0] / self.seg_data.shape[0])
            relevant_regions = np.intersect1d(np.unique(section_seg_data),
                                              cells[cells.section == section].structure_id.unique())
            for region in relevant_regions:
                region_cells = np.where(section_seg_data == region)
                globs_per_section[region][section]['region_area'] = region_cells[0].shape[0] * (scale_factor ** 2)
                globs_per_section[region][section]['region_area_left'] = np.where(region_cells[1] < center_x)[0].shape[
                                                                             0] * (
                                                                                 scale_factor ** 2)
                globs_per_section[region][section]['region_area_right'] = \
                    np.where(region_cells[1] >= center_x)[0].shape[
                        0] * (
                            scale_factor ** 2)

        return globs_per_section

    def process(self):
        # if os.path.isfile(f'{self.directory}/celldata-{self.id}.parquet') \
        #         and os.path.isfile(f'{self.directory}/maps.pickle.bz2'):
        #     self.logger.info(f"Loading cell data for {self.id}...")
        #     cell_dataframe = pandas.read_parquet(f'{self.directory}/celldata-{self.id}.parquet')
        #     maps = pickle.load(bz2.open(f'{self.directory}/maps.pickle.bz2', 'rb'))
        #     dense_masks = maps['dense_masks']
        # else:
        self.logger.info(f"Extracting cell data for {self.id}...")
        sections = sorted([s for s in self.bboxes.keys() if self.bboxes[s]])
        section_data = list()
        cell_data = list()
        for section in sections:
            cells, sec = self.process_section(section)
            section_data.append(sec)
            cell_data += cells

        cell_dataframe = pandas.DataFrame(section_data)
        cell_dataframe.to_csv(f'{self.directory}/sectiondata-{self.id}.csv')
        cell_dataframe = pandas.DataFrame(cell_data)
        self.logger.info(f"Calculating coverages for {self.id}...")
        self.calculate_coverages(cell_dataframe)

        self.logger.info(f"Extracting dense layers for CA regions in {self.id}...")
        cell_dataframe['dense'] = False

        dense_masks = [self.detect_dense_ca(cell_dataframe)]

        self.logger.info(f"Extracting dense layers for DG regions in {self.id}...")
        dense_masks += [self.detect_dense_dg(cell_dataframe)]

        dense_masks = functools.reduce(lambda x, y: {**x, **y}, dense_masks)

        self.logger.info(f"Calculating global parameters for {self.id}...")
        globs = self.calculate_global_parameters(dense_masks, cell_dataframe)
        self.logger.info(f"Converting sparse CA structure IDs for {self.id}...")
        for struct in ['CA1', 'CA2', 'CA3']:
            sparse_id = self.mcc.get_structure_tree().get_structures_by_acronym([f'{struct}sr'])[0]['id']
            generic_id = self.mcc.get_structure_tree().get_structures_by_acronym([struct])[0]['id']
            cell_dataframe.at[(cell_dataframe.structure_id == generic_id) & (cell_dataframe.dense == False),
                              'structure_id'] = sparse_id
            globs[sparse_id] = globs[generic_id]
            del globs[generic_id]

        self.logger.info(f"Saving cell data for {self.id}...")
        cell_dataframe.to_parquet(f'{self.directory}/celldata-{self.id}.parquet')

        self.logger.info(f"Saving global data for {self.id}...")
        maps = {
            'dense_masks': dense_masks,
            'globs': globs
        }
        with bz2.open(f'{self.directory}/maps.pickle.bz2', 'wb') as f:
            pickle.dump(maps, file=f)

    def get_cell_mask(self, section, offset_x, offset_y, w, h, mask):
        cell_mask_file_name = os.path.join(self.directory,
                                           f'cellmask-{self.id}-{section}-{offset_x}_{offset_y}_{w}_{h}.png')
        cell_mask = cv2.imread(cell_mask_file_name, cv2.IMREAD_COLOR)[:, :, 1]
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        offset = np.array([offset_x, offset_y])
        cnts = [Polygon(cnt.squeeze() + offset) for cnt in cnts if cnt.shape[0] > 2]
        cnts = [poly for poly in cnts
                if mask[int(poly.centroid.y) // 64, int(poly.centroid.x) // 64]
                and math.pi < poly.area * (self.subimages[section]['resolution'] ** 2) < 36 * math.pi]
        return cnts

    def get_brain_metrics(self, section):
        thumbnail_file_name = os.path.join(self.directory, f'thumbnail-{self.id}-{section}.jpg')
        thumbnail = cv2.imread(thumbnail_file_name, cv2.IMREAD_GRAYSCALE)
        brain_mask, bbox, ctrs = detect_brain(thumbnail)
        brain_area = sum([Polygon(ctr.squeeze()).area for ctr in ctrs])
        return brain_area, bbox

    def process_section(self, section):
        brain_area, brain_bbox = self.get_brain_metrics(section)
        struct_mask = self.get_structure_mask(section)

        brain_bbox = brain_bbox.scale(64)

        self.logger.debug(f"Experiment: {self.id}, processing section {section}...")
        struct_area = struct_mask.sum()

        cells_data = list()

        for offset_x, offset_y, w, h in map(lambda b: b.scale(64), self.bboxes[section]):
            cells = self.get_cell_mask(section, offset_x, offset_y, w, h, struct_mask)
            box_cell_data = self.polygons_to_cell_data(cells, section, brain_bbox.x + brain_bbox.w // 2)
            cells_data += box_cell_data

        return cells_data, {
            'experiment_id': self.id,
            'section_id': section,
            'brain_area': brain_area * ((self.subimages[section]['resolution'] * 64) ** 2),
            'struct_area': struct_area * ((self.subimages[section]['resolution'] * 64) ** 2)
        }

    def polygons_to_cell_data(self, cells, section, center_x):
        struct_ids = [self.get_struct_id(cell, section) for cell in cells]
        box_cell_data = [{'section': section, 'structure_id': struct_id, 'centroid_x': int(cell.centroid.x),
                          'centroid_y': int(cell.centroid.y),
                          'side': 'left' if cell.centroid.x <= center_x else 'right',
                          'area': cell.area * (self.subimages[section]['resolution'] ** 2),
                          'diameter': 2 * math.sqrt(cell.area * (self.subimages[section]['resolution'] ** 2) / math.pi),
                          'perimeter': cell.length * self.subimages[section]['resolution'], } for cell, struct_id in
                         zip(cells, struct_ids) if struct_id in self.structure_ids]
        return box_cell_data

    def get_struct_id(self, cell, section):
        y = int(cell.centroid.y // 64)
        x = int(cell.centroid.x // 64)
        struct_id = self.seg_data[y, x, section]
        if struct_id == 0:
            neighborhood = self.seg_data[y - 1: y + 2, x - 1: x + 2, section]
            ids, counts = np.unique(neighborhood, return_counts=True)
            sorted_indices = np.argsort(-counts).tolist()
            for i in sorted_indices:
                if ids[i] != 0:
                    struct_id = ids[i]
                    break
            if struct_id == 0:
                struct_id = self.default_struct_id
        return struct_id


class CellProcessor(DirWatcher):
    experiment_fields_to_save = [
        'id',
        'gender',
        'injection_structures',
        'injection_volume',
        'injection_x',
        'injection_y',
        'injection_z',
        'product_id',
        'specimen_name',
        'strain',
        'structure_abbrev',
        'structure_id',
        'structure_name',
        'transgenic_line',
        'transgenic_line_id',
        'primary_injection_structure'
    ]

    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir, annotate,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'cell-processor-{_processor_number}')
        self.annotate = annotate
        self.structure_ids = structs
        self.brain_seg_data_dir = structure_map_dir
        self.source_dir = input_dir
        self.output_dir = output_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.experiments = {int(e['id']): e for e in self.mcc.get_experiments(dataframe=False)}

    def process_item(self, item, directory):
        item = int(item)
        experiment = ExperimentCellsProcessor(self.mcc,
                                              item,
                                              directory,
                                              self.brain_seg_data_dir,
                                              self.structure_ids,
                                              self.experiment_fields_to_save,
                                              self.experiments[item],
                                              self.logger)
        experiment.process()

        if self.annotate:
            annotator = ExperimentDataAnnotator(int(item), directory, self.logger)
            annotator.process()

    def on_process_error(self, item, exception):
        retval = super().on_process_error(item, exception)
        self.logger.error(f"Error occurred during processing", exc_info=True)
        if type(exception) in [urllib.error.HTTPError, OSError, ValueError,
                               urllib.error.URLError, socket.gaierror, MemoryError]:
            return False
        else:
            return retval


class ExperimentCellAnalyzerTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment cell data analyzer")

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument('--annotate', action='store_true', default=False,
                            help='Annotate after processing')

    def prepare_input(self, connectivity_dir, **kwargs):
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        mcc.get_structure_tree()

    def execute_task(self, structs, structure_map_dir, **kwargs):
        analyzer = CellProcessor(structs=ast.literal_eval(structs), structure_map_dir=structure_map_dir, **kwargs)
        experiments = os.listdir(structure_map_dir)
        analyzer.run_until_count(len(experiments))


if __name__ == '__main__':
    task_mgr = ExperimentCellAnalyzerTaskManager()
    task_mgr.run()
