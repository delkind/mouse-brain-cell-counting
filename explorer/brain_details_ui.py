import os
import pickle

from build_cell_grid import build_cell_grid

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 140))

import PIL.Image as Image
import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display, Markdown, Javascript, HTML
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import matplotlib.pyplot as plt

from annotate_cell_data import create_section_image, get_brain_bbox_and_image, create_section_contours, get_contours, \
    create_section_contours_pdf
from explorer.explorer_utils import is_file_up_to_date, init_model, predict_crop
from explorer.ui import ExperimentsSelector
from localize_brain import detect_brain


def create_button(base_time, url, display_name, builder, download, output):
    def on_click(b):
        if not is_file_up_to_date(url, base_time):
            b.disabled = True
            b.description = f"Building {display_name}..."
            builder()
            b.description = f"{'Open' if not download else 'Download'} {display_name}"
            b.disabled = False

        with output:
            if download:
                display(HTML(f'<a href="/files/{url}" download id="download" hidden></a>'))
                display(Javascript('''
                    document.getElementById('download').click();
                    document.getElementById('download').remove();
                '''))
            else:
                display(Javascript(f"window.open('/files/{url}');"))
                display(HTML("Done."))

    if os.path.isfile(url):
        button = widgets.Button(description=f"{'Open' if not download else 'Download'} {display_name}")
    else:
        button = widgets.Button(description=f"Build {display_name}")

    button.on_click(on_click)
    button.layout.width = 'auto'
    return button


class SectionHistogramPlotter(object):
    class HeatmapAndPatchButtons(widgets.HBox):
        def __init__(self, experiment_id, base_time, input_dir, seg_data_dir, output):
            self.seg_data_dir = seg_data_dir
            self.experiment_id = experiment_id
            self.base_time = base_time
            self.input_dir = input_dir
            patches_url = f'{input_dir}/{self.experiment_id}/patches-{self.experiment_id}.pdf'
            self.patches = create_button(base_time, patches_url, "patches", self.build_patches, False, output)
            heatmaps_url = f'{input_dir}/{self.experiment_id}/heatmaps-{self.experiment_id}.pdf'
            self.heatmaps = create_button(base_time, heatmaps_url, "heatmaps", self.build_heatmaps, False, output)
            cell_patches_url = f'{input_dir}/{self.experiment_id}/cell-patches-{self.experiment_id}.pdf'
            self.cell_patches = create_button(base_time, cell_patches_url, "cell patches",
                                              lambda: self.build_cell_patches(cell_patches_url), False, output)
            cell_patches_png_url = f'{input_dir}/{self.experiment_id}/cell-patches-{self.experiment_id}.png'
            self.cell_patches_png = create_button(base_time, cell_patches_png_url, "cell patches",
                                                  lambda: self.build_cell_patches(cell_patches_png_url), False, output)
            super().__init__((self.cell_patches, self.cell_patches_png))

        def build_patches(self):
            pass

        def build_heatmaps(self):
            pass

        def build_cell_patches(self, url):
            build_cell_grid(self.experiment_id, self.input_dir, self.seg_data_dir, url)

    class AnnotationsButtonBar(widgets.HBox):
        def __init__(self, experiment_id, full_data, section, base_time, input_dir, output, seg_data_dir):
            self.seg_data = np.load(f'{seg_data_dir}/{experiment_id}/{experiment_id}-sections.npz')['arr_0']
            self.output = output
            self.section = section
            self.full_data = full_data
            self.experiment_id = experiment_id
            self.base_time = base_time
            self.directory = f'{input_dir}/{self.experiment_id}'
            self.bboxes = pickle.load(open(f'{self.directory}/bboxes.pickle', 'rb'))
            self.raw_image_url = f'{input_dir}/{self.experiment_id}/raw-image-{self.experiment_id}-{section}.jpg'
            self.raw_image = create_button(self.base_time, self.raw_image_url, "raw image", self.build_raw_image, True, output)
            self.contours_url = f'{input_dir}/{self.experiment_id}/cell-contours-{self.experiment_id}-{section}.png'
            self.contours = create_button(self.base_time, self.contours_url, "cell contours", self.build_contours, True, output)
            self.contours_pdf_url = f'{input_dir}/{self.experiment_id}/cell-contours-{self.experiment_id}-{section}.pdf'
            self.contours_pdf = create_button(self.base_time, self.contours_pdf_url, "cell contours", self.build_contours_pdf, True, output)
            self.annotated_url = f'{input_dir}/{self.experiment_id}/annotated-{self.experiment_id}-{section}.jpg'
            self.annotated = create_button(self.base_time, self.annotated_url, "annotated image", self.build_annotated, False, output)
            self.y_input = widgets.IntText(value=0, description='Y')
            self.x_input = widgets.IntText(value=0, description='X')
            self.predict_button = widgets.Button(description='Predict crop', )
            self.predict_button.on_click(self.do_predict)
            self.cell_model = None
            super().__init__((self.annotated, self.contours, self.contours_pdf, self.raw_image, self.x_input, self.y_input,
                              self.predict_button,))

        def build_raw_image(self):
            thumb, brain_bbox = get_brain_bbox_and_image(self.bboxes, self.directory, self.experiment_id,
                                                         self.section, True, scale=2)
            x, y, w, h = brain_bbox.scale(0.5)
            cv2.imwrite(self.raw_image_url, thumb[y: y + h, x: x + w])

        def build_contours(self):
            create_section_contours(self.section, self.experiment_id, self.directory,
                                    self.bboxes, self.contours_url, self.seg_data)

        def build_contours_pdf(self):
            create_section_contours_pdf(self.section, self.experiment_id, self.directory,
                                        self.bboxes, self.contours_pdf_url, self.seg_data)

        def build_annotated(self):
            thumb = create_section_image(self.section, self.experiment_id, self.directory, self.full_data,
                                         self.bboxes, self.seg_data)
            _, r, _ = detect_brain(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(self.annotated_url, thumb[r.y: r.y + r.h, r.x: r.x + r.w])

        def do_predict(self, b):
            if self.cell_model is None:
                self.cell_model = init_model('output/new_cells/model_0324999.pth', 'cpu', 0.5)

            # url = f'{self.directory}/raw-unscaled-{self.experiment_id}-{self.section}.jpg'
            # if os.path.isfile(url):
            #     thumb = cv2.imread(url, cv2.IMREAD_GRAYSCALE)
            #     x = 0
            #     y = 0
            # else:
            #     cv2.imwrite(url, thumb)
            thumb, brain_bbox = get_brain_bbox_and_image(self.bboxes, self.directory, self.experiment_id,
                                                         self.section, True, scale=1)
            x, y, w, h = brain_bbox

            x += self.x_input.value * 2
            y += self.y_input.value * 2

            crop = thumb[y: y + 624, x: x + 624]

            result = np.zeros((624, 624, 3), dtype=thumb.dtype)

            for i in range(2):
                for j in range(2):
                    result[i * 312: i * 312 + 312, j * 312: j * 312 + 312, :] = cv2.cvtColor(
                        predict_crop(crop[i * 312: i * 312 + 312, j * 312: j * 312 + 312], self.cell_model),
                        cv2.COLOR_BGR2RGB)

            cell_contours = get_contours(self.bboxes, self.directory, self.experiment_id, self.section)

            src = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

            for color, contours in cell_contours:
                cv2.polylines(src, contours, color=(0, 255, 0), thickness=1, isClosed=True)

            with self.output:
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].imshow(result)
                ax[0].set_title("Current prediction")
                ax[1].imshow(src[y: y + 624, x: x + 624])
                ax[1].set_title("Original prediction")
                plt.show()

    def __init__(self, experiment_id, section, data, base_time, input_dir, seg_data_dir, structure_tree):
        self.structure_tree = structure_tree
        self.experiment_id = experiment_id
        self.data = data
        # self.hist_button = widgets.Button(description=f"Show detailed histograms")
        # self.hist_button.layout.width = 'auto'
        # self.hist_button.on_click(self.clicked)
        self.output = widgets.Output()
        self.section = section
        display(Markdown('---'))
        if self.section != 'totals':
            self.section = int(section)
            display(widgets.Label(f"Experiment {self.experiment_id}, section {self.section}"))
            self.annotated_button_bar = self.AnnotationsButtonBar(self.experiment_id, self.data, self.section,
                                                                  base_time, input_dir, self.output, seg_data_dir)
            display(self.output)
            display(widgets.HBox((self.annotated_button_bar,)))
            if os.path.isfile(f'{input_dir}/{experiment_id}/thumbnail-{self.experiment_id}-{section}.jpg'):
                thumb = Image.open(f'{input_dir}/{experiment_id}/thumbnail-{self.experiment_id}-{section}.jpg')
                _, rect, _ = detect_brain(np.array(thumb.convert('LA'))[:, :, 0])
                thumb = thumb.crop((*(rect.corners()[0]), *(rect.corners()[1]),))
            else:
                thumb = None
        else:
            display(widgets.Label(f"Experiment {self.experiment_id}, totals"))
            display(self.output)
            display(widgets.HBox((self.HeatmapAndPatchButtons(experiment_id, base_time, input_dir, seg_data_dir, self.output),)))
            thumb = None

        with self.output:
            # plot_section_violin_diagram(self.data, structure_tree, thumb)
            if thumb is not None:
                plt.imshow(thumb)
            plt.axis('off')
            plt.show()
    #
    # def clicked(self, b):
    #     self.hist_button.disabled = True
    #     self.hist_button.description = 'Building detailed histograms...'
    #     # self.output.clear_output()
    #     with self.output:
    #         plot_section_histograms(self.data, self.structure_tree)
    #     self.hist_button.layout.display = 'none'


class BrainDetailsSelector(widgets.VBox):
    def __init__(self, input_dir, seg_data_dir):
        self.seg_data_dir = seg_data_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'./.mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.input_dir = input_dir
        self.experiments_selector = ExperimentsSelector()
        self.histograms_output = widgets.Output()
        self.section_combo = widgets.Dropdown(options=[],
                                              value=None,
                                              description='Choose section:')

        self.refresh_button = widgets.Button(description='Refresh experiment list')
        self.refresh_button.layout.width = 'auto'
        self.refresh_button.on_click(lambda b: self.refresh)

        self.display_button = widgets.Button(description='Go!')
        self.display_button.on_click(self.display_experiment)

        self.clear_button = widgets.Button(description='Clear plots')
        self.clear_button.on_click(lambda b: self.histograms_output.clear_output())

        self.experiments_selector.on_selection_change(lambda col, change: self.refresh())
        super().__init__((
            widgets.HBox((self.experiments_selector, self.refresh_button)),
            widgets.HBox((self.section_combo, self.display_button, self.clear_button)),
            self.histograms_output
        ))

        self.refresh()
        self.on_select_experiment([])

    def create_grid(self):
        experiment_ids = list(set(self.experiments_selector.get_selection()))
        experiment_id = experiment_ids[0]
        build_cell_grid.build_cell_grid(experiment_id,
                                        self.input_dir,
                                        self.seg_data_dir,
                                        f'{self.input_dir}/{experiment_id}/cell-grid.pdf')

    def refresh(self):
        self.experiments_selector.set_available_brains(set([int(e) for e in os.listdir(self.input_dir) if e.isdigit()]))
        experiment_ids = list(set(self.experiments_selector.get_selection()))
        self.on_select_experiment(experiment_ids)

    def on_select_experiment(self, experiment_ids):
        if len(experiment_ids) != 1:
            self.section_combo.options = []
            self.section_combo.value = None
            self.section_combo.disabled = True
        else:
            experiment_id = experiment_ids[0]
            directory = f"{self.input_dir}/{experiment_id}"
            with open(f'{directory}/bboxes.pickle', "rb") as f:
                bboxes = pickle.load(f)
            bboxes = {k: v for k, v in bboxes.items() if v}
            self.section_combo.options = ['all'] + list(str(i) for i in bboxes.keys())
            self.section_combo.value = self.section_combo.options[0]
            self.section_combo.disabled = False

    def display_experiment(self, b):
        experiment_id = list(self.experiments_selector.get_selection())[0]
        section = self.section_combo.value
        if not experiment_id or not section:
            return
        directory = f"{self.input_dir}/{experiment_id}"
        full_data = pd.read_parquet(f"{directory}/celldata-{experiment_id}.parquet")
        full_data_mtime = os.path.getmtime(f"{directory}/celldata-{experiment_id}.parquet")

        with self.histograms_output:
            if section == 'all':
                SectionHistogramPlotter(experiment_id, 'totals', full_data, full_data_mtime, self.input_dir,
                                        self.seg_data_dir,
                                        self.mcc.get_structure_tree())
                for section in sorted(full_data.section.unique().tolist()):
                    section_data = full_data[full_data.section == int(section)]
                    SectionHistogramPlotter(experiment_id, section, section_data, full_data_mtime, self.input_dir,
                                            self.seg_data_dir,
                                            self.mcc.get_structure_tree())
            else:
                if section != 'totals':
                    full_data = full_data[full_data.section == int(section)]
                SectionHistogramPlotter(experiment_id, section, full_data, full_data_mtime, self.input_dir,
                                        self.seg_data_dir,
                                        self.mcc.get_structure_tree())

# plot = SectionHistogramPlotter('brain', full_data)
