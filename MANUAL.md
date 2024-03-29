# Brain Explorer User Manual

This user manual will guide you on how to use the Brain Explorer Streamlit app.

## Overview

The Brain Explorer app is designed to visualize and explore mouse brain data. You can display several statistics and filter the data based on sex, strain, transgenic line, and brain ID. 

## Accessing the App
Use the button below to access the app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://delkind-mouse-brain-cell-counting-brain-explorer-3ek3v5.streamlit.app/)

## Using the App

### Sidebar

The sidebar on the left side of the app contains the controls for selecting and filtering the data to be visualized.

#### Basic Options

Under "Basic Options", you can select the following:

- **Parameter:** Select the parameter to visualize from the dropdown menu.
- **Hemishpere:** Select the hemisphere for which to visualize the parameter. Note not all the parameters are available for separate hemispheres
- **Aggregation (for microscopic parameters):** If you've selected a microscopic parameter (such as area, perimeter or diameter), select the aggregation method.
- **Sex:** Select one or more genders to filter the data. If none is selected, data for all genders will be included.
- **Strain:** Select one or more strains to filter the data. If none is selected, data for all strains will be included.
- **Transgenic Line:** Select one or more transgenic lines to filter the data. If none is selected, data for all transgenic lines will be included.

##### Parameters

You can select from the following parameters to visualize:

- **count3d:** The number of cells in a 3D model of the selected region.
- **density3d:** The density of cells in a 3D model of the selected region, calculated as the number of cells divided by the volume of the region.
- **volume:** The total volume of the selected region.
- **area:** The surface area of a single cell.
- **coverage:** The percentage of the area around a single cell that is covered by other cells.
- **diameter:** The diameter of a single cell.
- **perimeter:** The perimeter (or boundary) length of a single cell.

#### Region

Under "Region", you can select the brain region(s) to include in your data. You can select region from the tree and use search box to locate a region within the tree.

#### Selection Title

"Selection Title" is an automatically generated title that summarizes your current data selection. You can modify it if you wish.

#### Selection Management

Under "Selection Management", you can add your current selection to a group for comparison with other selections. Use the "Add to Selected" button to do this. 

You can view your selected groups in the "Selected" dropdown menu. From there, you can choose to remove one or more groups with the "Remove from Selected" button, or clear all your groups with the "Clear Selected" button.

#### Histogram Options

Under "Histogram Options", you can customize the histogram's appearance. Adjust the number of bins using the "Histogram bins" slider, and toggle the visibility of the median and the raw histogram (steps) with the respective checkboxes.

### Main Area

The main area of the app displays the histogram of your selected data and the statistical analysis of your selected groups.

The histogram visualizes the data of your current selection or selections. Each selection is represented by a differently colored histogram.

Below the histogram, you can view the medians for each of your selected groups and the results of statistical tests comparing the groups. The statistical tests performed include T-Test, RankSum, and KS-Test.

If no selection has been made, the app will display the original user manual.

## Conclusion

The Brain Explorer app provides a powerful and flexible tool for visualizing and exploring mouse brain data. By making use of the various selection and filtering options, you can focus on the data that's most relevant to your research.
