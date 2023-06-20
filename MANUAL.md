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

- **Parameter:** Select the parameter (count, density, etc.) to visualize from the dropdown menu.

- **Sex:** Select one or more sexes to filter the data. If none is selected, data for all sexs will be included.

- **Strain:** Select one or more strains to filter the data. If none is selected, data for all strains will be included.

- **Transgenic Line:** Select one or more transgenic lines to filter the data. If none is selected, data for all transgenic lines will be included.

- **ID:** Select one or more IDs to filter the data. If none is selected, data for all IDs will be included.

#### Region

Under "Region", you can select the brain region(s) to include in your data. Note that you have to select at least one region.

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

Below the histogram, you can view the medians for each of your selected groups and the results of statistical tests comparing the groups.

## Conclusion

The Brain Explorer app provides a powerful and flexible tool for visualizing and exploring mouse brain data. By making use of the various selection and filtering options, you can focus on the data that's most relevant to your research.
