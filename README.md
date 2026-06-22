# IU Programming with Python Assignment

Student: Vasant Pardeshi

## Overview

This project implements the Ideal Function Selection task using Python, Object-Oriented Programming, SQLAlchemy, SQLite, Pandas, NumPy, and Matplotlib.

The application:

- Loads and validates training, ideal, and test datasets.
- Stores data in a SQLite database.
- Selects the best ideal function for each training function using SSE.
- Maps test data points to the selected ideal functions.
- Stores mapping results in the database.
- Generates visualizations and analysis plots.
- Includes automated unit tests.

## Installation

bash pip install -r requirements.txt 

## Run Project

bash python main.py 

## Run Tests

bash python test_selector.py python test_mapper.py 

## Generated Output

- results.db
- figure1_selected_ideal_functions.png
- figure2_test_data_mapping.png
- figure3_mapped_vs_unmapped.png
- figure4_deviation_analysis.png

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- SQLAlchemy
- SQLite
- unittest