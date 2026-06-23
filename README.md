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

pip install -r requirements.txt 

## Run Project

python main.py 

## Run Tests

python test_selector.py python test_mapper.py 

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

## Documentation

The final report is available in the `docs/` directory.

## Git Workflow

This project was developed using Git for version control and GitHub for repository hosting.

### Clone Repository

```bash
git clone https://github.com/Vas1261/CSEMDSPWP01_Vasant.git
cd CSEMDSPWP01_Vasant
```

### Initial Commit

```bash
git add .
git commit -m "Initial commit for IU Programming with Python assignment"
```

### Push Changes to GitHub

```bash
git push -u origin main
```

### Future Updates

Whenever changes are made to the project:

```bash
git add .
git commit -m "Describe your changes"
git push
```

### Repository Link

GitHub Repository:

https://github.com/Vas1261/CSEMDSPWP01_Vasant

### Version Control Benefits

- Tracks all source code changes.
- Maintains project history and commit records.
- Enables backup and collaboration through GitHub.
- Provides reproducibility and transparency for project development.