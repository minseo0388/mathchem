# mathchem
수리 및 계산화학 python 구현 라이브러리<br>
Mathematical and Computational Chemistry Lecture Implementation Library written in Python<br>
25H1 수리 및 계산화학 강의 구현(충남대학교 화학과)

Choi Minseo
Sophomore (2025), Dept. of Chemistry<br>
College of Natural Sciences<br>
Chungnam National University<br>
Daejeon, Republic of Korea

## Functionality
This project implements various methods in mathematical and computational chemistry, including:
- Huckel approximation
- Variation Theory
- Particle in a Box model
- Visualization of results
- Basic linear algebra operations

## Library
This project is used as a library for performing molecular orbital calculations and visualizing results in the field of chemistry.
This will be uploaded to PyPI in the future.

```bash
pip install cnumathchem
```

## Installation
To install the project, clone the repository and install the required packages:
```bash
git clone https://github.com/minseo0388/mathchem.git
cd mathchem
pip install -r requirements.txt
```

And you can also install it directly from PyPI (RECOMMENDED):
```bash
pip install cnumathchem
```

And you should install dependencies for codes (if not already installed):
```bash
pip install numpy scipy matplotlib
```

## Usage
To use the library, import the necessary modules and call the functions as needed. 
Exact usage is provided in example/example.py.

For example:
```python
from cnumathchem import huckelapprox, variationtheory

# Perform Huckel approximation
results = huckelapprox.solve(...)

# Perform Variation Theory calculations
results = variationtheory.solve(...)

# Visualize results
from cnumathchem.visualize import proj_huckelapprox, proj_variationtheory
proj_huckelapprox.plot_molecular_orbitals(results)
proj_variationtheory.plot_variation_theory(results)
```
