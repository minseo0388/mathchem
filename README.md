# mathchem
수리 및 계산화학 python 구현 라이브러리<br>
Mathematical and Computational Chemistry Lecture Implementation Library written in Python<br>
25H1 수리 및 계산화학 강의 구현(충남대학교 화학과)<br>
This project is used as a library for performing molecular orbital calculations and visualizing results in the field of chemistry.<br>
This will be uploaded to PyPI in the future.

Choi Minseo<br>
Sophomore (2025)<br>
Dept. of Chemistry<br>
College of Natural Sciences<br>
Chungnam National University<br>
Daejeon, Republic of Korea

## Functionality
This project implements various methods in mathematical and computational chemistry, including:
- Huckel approximation
- Variation Theory
- Particle in a Box model
- Stoichiometry (Molar mass calculation)
- Visualization of results
- Basic linear algebra operations

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
from cnumathchem.huckelapprox import Huckel
from cnumathchem.variationtheory import VariationTheory
from cnumathchem.stoichiometry import Stoichiometry

# Perform Huckel approximation
# ... (setup code)
results = Huckel(adj_matrix).solve()

# Calculate Molar Mass
mass = Stoichiometry.calculate_molar_mass("C6H12O6")
print(f"Molar Mass: {mass} g/mol")

# Visualize results
# Note: Visualization modules are currently separate scripts in the visualize/ directory
# but can be adapted for library use.
```
