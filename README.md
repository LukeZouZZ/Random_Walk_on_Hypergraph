# Hypergraph-MCMC: Molecular Screening via Random Walk on Hypergraphs

A Python library for efficient molecular screening using MCMC random walks on hypergraphs. Given a query molecule, the algorithm finds structurally similar molecules from large chemical databases without exhaustive pairwise comparison.

## Key Features

- **Hypergraph Modeling**: Molecules as nodes, molecular fingerprint features as hyperedges
- **Metropolis-Hastings MCMC**: Proper MCMC with acceptance ratio for theoretical guarantees
- **Simulated Annealing**: Temperature scheduling for better exploration-exploitation balance
- **Lazy Similarity Computation**: On-demand similarity calculation with LRU caching for scalability
- **TF-IDF Hyperedge Weighting**: Prioritize informative substructures over common ones

## Installation

```bash
# 1. Create conda environment
conda create -n screening python=3.10 -y

# 2. Activate environment
conda activate screening

# 3. Install RDKit (via conda-forge for stability)
conda install -c conda-forge rdkit -y

# 4. Install other dependencies
pip install numpy pandas scipy pyyaml matplotlib seaborn tqdm

# 5. Clone and install the package
git clone https://github.com/yourusername/hypergraph-molecule-screening.git
cd hypergraph-molecule-screening
pip install -e .
```

### Verify Installation

```bash
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "from src import HypergraphBuilder; print('Project OK')"
```

## Quick Start

### Basic Usage

```bash
python scripts/run_screening.py --data datasets/molecules.csv --query 42 --steps 200
```

### Full Usage with Output

```bash
python scripts/run_screening.py \
    --data datasets/molecules.csv \
    --query 42 \
    --steps 200 \
    --output results.json \
    --save-trajectory \
    --plot
```

This command will:
- Screen molecules similar to query molecule #42
- Run 200 MCMC steps
- Save results to `results.json`
- Save full sampling trajectory
- Generate visualization plot `mcmc_results.png`

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data`, `-d` | (required) | Path to molecule data file (CSV, SDF, or SMILES) |
| `--query`, `-q` | (required) | Index of query molecule |
| `--steps`, `-n` | 200 | Number of MCMC steps |
| `--algorithm` | mh | Sampling algorithm: `mh` (Metropolis-Hastings), `sa` (Simulated Annealing), `pt` (Parallel Tempering) |
| `--initial-temp` | 1.0 | Initial temperature |
| `--final-temp` | 0.01 | Final temperature |
| `--cooling` | exponential | Cooling schedule: `exponential`, `linear`, `logarithmic`, `adaptive` |
| `--weighting` | tfidf | Hyperedge weighting: `none`, `frequency`, `tfidf`, `bm25` |
| `--fp-type` | morgan | Fingerprint type: `morgan`, `rdkit`, `maccs` |
| `--fp-radius` | 2 | Fingerprint radius (for Morgan, radius=2 means ECFP4) |
| `--similarity` | tanimoto | Similarity metric: `tanimoto`, `dice`, `cosine` |
| `--top-k` | 10 | Number of top results to return |
| `--output`, `-o` | None | Output JSON file path |
| `--save-trajectory` | False | Save full MCMC trajectory |
| `--plot` | False | Generate visualization plots |
| `--plot-output` | mcmc_results.png | Output path for plots |
| `--seed` | None | Random seed for reproducibility |
| `--quiet` | False | Suppress progress output |
| `--smiles-col` | smiles | Name of SMILES column in CSV |

### Example Commands

```bash
# Use different sampling algorithm
python scripts/run_screening.py --data datasets/molecules.csv --query 100 --steps 300 --algorithm sa

# Use BM25 weighting with higher temperature
python scripts/run_screening.py --data datasets/molecules.csv --query 42 --steps 200 --weighting bm25 --initial-temp 2.0

# Reproducible run with fixed seed
python scripts/run_screening.py --data datasets/molecules.csv --query 42 --steps 200 --seed 42 --output results.json

# Quick run without progress bar
python scripts/run_screening.py --data datasets/molecules.csv --query 42 --steps 100 --quiet
```

## Python API

```python
from src.data import MoleculeLoader, FingerprintExtractor
from src.hypergraph import HypergraphBuilder, TFIDFWeighting
from src.sampler import HypergraphRandomWalk, MetropolisHastingsSampler
from src.similarity import LazySimilarityCache

# Load molecules
loader = MoleculeLoader("datasets/molecules.csv")
molecules = loader.load()

# Extract fingerprints and build hypergraph
fp_extractor = FingerprintExtractor(fp_type='morgan', radius=2)
fingerprints = fp_extractor.extract(molecules)

# Build weighted hypergraph
builder = HypergraphBuilder(min_hyperedge_size=2)
hypergraph = builder.build(fingerprints)

# Apply TF-IDF weighting
weighting = TFIDFWeighting()
hypergraph = weighting.apply(hypergraph)

# Initialize components
random_walk = HypergraphRandomWalk(hypergraph, seed=42)
similarity_cache = LazySimilarityCache(molecules, metric='tanimoto', max_size=10000)

# Create MCMC sampler
sampler = MetropolisHastingsSampler(
    random_walk=random_walk,
    similarity_cache=similarity_cache,
    initial_temp=1.0,
    final_temp=0.01,
    cooling_schedule="exponential",
    seed=42
)

# Run screening
query_idx = 42
result = sampler.sample(
    query_idx=query_idx,
    n_steps=200,
    n_results=10,
    save_trajectory=True
)

# Print results
print(result.summary())

# Access top similar molecules
for idx, similarity in result.top_molecules:
    print(f"Molecule {idx}: similarity = {similarity:.4f}")
```

## Algorithm Overview

### 1. Hypergraph Construction

The molecular dataset is modeled as a hypergraph H = (V, E):
- **Vertices V**: Each molecule is a vertex
- **Hyperedges E**: Each molecular fingerprint bit (substructure) defines a hyperedge connecting all molecules containing that substructure

The incidence matrix H ∈ {0,1}^(|V| × |E|) where H_ve = 1 if vertex v belongs to hyperedge e.

### 2. Hyperedge Weighting (TF-IDF)

To prioritize informative substructures, we apply TF-IDF weighting:

```
w_e = log(|V| / |e|)
```

where |e| is the number of molecules containing hyperedge e.

### 3. Random Walk Transition Probability

The transition probability from molecule u to molecule v:

```
P(u → v) = Σ_{e: u,v ∈ e} [w_e / (|e| - 1)] / Σ_{v' ≠ u} Σ_{e: u,v' ∈ e} [w_e / (|e| - 1)]
```

This formulation:
- Favors transitions through smaller (more specific) hyperedges
- Weights by hyperedge importance (TF-IDF)

### 4. Metropolis-Hastings MCMC

Given a query molecule q, we sample molecules with probability proportional to their similarity:

```
π(v) ∝ exp(sim(v, q) / T)
```

The MH acceptance ratio:

```
α(u → v) = min(1, [π(v) · P(v → u)] / [π(u) · P(u → v)])
```

### 5. Temperature Scheduling

Temperature T decreases over iterations (exploration → exploitation):

- **Exponential**: T_t = T_0 · γ^t
- **Linear**: T_t = T_0 - (T_0 - T_f) · t / t_max
- **Logarithmic**: T_t = T_0 / log(1 + t)
- **Adaptive**: Adjusts based on acceptance rate

## Project Structure

```
hypergraph-molecule-screening/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── default.yaml           # Default configuration
├── datasets/
│   └── molecules.csv          # Molecule dataset
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Molecule data loading
│   │   └── fingerprint.py     # Fingerprint extraction
│   ├── hypergraph/
│   │   ├── __init__.py
│   │   ├── builder.py         # Hypergraph construction
│   │   └── weighting.py       # Hyperedge weighting schemes
│   ├── sampler/
│   │   ├── __init__.py
│   │   ├── random_walk.py     # Hypergraph random walk
│   │   ├── mcmc.py            # MCMC samplers
│   │   └── scheduler.py       # Temperature scheduling
│   ├── similarity/
│   │   ├── __init__.py
│   │   └── lazy_similarity.py # Lazy similarity computation
│   └── utils/
│       ├── __init__.py
│       └── visualization.py   # Result visualization
├── scripts/
│   └── run_screening.py       # CLI entry point
└── examples/
    └── demo.py                # Usage examples
```

## Output Format

### JSON Output (`results.json`)

```json
{
  "query_idx": 42,
  "n_steps": 200,
  "algorithm": "mh",
  "best_molecule": 817,
  "best_similarity": 0.9297,
  "n_unique_visited": 122,
  "total_time": 0.08,
  "top_molecules": [
    [817, 0.9297],
    [2435, 0.9034],
    ...
  ],
  "ground_truth": [...],
  "cache_stats": {
    "hits": 156,
    "misses": 122,
    "hit_rate": 0.56
  },
  "parameters": {...},
  "trajectory": [...],
  "similarities": [...]
}
```

### Visualization Output (`mcmc_results.png`)

The plot includes:
- Sampling trajectory (similarity over steps)
- Temperature schedule
- Acceptance rate over time
- Top molecules bar chart
