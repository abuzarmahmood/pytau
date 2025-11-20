# Pipeline Architecture

PyTau provides a modular pipeline architecture for reproducible, large-scale changepoint analysis. This page describes the pipeline components, batch processing capabilities, and best practices for managing large-scale analyses.

## Overview

PyTau's pipeline consists of four main components:

1. **Model Generation**: Creates and fits Bayesian changepoint models
2. **Data Preprocessing**: Prepares raw neural data for modeling
3. **I/O Management**: Handles data loading, model storage, and database management
4. **Batch Processing**: Enables large-scale model fitting across datasets

## Pipeline Components

### 1. Model Generation

**Purpose**: Creates and fits Bayesian changepoint models

**Key Module**: `pytau.changepoint_model`

**Components**:
- Model classes for different data types (Poisson, Gaussian, Categorical)
- Inference functions (ADVI, MCMC)
- Model selection utilities

**Example**:
```python
from pytau.changepoint_model import single_taste_poisson, advi_fit

# Create model
model = single_taste_poisson(data, states=3)

# Fit model
model, approx = advi_fit(model, fit=10000, samples=5000)
```

**Input**: Processed spike trains and model parameters

**Output**: Fitted model with posterior samples

### 2. Data Preprocessing

**Purpose**: Prepares raw neural data for modeling

**Key Module**: `pytau.changepoint_preprocess`

**Features**:
- Spike train binning
- Trial shuffling for null distributions
- Synthetic data generation
- Data validation and cleaning

**Example**:
```python
from pytau.changepoint_preprocess import preprocess_spike_data

# Preprocess data
processed_data = preprocess_spike_data(
    raw_spikes,
    bin_width=0.05,
    time_lims=(0, 2.0),
    shuffle=False
)
```

**Input**: Raw spike trains with preprocessing parameters

**Output**: Binned, processed spike count data

### 3. I/O Management

**Purpose**: Handles data loading, model storage, and database management

**Key Module**: `pytau.changepoint_io`

**Components**:
- `FitHandler`: Manages model fitting pipeline
- `DatabaseHandler`: Manages model database
- HDF5 data loading utilities
- Model serialization

**Example**:
```python
from pytau.changepoint_io import FitHandler

# Create handler
handler = FitHandler(
    data_dir='/path/to/data',
    save_dir='/path/to/models',
    animal_name='mouse1',
    session_date='2024-01-15',
    taste_name='sucrose',
    region_name='GC'
)

# Fit and save model
handler.fit_and_save(
    model_type='single_taste_poisson',
    states=3,
    fit_steps=10000
)
```

**Operations**:
- HDF5 data loading
- Metadata tracking
- Database integration
- Automatic model retrieval or fitting

### 4. Batch Processing

**Purpose**: Enables large-scale model fitting across datasets

**Key Module**: `pytau.utils.batch_utils`

**Features**:
- Parameter grid iteration
- Parallel processing support
- Progress tracking
- Error handling and logging

**Example**:
```python
from pytau.utils.batch_utils import batch_fit_models

# Define parameter grid
param_grid = {
    'states': [2, 3, 4, 5],
    'bin_width': [0.025, 0.05, 0.1],
    'model_type': ['single_taste_poisson', 'single_taste_poisson_varsig']
}

# Batch fit
results = batch_fit_models(
    data_dir='/path/to/data',
    param_grid=param_grid,
    n_jobs=4  # Parallel jobs
)
```

## Data Organization

PyTau uses a centralized database approach for model management:

### Centralized Storage

Models are stored in a central location with comprehensive metadata:

```
models/
├── database.db          # SQLite database with model metadata
├── animal1/
│   ├── session1/
│   │   ├── model_001.pkl
│   │   ├── model_002.pkl
│   │   └── ...
│   └── session2/
│       └── ...
└── animal2/
    └── ...
```

### Database Schema

The model database includes the following information:

**Model Identification**:
- Model save path
- Animal name
- Session date
- Taste/stimulus name
- Brain region name
- Experiment name
- Fit date

**Model Parameters**:
- Model type (e.g., `single_taste_poisson`)
- Data type (`actual`, `shuffled`, `simulated`)
- Number of states
- Fit steps (ADVI iterations)
- Time limits
- Bin width

**Model Contents**:
- Fitted model object
- Approximation (for ADVI)
- Lambda (emission rates)
- Tau (changepoint times)
- Data used to fit model
- Raw data (before preprocessing)
- Model and data details/parameters

### Database Operations

```python
from pytau.changepoint_io import DatabaseHandler

# Initialize database
db = DatabaseHandler(database_path='models/database.db')

# Query models
models = db.query(
    animal_name='mouse1',
    region_name='GC',
    states=3
)

# Add model to database
db.add_model(
    model_path='models/animal1/session1/model_001.pkl',
    metadata={
        'animal_name': 'mouse1',
        'session_date': '2024-01-15',
        'states': 3,
        # ... other metadata
    }
)

# Update model metadata
db.update_model(model_id=1, fit_date='2024-01-20')

# Delete model
db.delete_model(model_id=1)
```

## Batch Processing Workflows

### Single Parameter Sweep

Fit models across a range of state numbers:

```python
from pytau.changepoint_io import FitHandler

# Initialize handler
handler = FitHandler(
    data_dir='/path/to/data',
    save_dir='/path/to/models',
    animal_name='mouse1',
    session_date='2024-01-15'
)

# Sweep over states
for n_states in range(2, 8):
    handler.fit_and_save(
        model_type='single_taste_poisson',
        states=n_states,
        fit_steps=10000
    )
```

### Multi-Parameter Grid Search

Fit models across multiple parameter combinations:

```python
import itertools
from pytau.changepoint_io import FitHandler

# Define parameter grid
states_list = [2, 3, 4, 5]
bin_widths = [0.025, 0.05, 0.1]
model_types = ['single_taste_poisson', 'single_taste_poisson_varsig']

# Iterate over all combinations
for states, bin_width, model_type in itertools.product(
    states_list, bin_widths, model_types
):
    handler = FitHandler(
        data_dir='/path/to/data',
        save_dir='/path/to/models',
        bin_width=bin_width
    )

    handler.fit_and_save(
        model_type=model_type,
        states=states,
        fit_steps=10000
    )
```

### Parallel Processing

Process multiple datasets in parallel:

```python
from joblib import Parallel, delayed
from pytau.changepoint_io import FitHandler

def fit_session(session_info):
    """Fit model for a single session."""
    handler = FitHandler(
        data_dir=session_info['data_dir'],
        save_dir=session_info['save_dir'],
        animal_name=session_info['animal'],
        session_date=session_info['session']
    )

    handler.fit_and_save(
        model_type='single_taste_poisson',
        states=3,
        fit_steps=10000
    )

# List of sessions to process
sessions = [
    {'animal': 'mouse1', 'session': '2024-01-15', ...},
    {'animal': 'mouse1', 'session': '2024-01-16', ...},
    {'animal': 'mouse2', 'session': '2024-01-15', ...},
    # ... more sessions
]

# Process in parallel
Parallel(n_jobs=4)(
    delayed(fit_session)(session) for session in sessions
)
```

## Parallelization

PyTau supports parallel processing using GNU Parallel with isolated Theano compilation directories to prevent clashes.

### GNU Parallel Setup

```bash
# Install GNU Parallel
sudo apt-get install parallel

# Create job list
cat > jobs.txt << EOF
python fit_model.py --animal mouse1 --session 2024-01-15
python fit_model.py --animal mouse1 --session 2024-01-16
python fit_model.py --animal mouse2 --session 2024-01-15
EOF

# Run in parallel
parallel -j 4 < jobs.txt
```

### Isolated Theano Directories

To prevent compilation clashes, set separate Theano directories for each job:

```python
import os
import tempfile

# Create unique Theano directory for this job
theano_dir = tempfile.mkdtemp(prefix='theano_')
os.environ['THEANO_FLAGS'] = f'base_compiledir={theano_dir}'

# Now import PyMC3 and fit models
import pymc3 as pm
from pytau.changepoint_model import single_taste_poisson, advi_fit

# ... fit models ...

# Cleanup
import shutil
shutil.rmtree(theano_dir)
```

### Batch Script Example

See [single_process.sh](https://github.com/abuzarmahmood/pytau/blob/master/pytau/utils/batch_utils/single_process.sh) for a complete example of parallel processing with isolated compilation directories.

```bash
#!/bin/bash
# single_process.sh

# Set unique Theano directory
export THEANO_FLAGS="base_compiledir=/tmp/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Run Python script
python fit_model.py \
    --animal $1 \
    --session $2 \
    --states $3

# Cleanup
rm -rf /tmp/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
```

## Error Handling and Logging

### Logging Setup

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pytau.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pytau')

# Use in pipeline
logger.info('Starting model fit')
try:
    model, approx = advi_fit(model, fit=10000)
    logger.info('Model fit successful')
except Exception as e:
    logger.error(f'Model fit failed: {e}')
```

### Error Recovery

```python
def robust_fit(handler, model_type, states, max_retries=3):
    """Fit model with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            handler.fit_and_save(
                model_type=model_type,
                states=states,
                fit_steps=10000
            )
            logger.info(f'Fit successful on attempt {attempt + 1}')
            return True
        except Exception as e:
            logger.warning(f'Attempt {attempt + 1} failed: {e}')
            if attempt == max_retries - 1:
                logger.error('All attempts failed')
                return False
    return False
```

## Progress Tracking

### Simple Progress Bar

```python
from tqdm import tqdm

# Track progress over parameter grid
for states in tqdm(range(2, 8), desc='States'):
    for bin_width in tqdm([0.025, 0.05, 0.1], desc='Bin width', leave=False):
        handler.fit_and_save(
            model_type='single_taste_poisson',
            states=states,
            fit_steps=10000
        )
```

### Detailed Progress Tracking

```python
import time
import json

class ProgressTracker:
    def __init__(self, total_jobs):
        self.total_jobs = total_jobs
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.start_time = time.time()

    def update(self, success=True):
        if success:
            self.completed_jobs += 1
        else:
            self.failed_jobs += 1

        elapsed = time.time() - self.start_time
        rate = self.completed_jobs / elapsed if elapsed > 0 else 0
        remaining = (self.total_jobs - self.completed_jobs) / rate if rate > 0 else 0

        print(f'Progress: {self.completed_jobs}/{self.total_jobs} '
              f'({self.failed_jobs} failed) '
              f'[{elapsed:.1f}s elapsed, {remaining:.1f}s remaining]')

    def save(self, path='progress.json'):
        with open(path, 'w') as f:
            json.dump({
                'total_jobs': self.total_jobs,
                'completed_jobs': self.completed_jobs,
                'failed_jobs': self.failed_jobs,
                'elapsed_time': time.time() - self.start_time
            }, f)

# Usage
tracker = ProgressTracker(total_jobs=100)
for job in jobs:
    success = process_job(job)
    tracker.update(success)
tracker.save()
```

## Best Practices

### 1. Organize Data Consistently
Use a consistent directory structure across all experiments.

### 2. Use Database for Metadata
Store all model metadata in the database for easy querying.

### 3. Validate Before Batch Processing
Test pipeline on a single dataset before running large batches.

### 4. Monitor Resource Usage
Track memory and CPU usage to optimize parallelization.

### 5. Implement Checkpointing
Save intermediate results to recover from failures.

### 6. Log Everything
Comprehensive logging helps debug issues in large-scale analyses.

### 7. Version Control Parameters
Save parameter configurations for reproducibility.

### 8. Clean Up Temporary Files
Remove temporary Theano directories after processing.

## Example: Complete Pipeline

```python
import logging
from pytau.changepoint_io import FitHandler, DatabaseHandler
from pytau.changepoint_model import single_taste_poisson, advi_fit
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pytau_pipeline')

# Initialize database
db = DatabaseHandler('models/database.db')

# Define parameter grid
animals = ['mouse1', 'mouse2', 'mouse3']
sessions = ['2024-01-15', '2024-01-16', '2024-01-17']
states_list = [2, 3, 4, 5]

# Process all combinations
total = len(animals) * len(sessions) * len(states_list)
with tqdm(total=total, desc='Processing') as pbar:
    for animal in animals:
        for session in sessions:
            for states in states_list:
                try:
                    # Initialize handler
                    handler = FitHandler(
                        data_dir=f'data/{animal}/{session}',
                        save_dir='models',
                        animal_name=animal,
                        session_date=session
                    )

                    # Fit and save
                    handler.fit_and_save(
                        model_type='single_taste_poisson',
                        states=states,
                        fit_steps=10000
                    )

                    logger.info(f'Completed: {animal}/{session}/states={states}')

                except Exception as e:
                    logger.error(f'Failed: {animal}/{session}/states={states}: {e}')

                finally:
                    pbar.update(1)

logger.info('Pipeline complete')
```

## Next Steps

- See [Available Models](models.md) for model descriptions
- See [Inference Methods](inference.md) for fitting details
- See [Data Formats](data_formats.md) for data preparation
- Check the [API documentation](api.md) for detailed function signatures
