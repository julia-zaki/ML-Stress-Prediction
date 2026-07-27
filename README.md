# ML-Stress-Prediction
Explores how menstrual cycle phase, sleep, stress, and workload shape day-to-day cognitive and physical capacity, and uses those patterns to suggest when to schedule deep work, planning, or social tasks.


This project investigates how cognitive and physical capacities fluctuate across the menstrual cycle and how these natural variations can better inform task planning and effort allocation for women. While most productivity systems assume individuals work at a consistent level each day, this assumption overlooks changes in the menstrual cycle that can influence energy, mood, focus, and performance over time. This project aims to study how the menstrual cycle interacts with factors such as sleep, stress, and workload to shape day-to-day capacity. By analyzing these patterns, we aim to identify individual-specific energy peaks for various tasks, such as deep cognitive work, planning, and social interaction.

Project Literature Review to be uploaded soon!

## Setup

1. Download the [mcPHASES dataset](https://www.physionet.org/content/mcphases/1.0.0/) locally into `McPhases Data/` (raw CSVs are too large for GitHub).
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data Pipeline and Modeling Code

| File | Purpose |
|----------|---------|
| `data_preprocessing.ipynb` | Setup, data merging, handling outliers, data aggregation handling missing data|
| `feature_engineering.ipynb` | Data distribution, target distribtution, stratification, data splitting and validation |
| `modeling.ipynb` | Evaluation metrics, classification models, feature importance and interpretability, limitations |

## Documentation

Model and pipeline details: [`docs/models/README.txt`](docs/models/README.txt)
