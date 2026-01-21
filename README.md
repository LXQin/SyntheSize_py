# SyntheSize

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**A Python package for optimizing sample size in supervised machine learning with bulk transcriptomic sequencing data.**

SyntheSize is a supervised learning framework designed for determining the optimal sample size for classification tasks by utilizing synthesized data across various sample sizes. This framework employs the Inverse Power Law Function (IPLF) to accurately model different sample sizes and their corresponding classification accuracies.

### Publication

This package implements the algorithm described in:

> Qi Y, Wang X, Qin LX. "Optimizing sample size for supervised machine learning with bulk transcriptomic sequencing: a learning curve approach." *Briefings in Bioinformatics*. 2025 Mar 12;26(2):bbaf097.
> 
> **Paper:** https://academic.oup.com/bib/article/26/2/bbaf097/8071685

**Original R Implementation:** https://github.com/LXQin/SyntheSize

### Key Features

- **Multiple Classification Methods:** Support for Logistic Regression, SVM, KNN, Random Forest, and XGBoost
- **Learning Curve Fitting:** Uses the Inverse Power Law Function (IPLF) to model classification accuracy vs. sample size
- **Data Quality Assessment:** Heatmap and UMAP visualizations for evaluating generated/augmented data
- **Sample Size Optimization:** Determine the optimal sample size for your machine learning task
- **Transcriptomics Focus:** Specifically designed for microRNA-seq and RNA-seq data

### Use Cases

SyntheSize is particularly useful for:

1. **Study Design:** Determine required sample size before data collection
2. **Resource Optimization:** Balance cost vs. classification accuracy
3. **Method Comparison:** Compare different ML methods for your specific data
4. **Data Augmentation Validation:** Assess quality of synthetic data generation

## Installation

### From PyPI Test
<!-- TODO release on PyPI and update install command -->

```bash
pip install synthesize --index-url https://test.pypi.org/simple/
```

### From Source / For Development

```bash
git clone https://github.com/LXQin/SyntheSize_py.git
cd SyntheSize_py
make install-dev  
```
<!-- TODO Update to proper Makefile command -->

## Quick Start
<!-- TODO Update this to use the included data -->

```python
import pandas as pd
import numpy as np
from synthesize import eval_classifier, vis_classifier

# Load your data
# Assume data is a DataFrame with features and a 'groups' column

# Evaluate classifier performance across sample sizes
metric_real = eval_classifier(
    whole_generated=your_data.drop('groups', axis=1),
    whole_groups=your_data['groups'],
    n_candidate=[20, 40, 60, 80, 100],
    n_draw=10,
    methods=['LOGIS', 'SVM', 'XGB']
)

# Visualize learning curves and project sample size requirements
vis_classifier(
    metric_real=metric_real,
    n_target=[120, 150, 200],
    metric_name='f1_score'
)
```

## Core Functions

### Classification Methods

- **`LOGIS`**: Logistic Regression with Ridge penalty (L2 regularization)
- **`SVM`**: Support Vector Machine classifier
- **`KNN`**: K-Nearest Neighbors (k=5)
- **`RF`**: Random Forest with 100 estimators
- **`XGB`**: XGBoost gradient boosting

### Evaluation Functions

- **`eval_classifier`**: Evaluate classification performance across different sample sizes using stratified k-fold cross-validation
- **`heatmap_eval`**: Generate heatmap for comparing real and generated data
- **`UMAP_eval`**: UMAP visualization for data quality assessment
- **`vis_classifier`**: Visualize learning curves and IPLF projections

### Utility Functions

- **`fit_curve`**: Fit Inverse Power Law Function to accuracy vs. sample size data
- **`get_data_metrics`**: Load and preprocess real and generated datasets
- **`visualize`**: Create heatmap and UMAP visualizations
- **`sample_real_generated`**: Sample from both real and generated datasets

## Example Data

The package includes example datasets from TCGA breast cancer (BRCA) studies:
- `BRCASubtypeSel_test.csv`: Test dataset for BRCA subtype classification
- `BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv`: Generated/augmented training data

Access them from the `synthesize.Case` subpackage.

## Documentation

<!-- TODO Add link to the documentation page. -->
Documentation is available in the `docs/` directory and includes:
- API reference for all functions
- Example Jupyter notebooks
- Usage guides

Build documentation locally:
```bash
cd docs
make html
```
<!-- TODO Update to use the main Makefile -->



## Requirements

- Python >= 3.6
- PyTorch >= 1.3.1
- scikit-learn >= 1.0.0
- pandas >= 1.0.5
- numpy >= 1.19.1
- See `pyproject.toml` or `requirements.txt` for complete dependency list

## Citation
If you use SyntheSize in your research, please cite:

```bibtex
@article{qi2025synthesize,
  title={Optimizing sample size for supervised machine learning with bulk transcriptomic sequencing: a learning curve approach},
  author={Qi, Yunhui and Wang, Xinyi and Qin, Li-Xuan},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={2},
  pages={bbaf097},
  year={2025},
  publisher={Oxford University Press}
}
```

## Links

- **Research Paper:** https://academic.oup.com/bib/article/26/2/bbaf097/8071685
- **Original R Implementation:** https://github.com/LXQin/SyntheSize
- **SyNG-BTS for Data Synthesis:** https://github.com/LXQin/SyNG-BTS 
- **Documentation:** Built with Sphinx (see `docs/` directory) 
<!-- TODO Update documentation link -->


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the LICENSE file for details.

---

**Note:** This is the Python implementation of the SyntheSize algorithm. For the original R implementation and additional resources, see https://github.com/LXQin/SyntheSize
