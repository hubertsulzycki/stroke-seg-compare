# stroke-seg-compare
**Master's Thesis:** Zastosowanie technik uczenia maszynowego do automatycznego wykrywania zmian związanych z udarem na obrazach tomografii komputerowej mózgu (Application of machine learning techniques for the automatic detection of stroke-related changes in brain computed tomography images)

This repository contains the implementation and evaluation of Deep Learning models for the automatic segmentation of ischemic stroke lesions based on non-contrast computed tomography (NCCT) scans. The project utilizes PyTorch and the MONAI framework.

A core focus of this research is comparing the impact of data dimensionality (2D, 2.5D, and 3D) on segmentation quality using medical imaging architectures, including U-Net, ResU-Net, Attention U-Net, SegResNet, Swin UNETR, and V-Net.

## Prerequisites

* **Python**: 3.13.2
* **Hardwar**e: A CUDA-enabled GPU is highly recommended for training the models, especially for volumetric (3D) architectures.

## Setup & Installation
1. **Clone this repository to your local machine**:

   ```bash
   git clone https://github.com/hubertsulzycki/stroke-seg-compare
   cd stroke-seg-compare
   ```

2. **Create and activate a virtual environment (recommended)**:

    ```bash
   # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux / macOS
    python3 -m venv venv
    source venv/bin/activate
   ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **(Crucial for GPU Acceleration) Install proper PyTorch version**: 
The previous step installs the default version of PyTorch. To train models efficiently using an NVIDIA GPU, you must install the CUDA-enabled version of PyTorch. Run the command tailored to your CUDA version (e.g., CUDA 12.x or 13.x). Check the [official PyTorch website](https://pytorch.org/) for the exact command matching your system.

## Dataset Preparation
1. Download the dataset (only `image.zip` and `mask.zip` are required) from: https://github.com/GriffinLiang/AISD
2. In the root directory of the project, create the following folder structure: `data/raw/images/` and `data/raw/masks/`
3. Extract the downloaded data so that each patient has their own dedicated folder inside both the images and masks directories

**Expected directory structure after extraction**:
```
stroke-seg-compare/
├── configs/
│   └── data_splits.json         # Pre-calculated splits (included in repo)
├── data/
│   └── raw/
│       ├── images/              # Raw NCCT scans (.png)
│       │   ├── patient_01/
│       │   ├── patient_02/
│       │   └── ...
│       └── masks/               # Ground Truth expert masks (.png)
│           ├── patient_01/
│           ├── patient_02/
│           └── ...
├── scripts/                     # Scripts for training and evaluations
├── src/                         # Core source code (data splitter, models, trainer, evaluator)
│   └── data/                    
│   └── evaluation/
│   └── models/
│   └── training/               
├── requirements.txt
└── README.md
```

## Execution Pipeline
To reproduce the experiments from the thesis, execute the following steps in order:

### Step 1: Data Splitting
A pre-calculated patient split is already included in the repository (`configs/data_splits.json`) to ensure full reproducibility of the thesis results. You can proceed directly to training.

However, if you want to generate your own random split from scratch, you can run the splitter script. It will automatically divide the patient list into train, validation, and test sets to prevent data leakage and overwrite the config file:

```bash
python -m src.data.splitter
```
### Step 2: Model Training
The main training engine is located in `train.py`. Before running, you can adjust configuration variables (such as `ARCHITECTURE`, `MODE`, `BATCH_SIZE`) directly inside the script. Then you can run the script using: 

```bash
python -m scripts.train
```

The best model weights will be automatically saved in the `trained_models/` directory.

### Evaluation
To test the trained model on the test set and compute volume-based metrics (Dice, HD95, Sensitivity, Precision), adjust the model weights filename in the evaluation script and run:

```bash
python -m scripts.evaluation
```

## Author
**Hubert Sulżycki** 

Poznan University of Technology, 2026