# sportunion-1190-chatbot

Author: Artem Manin

Date: November 2025

## Part 1 - TBD

TDB

## Part 2 - TBD

TBD

## 📂 Repository Structure

TBD

```text
.
├── data/                                   # Dataset (ignored in Git)
│
├── docs/                                   
│   └── system_design_and_architecture.md   # Part 1: System design & Architecture markdown document
│
├── notebooks/                              
│   ├── 01_data_ingestion.ipynb             # Part 2: Data classification notebook
│   ├── 02_model_train.ipynb                # Part 2: Model training notebook
│   └── 03_model_predict.ipynb              # Part 2: Model prediction notebook
│
├── src/                                    
│   └── utils.py                            # Helper functions
│
├── runs/                                   # Training outputs (ignored except key artifacts)
│   └── classify/
│       ├── food101_e20_img224_frac0.1/     # Training run (20 epochs, 224px images, 10% of data)
│       │   ├── weights/
│       │   │   ├── best.pt                 # Best model checkpoint (highest validation accuracy)
│       │   │   ├── last.pt                 # Model state from the final epoch
│       │   │   └── ...                     # Other weight files (if saved)
│       │   ├── args.yaml                   # Training configuration (hyperparameters)
│       │   ├── results.csv                 # Training metrics per epoch (accuracy, loss, etc.)
│       │   └── summary.json                # Summary metadata for the run
│       │
│       ├── food101_e20_img512_frac0.1/     # Run with 20 epochs, 512px images, 10% of data
│       ├── food101_e50_img224_frac0.1/     # Run with 50 epochs, 224px images, 10% of data
│       ├── food101_e50_img512_frac0.1/     # Run with 50 epochs, 512px images, 10% of data
│       └── ...                             # More runs with different configs
│
├── .gitignore                              # Git ignore rules
├── README.md                               # Project description (this file)
└── requirements.txt                        # Python dependencies
```
