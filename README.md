# DS535 Project

This repository contains the implementation and necessary components for running the DS535 project. Below is a detailed structure and guide to setting up and executing the experiments.

## Repository Structure
```plaintext
DS535
├── data /{start_year}_{end_year}_pid
├── config
├── model /{start_year}_{end_year}_pid
├── results
├── algo_~~~.py
├── train_eval.py (main)
├── util.py
```

### Directory Description
1. **data/**
   - Contains datasets structured by `{start_year}_{end_year}_pid`. Please download the datasets using the link below and place them in this folder.
     
     **Download Link:** [Google Drive](https://drive.google.com/drive/folders/1L-oN86PMDzC-YH3v7l8aCMGnVCNvTVhb?usp=sharing)

2. **config/**
   - Contains configuration files for training and evaluation setups.

3. **model/**
   - Stores trained models structured by `{start_year}_{end_year}_pid`.

4. **results/**
   - Stores experiment results.

5. **algo_~~~.py**
   - Contains algorithm-specific implementation files.

6. **train_eval.py**
   - The main script for training and evaluating models.

7. **util.py**
   - Contains utility functions used throughout the project.

---

## Environment Setup
This project is built to run on **Google Colab**. Use the provided `requirements.txt` file to install the necessary dependencies:

```bash
python 3.10.12
pip install -r requirements.txt
```

---

## Running the Project
Execute the main script as follows:

```bash
python train_eval.py config={config_name} pid=0 device=0
```

### Arguments
- **config**: Specifies the experiment setup.
  - Example: `training_algorithm_name.config`
- **pid**: Used for seed configuration (experiments with the same year but different seeds result in varied outputs). Default: `0`.
- **device**: Specifies which CUDA device to use (if applicable). Default: `0`.

---

## Configuration Files
- **training_{algorithm_name}.config**
  - Configuration for training the specified algorithm.
- **eval.config**
  - Configuration for evaluating spatial algorithms.
- **eval_temporal.config**
  - Configuration for evaluating spatio-temporal algorithms.

---

## Transformer Workflow
To execute Transformer-related experiments, follow these steps:

1. **Generate Preprocessed Data**
   - Run the preprocessing script to create train, validation, and test pickle files.
   ```bash
   python preprocess_transformer.py
   ```

2. **Create Configuration Files**
   - Create `train_tf.config` and `eval_tf.config` files.
   - Ensure `tf=True` is included in both files.

3. **Train the Transformer Model**
   - Use the following command to train the model:
   ```bash
   accelerate launch train_eval_tf.py config=train_tf.config
   ```

4. **Evaluate the Transformer Model**
   - Use the following command to evaluate the model:
   ```bash
   accelerate launch train_eval_tf.py config=eval_tf.config
   ```

---

## Additional Notes
1. When running **eval.config** or **eval_temporal.config** for the first time, the script automatically creates a dataset named `2023_2023_0` in the `data/` folder. 
   - **Caution:** This dataset can be very large.

2. The **temporal_GraphSage_GRU** algorithm is not trained due to time constraints.

---

For any further details or questions, please reach out or refer to the documentation in this repository.

