# N-PIP

**N-PIP** (<u>N</u>annochloropsis - <u>P</u>romoter <u>I</u>nterpretable <u>P</u>rediction) : a novel interpretable deep learning framework
for **promoter identification in Nannochloropsis**

## Required Packages

- python 3.6
- tensorflow-gpu 1.10.0
- keras 2.2.0
- numpy 1.14.5
- matplotlib 3.2.2
- scikit-learn 0.23.2
- scipy 1.5.4

**Scrambler code reference:**

> Linder, Johannes, et al. "Interpreting neural networks for biological sequences by learning stochastic masks." *Nature machine intelligence* 4.1 (2022): 41-54.

> http://www.github.com/johli/scrambler

## Run N-PIP with trained parameter

To start making predictions, you can place your fasta file in the ***promoter_data*** folder, noting that all sequences must be 1,000 kb in length.

```sh
python run_npip.py strain_name fasta_file save_figs
```

Command-line arguments:

- ***strain_name*** can be set to **CCMP525**, **CCMP526**, **CCMP529**, **CCMP531**, **CCMP537**, **IMET1**, determine the trained parameters of which strain to use.

- ***fasta_file*** can be set to your fasta file name, such as *CCMP525.fa*.

- ***save_figs*** can be set to **True** or **False**, noting that if the number of samples is large, it may take a long time to save figures.

Output files will be generated in the ***npip_output*** folder:

- label.npy (predictive confidence)
- importance_scores.npy
- importance_pwm.npy
- figures (optional)

## Retrain N-PIP

If you want to repeat the training steps in the paper, or retrain with your own dataset, please follow these steps:

1. Train the predictor - ADNPPro (this step will run five-fold cross-validation).

   ```sh
   python run_adnnpro_5fold.py pos_data_path neg_data_path n_5fold save_name
   ```

   Command-line arguments:
   
   - ***pos_data_path*** can be set to your positive dataset path, such as *promoter_data\CCMP525.fa*.
   - ***neg_data_path*** can be set to your negative dataset path, such as *promoter_data\randomCCMP525.fa*.
   - ***n_5fold*** determine the repeat number of five-fold cross-validation.
   - ***save_name*** determine the h5 file name you want to set. Models and validation results will be saved in the ***adnppro_trained_model*** folder.
2. Train the interpreter - Scrambler.

   ```sh
   python run_scrambler_train.py pos_data_path load_name save_name
   ```

   Command-line arguments:

   - ***pos_data_path*** can be set to your positive dataset path, such as *promoter_data\CCMP525.fa*. In our work, only positive dataset are used to train the interpreter.
   - ***load_name*** should be set same to the ***save_name*** in the previous step.
   - ***save_name*** also determine the h5 file name you want to set. Models will be saved in the ***scrambler_trained_model*** folder.

