# Character-Level RNN Model

This repository contains a character-level Recurrent Neural Network (RNN) implementation to train and predict sequences of text. The model predicts the next character(s) based on a given input sequence.

## Features
- Train the model on a custom text dataset.
- Predict the next character(s) for input sequences.
- Supports outputting the top 3 likely predictions for each sequence.
- Saves and loads trained models.

---

## Requirements

- Python 3.x
- NumPy
- PyTorch

---

## Files

- `src/modularizedLSTM.py`: Main script for training and testing the model.
- `example/preds.txt`: Output file containing predictions from the test phase.
- `work/`: Directory to save and load the model.

---

## How to Run

The script supports two modes: `train` and `test`.

### **Training**

Run the script in `train` mode to train the model using a dataset.

```bash
python modularizedRNN.py train --work_dir <work_directory>
```

### **Testing**
Run the script in test mode to evaluate the model on a test dataset and generate predictions.

```bash
python modularizedRNN.py test --work_dir <work_directory> --test_data <test_file> --test_output <output_file>
```
