# CSE447-project

This repo contains a program that takes in a string of character and tries to predict the next character.

## Input format

`example/test.txt` and `example/test16k.txt` are two files of input.
Each line in this file correspond to a string, for which the program will guess what the next character should be.

## Output format

A file specified by you is outputted each time the program is run. When running the test command, simpl. Each line in the output corresponds to the guess of the program for that particular line in the input file. The program will guess 3 characters for each string.

# Character-Level LSTM Model

This repository contains a character-level Long-Short Term Memory (LSTM) implementation to train and predict sequences of text. The model predicts the next character(s) based on a given input sequence.

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
- `work/`: Directory to save the model.
- `src/modularizedRNN.py`: RNN script for training and testing the model--OUTDATED

---

## How to Run

The script supports two modes: `train` and `test`.

You must run "test" and "train" from the main directory.

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
## Docker
You can also run this file on Docker, by running submit.sh and then using Docker run to run our program.