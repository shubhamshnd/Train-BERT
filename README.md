
# BERT Training and Chatting

This repository contains scripts to fine-tune a BERT model for question-answering tasks and to use the trained model for chatting.

## Files in the Repository

- `chat_bert_ft.py`: Script for fine-tuning a BERT model for question-answering tasks.
- `Train.py`: Script for training the model using the provided dataset.
- `qa_train.csv`: Example training dataset for the question-answering model.

## Preparing the Data

To prepare the data, create a CSV file with the format of `qa_train.csv` containing the following columns:
- `question`: The question to be answered by the model.
- `answer`: The correct answer to the question.

## Usage

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the training script:
    ```bash
    python Train.py
    ```

3. After training, use the fine-tuned model for chatting:
    ```bash
    python chat_bert_ft.py
    ```

## Requirements

The required Python packages are listed in the `requirements.txt` file.
