
# Simple Question-Answering System with PyTorch and RNN

This project is a simple Question-Answering (QA) system built with **PyTorch** and a Recurrent Neural Network (**RNN**). The model is trained on a small dataset of question-answer pairs and learns to predict a single-word answer for a given question.

-----

## Key Features

  * **Simple RNN Architecture**: Implements a foundational NLP model using `nn.Embedding`, `nn.RNN`, and `nn.Linear` layers in PyTorch.
  * **End-to-End Pipeline**: Covers the complete process from raw text preprocessing and vocabulary building to model training and inference.
  * **Custom Data Handling**: Uses a custom PyTorch `Dataset` class to efficiently convert text data into tensors for the `DataLoader`.
  * **Text Preprocessing**: Includes functions for tokenizing text by lowercasing, removing punctuation, and splitting it into words.
  * **Direct Inference**: Features a `predict` function to ask new questions to the trained model and get direct answers.

-----

## Methodology

The system follows a standard NLP pipeline from data preparation to inference.

### **Data Preprocessing**

The process begins by cleaning and preparing the text data.

  * **Tokenization**: Questions and answers are converted to lowercase, punctuation is removed, and the text is split into a list of words (tokens).
  * **Vocabulary Building**: A vocabulary is created by mapping every unique word in the entire dataset to a unique integer index. An `<UNK>` token is reserved for unknown words encountered during inference.

### **Model Architecture**

The QA model is a `SimpleRNN` built using three main layers in PyTorch:

1.  **Embedding Layer**: Converts the integer indices of the input question into dense vector representations (embeddings) of a fixed size.
2.  **RNN Layer**: Processes the sequence of embeddings to capture contextual information. It outputs the final hidden state of the sequence.
3.  **Linear Layer**: A fully connected layer that takes the RNN's final hidden state and maps it to the size of the vocabulary, producing a score for each possible answer word.

### **Training**

The model is trained to minimize the difference between its predictions and the actual answers using the following:

  * **Loss Function**: **CrossEntropyLoss**, which is suitable for multi-class classification problems like predicting the correct word from a vocabulary.
  * **Optimizer**: **Adam**, an efficient optimization algorithm.
  * **Epochs**: The model is trained for 20 epochs, iterating over the entire dataset each time to learn the patterns.

-----

## Results

After 20 epochs of training, the model successfully learns the relationships in the QA dataset, with the training loss decreasing significantly. It can then predict single-word answers for questions similar to those in the training set.

**Example Prediction:**

```python
# Ask the trained model a question
predict(model, "What is the largest planet in our solar system?")

# Expected Output:
# jupiter
```
