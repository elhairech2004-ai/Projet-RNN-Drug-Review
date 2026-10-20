# Drug Review Sentiment Classification

Comparison of three manually implemented recurrent architectures
using PyTorch: Vanilla LSTM, CIFG and WMC.

## Dataset

UCI Drug Review Dataset (Drugs.com).

Reviews are classified using their ratings:
- Negative: ratings 1–4
- Neutral: ratings 5–6
- Positive: ratings 7–10

After cleaning and removing duplicate texts:
- Training: 64,140 reviews
- Validation: 16,036 reviews
- Test: 48,266 reviews

The vocabulary is built exclusively from the training set.

## Models

All recurrent cells are implemented manually with nn.Linear,
without using nn.LSTM.

- Vanilla LSTM: 2,691,971 trainable parameters
- CIFG: 2,659,075 trainable parameters
- WMC: 2,741,123 trainable parameters

## Experimental setup

- Vocabulary size: 20,000
- Maximum sequence length: 128 tokens
- Embedding dimension: 128
- Hidden dimension: 128
- Dropout: 0.3
- Training budget: 5 epochs per experiment
- Baseline optimizer: Adam
- Baseline learning rate: 0.001
- Baseline batch size: 64
- Random seed: 42
- Class-weighted cross-entropy loss

Additional Vanilla experiments compare learning rates,
Adam versus RMSprop, and batch sizes 64 versus 128.

Models and checkpoints are selected using validation Macro-F1.

## Test results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Vanilla LSTM | 68.35% | 0.5771 |
| CIFG | 71.59% | 0.5923 |
| WMC | 70.68% | 0.5857 |
| Tuned Vanilla with RMSprop | 71.97% | 0.5874 |

CIFG is the final model selected on validation.
It also achieves the highest test Macro-F1.
Neutral reviews remain the most difficult class.

## Running the notebook

1. Download the .ipynb file from this repository.
2. Import it into Google Colab.
3. Mount your own Google Drive when prompted.
4. Run the cells in order.

The notebook downloads the dataset and saves checkpoints to Drive.
A GPU is recommended for training. CPU evaluation is supported.

Enable RUN_HYPERPARAMETERS for the additional experiments.
Enable RUN_FINAL_TEST after completing model selection.

## Files

- .ipynb: implementation and experiments
- Rapport_RNN_Drug_Review.pdf: experimental report

## Limitations

Results are based on one random seed and five training epochs.
Only Vanilla received additional hyperparameter tuning.
The architecture ranking requires confirmation across multiple runs.
