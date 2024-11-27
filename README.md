# Project Setup Instructions

To set up the project, follow these steps:

1. **Create a virtual environment:**
    ```bash
    python3 -m venv venv-sentiment
    ```

2. **Activate the virtual environment:**

    - **On macOS/Linux:**
        ```bash
        source venv-sentiment/bin/activate
        ```

    - **On Windows:**
        ```bash
        .\venv-sentimentt\Scripts\activate
        ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

# Using the data preprocessing Script

This project includes a data processing script that reads a CSV file, processes it, and splits it into training and testing datasets using the Hugging Face DatasetDict. The script allows customization of the train-test split ratio and saves the processed dataset in the Hugging Face format for easy use in machine learning workflows.
### Command-Line Arguments:

- **`csv_path`** (required): Path to the input CSV file that you want to process.
  
- **`save_path`** (required): Path where the processed dataset will be saved.

- **`train_percent`** (required): Percentage of data to be used for the training set (e.g., 0.01 for 1%).

- **`test_percent`** (required): Percentage of data to be used for the test set (e.g., 0.001 for 0.1%).


### Example Usage:

```bash
python data_process.py /path/to/csv /path/to/save/directory train_percent test_percent
```

# Fine-tuning RoBERTa on a Custom Dataset - fine_tune.py
This project includes a RoBERTa fine-tuning script that fine-tunes a pre-trained RoBERTa model on a custom dataset for sequence classification tasks, such as sentiment analysis or other types of text classification. The script uses Hugging Face's transformers library and provides an easy way to fine-tune a RoBERTa model for your specific task.
### Command-Line Arguments:

- **`--path_to_dataset`** (required): Path to the dataset in Hugging Face datasets format (can be a local directory or Hugging Face dataset).
  
- **`--epochs`** (required): Number of epochs for training (default is 1).

- **`--batch_size`** (required): Batch size for both training and evaluation (default is 8).

- **`--output_dir`** (required): Directory to save the trained model and tokenizer.


### Example Usage:

```bash
python train_roberta.py \
    --path_to_dataset "huggingface_dataset_twitter" \
    --epochs 3 \
    --batch_size 16 \
    --output_dir "output_model"
```

# Sentiment Analysis API
This project includes a FastAPI-based sentiment analysis API that takes a sentence as input and predicts whether the sentiment is positive or negative using a pre-trained model. The model can be fine-tuned or pre-trained using Hugging Face's transformers library. This API has two endpoints: one for checking the status of the model and server, and another for performing sentiment analysis predictions.

1**Run the App**
    **Start the application using uvicorn:**
    ```bash
    uvicorn main:app --reload
    ```
    **The app will run on http://127.0.0.1:8000 by default.**

2**Sending Requests**
    **To process a sentiment, use the following curl command:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"sentence": "I love this product!"}'
```

# next steps
1. Convert PyTorch model to TensorFlow for faster inference.
2. Clear PyTorch cache after every request to avoid memory accumulation.
3. Convert FastAPI directly into gRPC endpoint with TensorRT for optimized inference.
4. Enable TensorRT inference to run on JavaScript, eliminating the need for additional handshakes.
5. Use CUDA streams to replicate the same model on the same GPU to increase concurrency.

