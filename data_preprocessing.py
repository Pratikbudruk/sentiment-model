import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import argparse

def load_and_split_csv(csv_path: str, save_path: str, train_percent: float, test_percent: float):
    """
    Function to load a CSV, clean it, split into train/test datasets,
    and save as a Hugging Face DatasetDict.

    Args:
    - csv_path (str): Path to the CSV file.
    - save_path (str): Path to save the cleaned dataset and splits.
    - train_percent (float): Percentage of data for the training set.
    - test_percent (float): Percentage of data for the testing set.

    Returns:
    - dataset_dict (DatasetDict): A dictionary with 'train' and 'test' splits.
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Attempting to clean and reload the file.")

        try:
            with open(csv_path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace")

            cleaned_path = save_path + "/cleaned.csv"
            with open(cleaned_path, "w", encoding="utf-8") as f:
                f.write(content)

            df = pd.read_csv(cleaned_path)
            print("File cleaned and loaded successfully.")

        except Exception as e:
            print(f"Error during cleaning: {e}")
            return None

    if df.shape[1] < 2:
        print("Error: The CSV file does not have enough columns.")
        return None

    # Assuming the first column is 'label' and last column is 'twitter'
    first_column = df.iloc[:, 0]
    last_column = df.iloc[:, -1]

    # Create new DataFrame
    new_df = pd.DataFrame({
        'label': first_column,
        'twitter': last_column
    })

    total_rows = len(new_df)
    train_size = int(total_rows * train_percent)  # Percentage of total data for training
    test_size = int(total_rows * test_percent)  # Percentage of total data for testing

    train_df, temp_df = train_test_split(new_df, train_size=train_size, random_state=42)
    test_df, _ = train_test_split(temp_df, train_size=test_size, random_state=42)

    # Step 5: Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Step 6: Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    dataset_dict.save_to_disk(save_path + "/twitter_dataset")

    return dataset_dict


def main():
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Process and split a CSV file into Hugging Face Dataset")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("save_path", type=str, help="Path to save the output Hugging Face dataset")
    parser.add_argument("train_percent", type=float, help="Percentage of data for the training set (e.g., 0.01 for 1%)")
    parser.add_argument("test_percent", type=float,
                        help="Percentage of data for the testing set (e.g., 0.001 for 0.1%)")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to load, split, and save the dataset
    dataset_dict = load_and_split_csv(args.csv_path, args.save_path, args.train_percent, args.test_percent)

    # Check if the dataset was successfully processed
    if dataset_dict:
        print("Dataset loaded and split successfully!")
        print(dataset_dict)
    else:
        print("An error occurred during processing.")


if __name__ == "__main__":
    main()

# python your_script_name.py /path/to/csv /path/to/save/directory 0.01 0.001
