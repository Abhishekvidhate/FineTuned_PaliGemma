import pandas as pd
import os
from datasets import Dataset, DatasetDict, Features, Value, Image

# define paths to your train and test csv files

data_dir = ""

train_csv_file = os.path.join(data_dir, "train_caption.csv")
test_csv_file = os.path.join(data_dir, "test_caption.csv")

# function to load images
def load_image(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()

# create a huggingface dataset from dataframe
def create_dataset(csv_file, split_name):
    df = pd.read_csv(csv_file)  # load csv files with captions
    df = df.dropna(subset=['messages'])
    df['image'] = df['image_name'].map(lambda image_name: load_image(os.path.join('', image_name)))
    df = df.drop(columns=['image_name'])
    df = df.reset_index(drop=True)
    #define features of dataset
    features = Features({
        'image': Image(),
        'messages': Value('Dict')
    })
    #create dataset object from dataframe
    dataset = Dataset.from_pandas(df, features=features)
    return dataset


# Create train and test datasets
train_dataset = create_dataset(train_csv_file, "train")
test_dataset = create_dataset(test_csv_file, "test")

# combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})


# push to huggingface
dataset_dict.push_to_hub("abhishekvidhate/PhysicsKinematisQA")
