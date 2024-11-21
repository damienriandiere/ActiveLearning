import tensorflow_datasets as tfds
from datasetB_main import MyDataset

def test_my_dataset():
    builder = MyDataset()
    builder.download_and_prepare()
    dataset = builder.as_dataset(split="train")
    print(dataset)
    assert dataset is not None

test_my_dataset()