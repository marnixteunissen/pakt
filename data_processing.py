import os
import tensorflow.keras.preprocessing as preprocessing
import random
import shutil
import sys
from pathlib import Path


def train_test_split(channel_dir, channel, test_split=0.2):
    """
    Split the data into training and test data
    :param project_dir: Path (string)
                        Path to the project directory
    :param test_split:  Float, Deflaut: 0.2
                        Percentage of the data used for testing
    :param startstr:    String, Default: 'Video'
                        Start string of the folder name containing the videos
    :param part:        Float, Default; 1.0
                        Percentage of the video data used to create the samples (train+test)
    :return:            Lists
                        Two listst of video directory names, training data, testing data
    """
    class_dirs = os.listdir(channel_dir)
    data_dir = Path(channel_dir).parent.absolute()
    print(data_dir)
    test_dir = os.path.join(data_dir, channel + '_test')
    os.mkdir(test_dir)
    for cl in class_dirs:
        os.mkdir(os.path.join(test_dir, cl))
        files = os.listdir(os.path.join(channel_dir, cl))

        random.shuffle(files)
        nr_test = int(test_split * len(files))

        test_data = files[:nr_test]
        for file in test_data:
            shutil.move(os.path.join(channel_dir, cl, file), os.path.join(test_dir, cl, file))


def create_data_sets(data_dir, channel, batch_size=8, image_size=(360, 640), split=0.10):
    channel_dir = os.path.join(data_dir, channel)
    if channel + '_test' not in os.listdir(data_dir):
        train_test_split(channel_dir, channel)
    height, width = image_size[0], image_size[1]

    train_ds = preprocessing.image_dataset_from_directory(
        channel_dir,
        validation_split=split,
        subset="training",
        seed=123,
        image_size=(height, width),
        smart_resize=True,
        batch_size=batch_size)

    val_ds = preprocessing.image_dataset_from_directory(
        channel_dir,
        validation_split=split,
        subset="validation",
        seed=123,
        image_size=(height, width),
        smart_resize=True)

    test_ds = preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, channel + '_test'),
        image_size=(height, width),
        smart_resize=True)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    dir = r'C:\Users\marni\Documents\Pakt\data'
    ch = 'back'
    train, val, test = create_data_sets(dir, ch)
    print('Classes:', train.class_names)