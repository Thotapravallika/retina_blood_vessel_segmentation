import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate


# Create directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# load data
def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)


# data augmentation
def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for ind, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extract the name
        name = x.split("/")[-1].split(".")[0]

        # Read image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment:

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]


        else:

            X = [x]
            Y = [y]

        # Resizing images and masks
        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """ Seeding """

    np.random.seed(42)

    """ Load the data"""
    data_path = "/home/akash/PycharmProjects/retina_blood_vessel_segmentation/dataset"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"train: {len(train_x)}, {len(train_y)}")
    print(f"test: {len(test_x)}, {len(test_y)}")

    """Create directories to save the augmented data"""

    create_dir("augmented_data/train/images/")
    create_dir("augmented_data/train/masks/")
    create_dir("augmented_data/test/images/")
    create_dir("augmented_data/test/masks/")

    # Data augmentation
    augment_data(train_x, train_y, "augmented_data/train/", augment=True)
    augment_data(test_x, test_y, "augmented_data/test", augment=False)
