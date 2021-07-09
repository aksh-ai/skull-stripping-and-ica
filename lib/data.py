# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import torch as th
import torchio as tio
import multiprocessing
from sklearn.model_selection import train_test_split
from lib.utils import get_train_transforms, get_validation_transforms

def get_dataset_from_path(image_paths: any, label_paths: any) -> tio.SubjectsDataset:
    '''
    A function to load dataset from the given paths of images and labels (masks)

    Args:
      * image_paths: Directory path of the images
      * label_paths: Directory path of the labels

    Returns:
      * dataset: Returns a tio subjects dataset containing images and labels
    '''
    # check if number of smaples are equal or not
    assert len(image_paths) == len(label_paths), "Number of samples in images and labels don't match"

    # create an empty list to store the dataset
    subjects = []
    
    # iterate through the image and label paths and store them as Subjects in the list
    for image_path, label_path in zip(image_paths, label_paths):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        subjects.append(subject)

    # return the images and labels as a subject dataset
    return tio.SubjectsDataset(subjects)

def load_datasets(image_paths: any, label_paths: any, test_size: float = 0.1, random_state: int = None, train_transforms: any = None, valid_transforms: any = None, volume: str = "whole", patch_size: int = 128, samples_per_volume: int = 128, max_queue_length: int = 128):
    '''
    A function to load datasets from the given paths of images and labels (masks) as whole images or random sub-volumes
    (patches) for training and validation/testing.

    Args:
      * image_paths: Directory path of the images
      * label_paths: Directory path of the labels
      * test_size: Test size for splitting the dataset into training and validation set
      * random_state: An integer specifiying a seed for reproducability in the data split
      * train_transforms: Transforms object containing necessary transformations of tio or torchvision for data preprocessing and augmentation that has to applied on the training set
      * valid_transforms: Transforms object containing necessary transformations of tio or torchvision for data preprocessing and augmentation that has to applied on the validation set
      * volume: If "patch" or "patches" is given, random sub-volumes (patches) will be generated on the fly for training and validation
      * patch_size: An integer specifying the size of the sub-volume (patches) to be generated
      * samples_per_volume: An integer specifying how many samples has to be generated per sub-volume
      * max_queue_length: An integer alue specifying the batch size for the queue of sub-volumes

    Returns:
      * training_set: A torch dataset with on-the-fly preprocesisng and transformations that can be used with data loader 
      * validation_set: A torch dataset with on-the-fly preprocesisng and transformations that can be used with data loader 
    '''
    # check if number of smaples are equal or not
    assert len(image_paths) == len(label_paths), "Number of samples and labels are not equal"

    # get the splitted dataset paths base dn test size
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, label_paths, test_size=test_size, random_state=random_state)

    # empty list to store the training set
    training_set = []
    
    # iterate through the image and label paths and store them as Subjects in the list
    for image_path, label_path in zip(X_train, y_train):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        training_set.append(subject) 

    # get the training set as a Subjects Dataset with transforms (preprocessing & augmentation)
    training_set = tio.SubjectsDataset(training_set, transform=get_train_transforms() if train_transforms is None else train_transforms)

    # empty list to store the validation set
    validation_set = []
    
    # iterate through the image and label paths and store them as Subjects in the list
    for image_path, label_path in zip(X_valid, y_valid):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        validation_set.append(subject)

    # get the validation set as a Subjects Dataset with transforms (preprocessing & augmentation)
    validation_set = tio.SubjectsDataset(validation_set, transform=get_validation_transforms() if valid_transforms is None else valid_transforms)

    # Generate random sub-volumes (patches) from the dataset as a Queue dataset
    if volume.lower() in ['patch', 'patches']:
        # sampler for random sub-volume generation
        sampler = tio.data.UniformSampler(patch_size)
        
        # get the sub-volume based training set
        training_set = tio.Queue(
            subjects_dataset=training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        # get the sub-volume based validation set
        validation_set = tio.Queue(
            subjects_dataset=validation_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            shuffle_subjects=False,
            shuffle_patches=False,
        )
    
    # print dataset stats
    print(f"Volume Mode: {volume.upper()} | Dataset: {len(image_paths)} Images")
    print(f'Training set: {len(training_set)} Images')
    print(f'Validation set: {len(validation_set)} Images')
    
    # return the loaded datasets
    return training_set, validation_set