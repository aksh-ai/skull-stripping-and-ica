import torch as th
import torchio as tio
import multiprocessing
from sklearn.model_selection import train_test_split
from lib.utils import get_train_transforms, get_validation_transforms

def get_dataset_from_path(image_paths, label_paths):
    assert len(image_paths) == len(label_paths)

    subjects = []
    
    for image_path, label_path in zip(image_paths, label_paths):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        subjects.append(subject)

    return tio.SubjectsDataset(subjects)

def load_datasets(
    image_paths, 
    label_paths, 
    test_size=0.1, 
    random_state=None, 
    volume="whole", 
    patch_size=128,
    samples_per_volume=128,
    max_queue_length=128):
    assert len(image_paths) == len(label_paths), "Number of samples and labels are not equal"

    data_length = len(image_paths)
    validation_length = int(test_size * data_length)
    training_length = data_length - validation_length

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, label_paths, test_size=test_size, random_state=random_state)

    training_set = []
    
    for image_path, label_path in zip(X_train, y_train):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        training_set.append(subject) 

    training_set = tio.SubjectsDataset(training_set, transform=get_train_transforms())

    validation_set = []
    
    for image_path, label_path in zip(X_valid, y_valid):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        validation_set.append(subject)

    validation_set = tio.SubjectsDataset(validation_set, transform=get_validation_transforms())

    if volume.lower() in ['patch', 'patches']:
        sampler = tio.data.UniformSampler(patch_size)
        
        training_set = tio.Queue(
            subjects_dataset=training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        validation_set = tio.Queue(
            subjects_dataset=validation_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            shuffle_subjects=False,
            shuffle_patches=False,
        )
    
    print(f"Volume Mode: {volume.upper()} | Dataset: {len(image_paths)} Images")
    print(f'Training set: {len(training_set)} Images')
    print(f'Validation set: {len(validation_set)} Images')
    
    return training_set, validation_set

def generate_patches():
    pass

