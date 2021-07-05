import torch as th
import torchio as tio
import multiprocessing
from torch.utils.data import Dataset, DataLoader, random_split
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

def get_datasets(images, 
                labels, 
                test_size=0.1, 
                random_state=51, 
                volume="whole", 
                patch_size=32,
                samples_per_volume=32,
                max_queue_length=128,
                num_workers=multiprocessing.cpu_count()):
    dataset = get_dataset_from_path(images, labels)
    
    data_length = len(dataset)
    validation_length = int(test_size * data_length)
    training_length = data_length - validation_length

    training_set, validation_set= random_split(dataset, (training_length, validation_length), generator=th.Generator().manual_seed(random_state))

    training_set = tio.SubjectsDataset(training_set, transform=get_train_transforms())

    validation_set = tio.SubjectsDataset(validation_set, transform=get_validation_transforms())

    print(f"Dataset: {len(dataset)} Images")
    print(f'Training set: {len(training_set)} Images')
    print(f'Validation set: {len(validation_set)} Images')

    if volume.lower() == 'whole':
        return training_set, validation_set
    
    elif volume.lower() in ['patch', 'patches']:
        sampler = tio.data.UniformSampler(patch_size)
        
        training_patches = tio.Queue(
            subjects_dataset=training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        validation_patches = tio.Queue(
            subjects_dataset=validation_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        return training_patches, validation_patches