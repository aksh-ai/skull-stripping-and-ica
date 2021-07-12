# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import os
import argparse
import nibabel as nib
from nilearn import plotting, image
from nilearn.decomposition import CanICA

def rsfmri_ica(args):
    # prepare parameters
    parameters = dict(
                n_components=args.n_components,
                mask_strategy='template',
                random_state=0,
                memory_level=args.memory_level,
                memory='cache',
                verbose=args.verbose,
            )
    
    # instantiate the CanICA model
    ica = CanICA(**parameters)
    # perform ica on the input image
    ica.fit(args.input_path)

    # get the components of ica as images
    components_img = ica.components_img_
    
    # save the components as an image
    components_img.to_filename(args.output_path)
    
    # if visualize is selected, atlas map of the components are shown
    if args.visualize:
        plotting.plot_prob_atlas(components_img, title="All ICA Components")
        plotting.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input path to the rs-fMRI image")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path to save the CanICA components image")
    parser.add_argument("-n", "--n_components", type=int, required=False, default=20, help="Number of components to perform CanICA")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=10, help="Prints stats based on verbose value")
    parser.add_argument("-vis", "--visualize", type=bool, required=False, default=True, help="Whether to display the atlas map of the components image or not")
    parser.add_argument("-m", "--memory_level", type=int, required=False, default=2, help="Memory usage value")

    # parse args and compute ica
    rsfmri_ica(parser.parse_args())
    
    exit()