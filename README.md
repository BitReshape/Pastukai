# Pastukai


## Set up of Dev environment
- Install Python 3.8 
- Install IDE (Visual Studio Code or Pycharm)
- Install pip dependencies

``pip install -r requirements.txt``

## Set up of dataset
- Download the 2019 image training dataset on following page:
https://challenge.isic-archive.com/data
- In the data directory you find the split_data.py script.
Run it with following parameters:

    `python split_data.py --path ${PATH_TO_YOUR_UNZIPPED_DATASET}`

    This will create under data a directory images with training_set and
    val_set as subdirectories with the corresponding images.
 