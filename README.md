# **Image-Classifier**

Python command line application that can train an image classifier on a dataset , then predict new images using the trained model.

## Requirements

The Code is written in Python 3 . If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

to upgrade Python

`pip install python --upgrade`

Additional Packages that are required are: Numpy, Pandas, Matplotlib, Pytorch, PIL and json.
You can download them using pip

`pip install numpy pandas matplotlib pil`

To intall Pytorch head over to the Pytorch site select your specs and follow the instructions given.

## Command Line Application

* Train a new network on a dataset using `train.py`

   - Basic Usage : `python train.py data_directory`
   - Prints out training loss, validation loss, and validation accuracy 
     as the network trains
   - Options
        - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
        - Choose architecture: `python train.py data_dir --arch "vgg"`
        - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
        - Use GPU for training: `python train.py data_dir --gpu`
* Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
   
   - Basic usage: `python predict.py /path/to/image checkpoint`
   - Options
        - Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
        - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
        - Use GPU for inference: `python predict.py input checkpoint --gpu`

## JSON File

In order for the network to print out the name of the flower a .json file is required. By using a .json file the data can be sorted into folders with numbers and those numbers will correspond to specific names specified in the .json file.


