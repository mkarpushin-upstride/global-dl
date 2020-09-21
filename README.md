# UpStride Global DL

This repository contains modules that are common across all deep learning computer vision applications. 

### What is this repository for?

* Packages and modules which can be used globally across computer vision applications. 

### How do I get set up?

This repository should be set as a submodule in order to use the packages/modules or scripts with this repository. 

### I have some free time and I want to help, what can I do ?

Fist, thanks a lot ! We are very grateful to everyone who wants to contribute. The easiest way is to raise an issue if you discover any bug or if you have any idea for improving this project.
If you wish to contribute, pull request are welcome. We will try to review them as fast as possible. 
And if you could write some unit-tests at the same time you submit a pull request, it will be even easier for us to integrate your work :)

### Repository organization

A deep learning project often has the same steps : 

- Preprocess the data and prepare them for training, for instance by writing them in TfRecord file
- Run an architecture search, a tuner or just a simple training task
- Export the model to a deployment friendly file

the organization of this code follows these steps:

* `global_conf.py` contains functions to configure tensorflow (xla, memory_growth, ...) and needed for each of these steps
* `keras_tuner/training.py` contains functions useful for keras-tuner. for now the training_arguments
* `tfrecord/create_tfrecord.py` contains tools to simplify the creation of TFRecord files
* training
    - `alchemy_api.py` : all functions and callbacks to communicate with the Alchemy plateform. Require `metrics`
    - `export.py` : functions to send data to aws or upstride plateform : Require `trt_convert` and `alchemy_api`
    - `metrics.py` : functions to compute accuracy, number of trainable parameters, flops, information_density and net_score
    - `optimizers.py` : functions to get optimizers and lr scheduler. 
    - `training.py` : base code for training NN. Require `optimizer`
    - `trt_convert.py` : function to convert to tensorRT. This one will probably not be needed anymore as Upstride engine can run on any plateform

### how do I test ?

run `python test.py`

