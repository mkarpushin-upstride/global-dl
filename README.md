# README

This is a global repository which would contain modules that are common across all deep learning computer vision applications. 

### What is this repository for?

* Packages and modules which can be used globally. 

### How do I get set up?

This repository should be clone as a submodule in order to use the packages/modules or scripts with this repository. 

### Contribution guidelines

* Writing tests
* Code review

### Who do I talk to?

* Benchmark Team 

### Repository organisation
* global_conf.py : this file contains functions to configure tensorflow (xla, memory_growth, ...)
* keras_tuner/training.py : contains functions usefull for keras-tuner. for now the training_arguments
* tfrecord/create_tfrecord.py : contain the definition of the TfRecordManager and Split objects
* training
    - alchemy_api.py : all functions and callbacks to communicate with the Alchemy plateform. Require `metrics`
    - export : functions to send data to aws or upstride plateform : Require `trt_convert` and `alchemy_api`
    - metrics : functions to compute accuracy, number of trainable parameters, flops, information_density and net_score
    - optimizers : functions to get optimizers and lr scheduler. 
    - training : base code for training NN. Require `optimizer`
    - trt_convert : function to convert to tensorRT. This one will probably not be needed anymore as Upstride engine can run on any plateform
### how do I test ?

run `python test.py`


### I have some free time, what task can I do ?
