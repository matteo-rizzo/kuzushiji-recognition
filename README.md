# CognitiveServices
Didactic project of the Cognitive Services course of the Computer Science master's degree of University of Padua (Italy).


## Working environment

This is the recommended environment, but it may work also with other versions.

- Python 3.6.8 (&ge; 3.6 required)
- Tensorflow 1.14.0

## Project setup

1. Clone/download the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Modify configuration parameters as described in next section.

### Test configuration
To evaluate the models follow this steps:

1. Download the Kaggle dataset from [this page](https://www.kaggle.com/c/kuzushiji-recognition/data) 
  (about 7 GB), create a folder with the following structure:

   ```
       repo_folder/
       ├── datasets/
       │   └── kaggle/
       │       ├── testing/
       │       │   └── images
       │       └── training/
       │           └── images
   ```
   Decompress `test_images.zip` and place all the images in `testing/images/`. Place train images in 
   `training/images`.

  Insert the following files into the `datasets/kaggle` folder:

  * `image_labels_map.csv` (i.e. `train.csv`)
  * `sample_submission.csv`
  * `unicode_translation.csv`

2. Adjust configuration in  `networks/configuration/params_model_CenterNet.json`. Use the `evaluate` 
  parameters to print metrics. Detector and classifier have separate configurations. To print metrics 
  set `evaluate` to `true`, leaving the other parameters to false. 
  You can find a list of prepared JSON in [here](networks/configuration/demo). You may adjust the batch size via the
  `batch_size_predict` parameter.

3. Create a folder in `networks/experiments` with the following structure:
    ```
         experiments/
         ├── run_name
         ├── run_name_2/
         │   └── weights
         ├── run_name_3/
         │   └── weights
    ```

4. Download the desired weights from [GDrive](https://drive.google.com/drive/folders/1PruhKVboInMXX1dtJb-j1Af3EK8F6Xzy?usp=sharing). Place the detector weights inside `run_name_2/weigths/`
 and the classifier weights in `run_name_3/weights/`
 
5. Weights file are named as `weights.11-0.13.hdf5`. Here 11 is the epoch number. 
    - Set this number in 
`initial_epoch` parameter inside the JSON file for both models
    - Set the parameter `run_id` to the chosen '*run_name*'
    - Set `restore_weights` to `true` for both models.
    
6. Run program with `python3 __centernet__.py` 

### Train configuration
Use the same configuration, setting `train` parameter to true. `restore_weights` can be set to true, to 
continue training from previous weights. In this case training will restart from specified initial 
epoch. Otherwise set it to `false` and set `initial_epoch` to 0, to restart. To make new experiments
without overwriting existing weights, just change the `run_id` parameter: a new experiment folder will be created.



 
