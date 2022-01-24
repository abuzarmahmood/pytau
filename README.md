  ___        _   _ _            
 / _ \ _   _| |_| (_)_ __   ___ 
| | | | | | | __| | | '_ \ / _ \
| |_| | |_| | |_| | | | | |  __/
 \___/ \__,_|\__|_|_|_| |_|\___|
==================================================
                                
- Perform inter-region transition correlations on models fit to each taste 
    individually

 ____        _           ___            
|  _ \  __ _| |_ __ _   / _ \ _ __ __ _ 
| | | |/ _` | __/ _` | | | | | '__/ _` |
| |_| | (_| | || (_| | | |_| | | | (_| |
|____/ \__,_|\__\__,_|  \___/|_|  \__, |
                                  |___/ 
==================================================
- Models stored in CENTRAL LOCATION and accessed by indexing an info file,
    which also contains model parameters and metadata
- Metadata also stored in model file to be able to recreate info file in case
    something happens

Database for models fit:
    - Columns : 
        -   Model save path
        -   Animal Name
        -   Session date
        -   Taste Name
        -   Region Name
        -   Experiment name (user-given name to separate batches of fits)
        -   Fit Date

        -   Model parameters:
            - Model Type (See below) 
            - Data Type (Shuffled, simulated, actual)
            - States Num
            - Fit steps
            - Time lims
            - Bin Width

Data stored in models:
    - Model
    - Approx
    - Lambda
    - Tau
    - Data used to fit model
    - Raw data (before preprocessing to feed to model)
    - Model and data details/parameters (as above)

- Considering we might want to compare different model types:
    - Unconstrained sequantial
    - Baised transition priors, sequential
    - Hard padding between transitions, sequential
    - Joint region model
    We need to have a standardized pipeline for fitting and retrieving these models

 ____  _            _ _            
|  _ \(_)_ __   ___| (_)_ __   ___ 
| |_) | | '_ \ / _ \ | | '_ \ / _ \
|  __/| | |_) |  __/ | | | | |  __/
|_|   |_| .__/ \___|_|_|_| |_|\___|
        |_|                        
==================================================

-- Filelist:

1) Model file
    - Invoked to generate model and perform inference
    - Currently <<poisson_all_tastes_changepoint_model.py>>
    - Should be agnostic to, and ignorant of what type of data is being fed.
        Should only fit model and return output
    - These should be functions

    - Input:
        1) Processed spike trains 
        2) Model parameters (model type, states, fit steps)
    - Output:
        1) Model
        2) Approx
        3) Lambda
        4) Tau
        5) Data used to fit model

2) Data pre-processing code:
    - Currently <<changepoint_fit.py>>
    - Should handle different data "types" (shuffle, simulated, actual)
    - These should be functions

    - Input:
        1) Raw spike trains
        2) Spike train parameters (time lims, binning)
        3) Desired data type
    - Output:
        1) Processed spike trains

3) I/O helper code
    - Takes data path and model parameters
        - If model has not been fit, runs fit, else pulls model from memory
    - This should be a "data_handler" class that can load, process, and
        return the data, write data to file, and write an entry in the database
    - The model creation and model fit functions can be imported as methods to
        streamline fitting
    Operations:
        - Loads data from HDF5, preprocesses, and feeds to modeling code
        - Collects outputs from modeling code, and combines with appropriate metadata
        - Writes model file to appropriate directory
        - Row with fit data appended to central dataframe

4) Run script
    - Script to iterate over data
    - Input:
        1) List of data to iterate over
