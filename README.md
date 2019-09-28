# Machine-Learning-Project-on-Bitcoin
This is a Deep Learning course project to apply many different models on Bitcoin and corresponding factors to forecast Bitcoin daily return, and finally generate trading strategy.

############################### Goal ############################
The goal of this project is to build a price prediction model for bitcoin based on historical prices and other features.

############# Part I-Data Gathering and Analysis ################
PartI.ipynb
################## Part II-Data Pre-processing ##################
basic_io.py

For the modeling part, to maintain uniformity across projects, we will define the following time periods for train, test and validation
Train: Jan 1, 2010 - June 30, 2018
Validation: July 1, 2018 - Dec 31, 2018
Test: Jan 1, 2019 - June 30, 2019
At each date, you have to lookback 28 days and make a prediction for the next 7 days. For example for a prediction date of March 28, you have to look at historic data and features from March 1-28 (28 days), and use it to make a prediction for March 29-April 4 (7 days).

############### Part III-Build and Train Models #################
model.py
constants.py
Main_Model.ipynb

We will use mean absolute error (MAE) as our metric of choice. 
1. Build 2 simple benchmark predictions, and calculate MAE for the following on the validation set
2. Build and train a simple neural network by flattening the data, and using 2 (dense) layers and calculate the train and validation loss for each epoch
3. Build and train an RNN model with an LSTM layer and print the train and validation loss
4. Build and train an RNN model with a GRU layer and print the train and validation loss
5. Build and train an RNN model with a GRU layer and recurrent dropout and print the train and validation loss for each epoch
6. Add an additional GRU layer to (5) above with dropout and recurrent dropout and print the train and validation loss
7. (bonus) Build a classic time series model (ARIMA, or ARIMA with dynamic regression) and calculate train and validation loss [ ref ]
8. (bonus) Build an ensemble model that combines predictions from an RNN and an ARIMA model, and evaluate (using validation set) if they result in an improved performance.

Then Evaluate each of the models on the test set, calculate MAE and RMSE (root mean squared error)

################# Part IV-Trading Strategy ###################
Trading Strategy.ipynb
Construct a very simple strategy.
