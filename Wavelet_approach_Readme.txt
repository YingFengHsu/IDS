Step 1: Wavelet(SWT) decompose the dataset DoS-Test.csv and DoS-TrainVali.csv using wavelet.py. Also, isolate the labels using labels.py.
Step 2: Use the dataset and labels generated in step 1 to perform parameter tuning with Paratune.py. Parameter tuning is done using Optuna.
Step 3: Use the parameters generated in step 2 to obtain the test results of random forest + wavelet.
Step 4: Obtain the feature importance of each model.
Step 5: Use the parameters generated in step 2 and feature importance generated in step 4 to obtain the test results of random forest + wavelet + feature selection.

Step 6: Compare the test result obtained in steps 3 and 5. 
