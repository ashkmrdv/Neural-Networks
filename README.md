This is a neural network which was designed to categorise  and predict handwritten number digits.

Some of the key remarks of the model and the methodologies used are as follows


1. Packages used to run the program were pydot,numpy, keras, tensorflow, matplotlib. The program was run using anaconda environment.
2. I had to set the environment variable 'KMP_DUPLICATE_LIB_OK']='True' to avoid error while running it 
3. I have preferred CNNs over MLPs because of their tolerance to position 
4. To improve generalization and tolerance to new values, drop out regularization has been used
5. Further, more types of regulaization which has not been mentioned in the tutorial such as kernel, bias and activity regularization have also been used.
6. To make sure the model is tolerant in size, phase and angle variations, Data augmentation using the Cut mix augmentation type has been implemented for preprocessing the data
7. Upon running for the test use cases, the percentage correctly classified was about 98.63 of the test data
