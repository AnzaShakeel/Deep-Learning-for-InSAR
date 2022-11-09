# Deep-Learning-for-InSAR
Detection of tectonic and volcanic deformation as anomalies in InSAR: deep-learning tailored to differential data
		                                             Read me 
-----------------------------------------------------------------------------------------------------------------------------------------------------------
		How to run temporal self attention for InSAR data
			- Required instllations:
				Python == 3.8
				Numpy
				Tensorflow-gpu == 2.6
				Keras == 2.6
				Other modules will installed with keras and tensorflow like hfpy etc
				All other modules that are called in the code can be installed using the command: pip install module name 
			- Input Details
				A text file that contains paths for each interferogram
				Data in the text file is stored in sequential order, i.e. each batch of 26 interferograms in temporal order and so on till time T
				Consecutive data batches contain overlapping interferograms (Read paper for further clarification)
				All interferograms are stored '.mat' file format in the folder in parent directory
			- Running the script
				Open jupyter notebook in the parent directory
				Select and open Test_Self_attention_with_InSAR.ipynb (which is the main file) 
				This file is importing 4 .py files containing functions that are called in the main notebook
				ALADDIn_model.py ---> contains functions that define the netwrok architecture of Bi-deep model of AlADDIn
				lambda_layers.py ---> contains functions of customized layers that are part of the network architecture
				generate_data.py ---> contains functions to read and load data from the text file and organize it in executable format
				process_output.py ---> contains functions that save the input and output interferograms and epoch time-series in .png format in separate folders                                  for each data batch 							
-----------------------------------------------------------------------------------------------------------------------------------------------------------
