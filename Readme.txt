•	Covid_Image_Classification.ipynb contains the code to generate the Xception_model_Full.h5 file which is used by app.py.
•	app.py is the flask server that will access Xception_model_Full.h5 and predict the probability of Covid-19 in the given CT scan.
•	Index.html is the frontend system where the users can upload their CT-scans and receive the results. 
•	The frontend page Index.html will be automatically rendered when the app.py server file is executed. The server file can be executed as follows:

	export FLASK_APP=flask_server.py
	flask run

•	Dataset used for training the classification model can be found at https://www.kaggle.com/azaemon/preprocessed-ct-scans-for-covid19 

•	requirements.txt has the required dependencies for executing the machine learning model. These packages can be installed using the pip install option.

•	The folder Sample_CT_scans contains Covid positive and negative samples to test the application.  
