If you are using conda:

You can use spec-file.txt to install a working environment by placing the file in the working directory and running:
conda create --name myenv --file spec-file.txt
"myenv" needs to be changed to prefered name for this new environment.

The file was created by running:
conda list --explicit > spec-file.txt
Notes: The file is saved in the working directory. Make sure that the txt file is saved as utf-8 (an option if you go to save-as in a notepad), otherwize conda might fail to install new environment from this file.



If you are using pip:

Create new environment and run:
pip install -r requirements.txt
The file is created by running:
pip freeze > requirements.txt

