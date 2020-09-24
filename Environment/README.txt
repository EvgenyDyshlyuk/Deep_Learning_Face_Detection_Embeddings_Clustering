conda list --explicit > spec-file.txt

the file is saved in working directory

make sure that the txt file is saved as utf-8 (an option if you go to save as in notepad)


on the other computer:
conda create --name myenv --file spec-file.txt