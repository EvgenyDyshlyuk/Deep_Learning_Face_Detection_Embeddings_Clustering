# Face Detection, Allignment, Embeddings, Clustering
(This is a personal ML project for Face Recognition from home archive photos)

## About the Project

- In this project MTCNN (multi-task convolutional neural network, 2016 original article https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) is used to detect faces on photos and find landmarks: positions of eyes, nose, and mouth (marked in greeen on the figure).

![Landmarks preview](https://raw.githubusercontent.com/EvgenyDyshlyuk/DeepLearning_face_detection_embeddings_clustering/master/Figures/Landmarks.png)

- FaceNet NN (2015 original article: https://arxiv.org/abs/1503.03832) is used to generate face embeddings (128 point vector facial features encodings learned by FaceNet)

- Embeddings are further used to cluster photos to groups corresponding to different people (below), using t-SNE - unsupervised clustering procedure.
![**t-SNE representation for my photo archive](https://raw.githubusercontent.com/EvgenyDyshlyuk/DeepLearning_face_detection_embeddings_clustering/master/Figures/tSNE_all.png)

- Current (2020.08) state of Deep Learning in face recognition is outlined in: "Deep Face Recognition: A Survey" (2020 original article: https://arxiv.org/abs/1804.06655).

## Notebook Content:
1. **Correct jpg files metadata** (correct creation date if it is wrong)
2. **MTCNN** (detect faces on photos, find landmarks)
3. **Filter results** (remove "bad quality" facial images)
4. **Get face embeddings** (obtain 128-dim vector/embedding representing faces features numerically with FaceNet)
5. **Cluster** (t-SNE: 2-D representation of clustering similar photos together and separating different photos further from each other).

## Environment
In order to get to the results a number of libraries have to be imported (please check the Environment folder for help on how to setup the environment automatically).
- Some standard libraries:
  - conda install pandas
  - conda install -c conda-forge scikit-learn
  - conda install -c conda-forge matplotlib
  - conda install -c anaconda seaborn
  - conda install -c conda-forge tqdm
- Some libraries for work with images:
  - conda install -c conda-forge piexif
  - conda install -c conda-forge opencv
  - conda install -c anaconda pillow
- MTCNN installation:
  - conda install -c conda-forge mtcnn/pip3 install mtcnn
- Keras installation:
  - conda install -c conda-forge keras

## Libraries and Helper Functions
Libraries and helper functions are located in utils.py, which is run at the beginning of the notebook.