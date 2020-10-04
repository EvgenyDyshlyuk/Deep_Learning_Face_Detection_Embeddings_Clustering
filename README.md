# Face Detection, Allignment, Embeddings, Clustering
(This is a personal ML project for Face Recognition from home archive photos)

### About
- In this notebook pretrained MTCNN (multi-task convolutional neural network, 2016 original article https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) is used to detect faces on photos and find landmarks (eyes, nose, mouth) on faces.
- Pretrained FaceNet NN (2015 original article: https://arxiv.org/abs/1503.03832) is used to generate face embeddings (128 point vector encoding discriminative facial features).
- Both NN are based on FaceNet architecture - google development, which won prestigeous ImageNet competition in 2015 surpassing human performance in face recognition task.
- Current (2020.08) state of Deep Learning in face recognition is outlined in: "Deep Face Recognition: A Survey" (2020 original article: https://arxiv.org/abs/1804.06655).

### Environment, libraries
- In order to get to the results a number of libraries have to be imported (please check the Environment folder for help on how to setup the environment).
- Libraries and helper functions are in utils.py, which is run at the beginning of this notebook.

### Notebook content:
0. Correct jpg files metadata
1. MTCNN (detect faces on photos)
2. Filter results (remove bad quality photos)
3. FaceNet embeddings (obtain 128-dim vector/embedding representing faces features numerically with FaceNet)
4. t-SNE: 2-D representation of clustering similar photos together and separating different photos further from each other.
![**t-SNE representation for my family](https://raw.githubusercontent.com/EvgenyDyshlyuk/Face_detection_embeddings_clustering/master/Test/all.png)
