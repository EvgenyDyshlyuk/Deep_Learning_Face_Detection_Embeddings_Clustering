# Face Detection, Allignment, Embeddings, Clustering
(This is a personal ML project for Face Recognition from home archive photos)

### About
- Here is a very nice review of the current (2020.08) state of Deep Learning in face recognition: "Deep Face Recognition: A Survey" https://arxiv.org/abs/1804.06655
- In this notebook we use pretrained MTCNN (multi-task convolutional neural network) to detect faces on photos and find landmarks (eyes, nose, mouth) on faces.
- Pretrained FaceNet NN is used to generate face embeddings (128 point vector encoding discriminative features of faces as learned by NN).
- Both NN are based on FaceNet architecture - google development, which won prestigeous ImageNet competition in 2015 surpassing human performance in face recognition task. 

### Environment, libraries
- In order to get to the results a number of libraries have to be imported (please check the Environment folder for help on how to setup the environment).
- Libraries and helper functions are in utils.py, which is run at the beginning of this notebook.

### Notebook content:
0. Correct jpg files metadata
1. MTCNN (detect faces on photos)
2. Filter results (remove bad quality photos)
3. FaceNet embeddings (obtain 128-dim vector/embedding representing faces features numerically with FaceNet)
4. t-SNE: 2-D representation of clustering similar photos together and separating different photos further from each other.
