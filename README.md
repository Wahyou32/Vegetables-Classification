# Vegetables Classification
This is one of my machine learning project, focused to classify vegetables based on image. This project is already can be accessed in REST API created using Fastapi. This project is created using Convolutional Neural Network (CNN). There are 15 classes in this project, they are:

1. Beans
2. Bitter Gourd
3. Bottle Gourd
4. Brinjal
5. Broccoli
6. Cabbage
7. Capsicum
8. Carrot
9. Cauliflower
10. Papaya
11. Potato
12. Cucumber
13. Pumpkin
14. Radish
15. Tomato

# Requirements
This project use an external library such as pandas, numpy, tensorflow, keras, zipfile, and fastapi (library to create a REST API for python).

# Datasets
The datasets i used was from https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

# How to Run 
Before running this project, you need to train and save the machine learning model by running the train.py using this command:

```
python train.py
```
Then, you have to wait untill training precess finish and the vegetable_model folder created. In this folder, your machine learning model will be saved.

After having a model, we need to run the api by the following command:
```
uvicorn app:app --reload
```

Then, access the REST API by accessing this url in your browser
127.0.0.1:8000/docs
