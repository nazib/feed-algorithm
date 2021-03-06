# Whait is Feed Algorithm
This is the algorithm developed for Travelloapp.com. Travello is a travel based social network app. The app requires an intelligent way to provide new feeds for its users (see here: https://travelloapp.com/feed). The algorithm uses variational auto-encoder to learn feature coefficients and use those coefficients to rank the user's posts and present the feed items which relevent to particular app user. The REST api of this algorithm called everytime an user check his/her feed in the app. The algorithm is currently running behind the Travelloapp feed system in travello's own Google kubernets engine.     

# Quick start

## run with docker-compose (should work on any machine)

for simplicity this runs tensorflow and the python web server inside the same container, training will be only using CPU, so it will be slower than running natively on GPU

- install docker-compose (from Docker.app for macos)
```
docker-compose build web
docker-compose up web
```
- the web server will be on http://localhost:8080

### running tensorboard
```
docker-compose exec web bash

#### 
tensorboard --logdir=logdir --bind_all # --bind_all makes it listen on 0.0.0.0

```

### train new model with new data

- connect to datascience read replica with gcloud sql proxy
- run extract_feed_data script outside docker
```
MYSQL_USER=xxx MYSQL_PASS=xxx LIMIT=100000 ./extract_feed_data.sh Data/feed_data.tsv
```
- run process_data_train_model.py inside docker
```
docker-compose exec web bash

#### 
python process_data_train_model.py
```

## run locally (this crashes on macos due to tensorflow binary issue)
```
python -m venv venv # create venv (optional)
source venv/bin/activate # activate venv
pip install -r requirements.txt

# copy some base models and Data
./scripts/copy_base_data.sh

# run flask dev server
export FLASK_APP=app.py 
export FLASK_ENV=development
python -m flask run --host=0.0.0.0 --port 8080

```

---

## File List and their purposes

1. model.py --- Non-linear model definition file.
2. train_var.py --- Training of Non-linear model.
3. test.py --- testing of non-linear model.
4. losses.py --- loss functions for training of the non-linear model
5. Ranking.py  --- linear model training and testing file (needs to be changed)
6. preprocess_data.py --- preprocessing raw data obtained from databases and making data applicable for ML algorithms
7. process_data.py --- similar to preprocess_data.py file
8. rank_logicks.py --- contains the Global and Local Ranking logics for rank score generation
9. FeedRankApi (Folder) --- contains REST api for the feed ranking (Django version)
10. Flask-delp (Folder) --- contains REST api for the feed ranking (Flask version)
11. web-api.py Flask REST api demmo for LinearModel.py

---

## Training Linear algorithm
 The LinearModel.py file contains linear feed ranking model. It contains a class name LinearModel that warps python native linear regression models. In this implementation, following liear models are warped into this class:

 1. Lesso
 2. Linear Regression
 3. Ridge Regression
 4. Random Forrest

 From detail analysis, Random Forrest is found best for travello feed data. To train a selected model, follow the example code shown below:
 ```python

    obj = LinearModel("RandomForrest", 0.04) ### Declared the object
    obj.prepare_feed_data(FileName,Training_ratio,Label_name) ## Loading and pre-processing feed data file
    obj.fit() ### Training the model
```    
Each of the argument is explained below:

1. FileName = the full directory and filename of the csv file that contains feed data
2. Training_ratio = is the parcent of feeddata is to be used for training the model (80% is used in our case)
3. Label_name = any arbitrary column in the feeddata csv file should be used as the labal data for the training

To get the rank score of a particular feed-post, the feed-post features should be given as input data like below:

For obtaining Global Rank
```python
   obj.Rank(data) ### Here data is numpy array with 7 element 
```
The elemets of the in the input array ("data" in this case is 1D array), following order must be followed:
```python

data[0] = Comments
data[1] = Age
data[2] = PostLength
data[3] = Hash
data[4] = Latitude
data[5] = Longitude
data[6] = Url
```
Since the model is trained using 'Like' column as label in this case, the rank calculation considers other features except the like feature. The rank score calculated here is Global rank score. To obtain the ranks of a bunch of feed posts, the BulkRank function is to be used. In this case, the input data to the function must be 2D array. The columns of the array is as before (shown in the previuous code block) and each row of the data array is an instance of feedpost.    

## Training NonLinear Model
The python file NonLinearModel.py contains the non-linear feedrank model. There are two other file named train_vae.py and test.py, each of which is the basic nonliner algorithm developed initially. The model.py contains the deep-learning model and the NonLinearModel.py file defines a wraping class of that model. The training of the non-linear model is very similar to that of linear model.

```python    
obj = NonLinearModel()
#### Train ####
obj.fit("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/","AllFeedData.csv")

```
The fit() function takes data directory and feeddata file name as argument and start training the model. After training, the trained non-linear model is saved in a folder named "logs". Each time a new model is trained, the model is saved inside the subdirectory of the "log" directory using data and time as the name of the folder. Rank calculation after training is also similar to linear models. 
``` python

data = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv")
cols = data.columns 
values = data[cols[3:]].values
glb_ranks = obj.predict(values)
```
Here, to get the global rank scores for feed-data of the csv file "AllFeedData.csv" predict function is used. The data is read in python dataframe and provided as argument to the predict function.
The nonlinear model can also be trained by running python command as follows:

```python
python train_vae.py
```
A non-linear model will be saved in the subdiretory of "logs" with command execution data and time. 
