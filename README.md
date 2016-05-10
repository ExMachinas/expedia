# expedia
Kaggle Project for Expedia Hotel Recommendations


# Initial Setting
```
$ git clone git@github.com:ExMachinas/expedia.git
$ cd expedia
$ mkdir data
$ cd data
```
Download test data files into data folder (see: https://www.kaggle.com/c/expedia-hotel-recommendations/data)
* test.csv.gz
* train.csv.gz
* destinations.csv.gz
* sample_submission.csv.gz

# Run
```
$ cd ..
$ python main.py
```

# Run in background with out.log
```
$ nohup python -u main.py >> out.log 2>&1 &
```

