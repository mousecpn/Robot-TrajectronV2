![logo](asset/RT_logo.png)


### Environment

 - numpy==1.24.4
 - tqdm==4.45.0
 - matplotlib==3.7.2
 - opencv-python==4.8.0.74
 - scikit-learn==0.22.1
 - scipy==1.10.1
 - seaborn==0.12.2
 - torch==2.0.0
 - CUDA==11.7
 - torchvision==0.15.1
 - pybullet


### Dataset

```
$ ./data_collection_dist.sh NUM_PROCS TOTAL_DATA DATA_DIR
```



### Train

```
$ python main.py --eval_every 10 --vis_every 10 --preprocess_workers 0 --batch_size 256 --log_dir experiments/RobotTrajectron/models --train_epochs 100 --conf config/config.json --data_path /path/to/dataset
```


### Test
```
$ python evalute.py --batch_size 256 --conf config/config_test.json --data_path /path/to/dataset --checkpoint /path/to/checkpoint
```


### Visualization
```
$ python visualization.py --conf config/config_test.json --data_path /path/to/dataset --checkpoint /path/to/checkpoint
```


### Shared Control
Shared control code will be available soon.

