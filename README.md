# DRNDEF
> Drift Reduced Navigtion with Deep Explainable Features

### Developer Checklist
- This code is tested for Ubuntu 20.04 (and Pop_OS 20.04) with ROS Noetic and CARLA 0.9.11
- Follow [pep-8 guidelines](https://www.python.org/dev/peps/pep-0008/)
- Use VS-Code autoepep8 tool. Already included in the python package. Enable by *ctrl+shift+i*
- Everything shall be imported from the `root` path. To run a particular file as a script, do: `python -m module.file`

## Repository Structure

The repository contains three main modules: `nn`, `util`, and `scene`. `nn` contains the model architecture and training scripts. `util` contains basic utility scripts for range image creation, ackermann control, etc. `scene` contains scene configuration files required for inference.

## Training New Models

To train a new model, you can run

```
python -m nn.train_drift --filename path/to/filenames.csv  --epochs 500 --learning_rate 0.00001 --model_dir trained_models/contrastive/jan21/ --batch_size 4
```

## Inference

To run inference on CARLA Simulator, do the following `aloam_velodyne` is required to run in the background for any of the three inference options. To run `aloam_velodyne`, clone [this](https://github.com/ss26/A-LOAM) repository and `catkin_make` at the repository's root to build, as you would with any ROS workspace.

```
roslaunch aloam_velodyne aloam_carla.launch
```

You can run any **ONE** of the following commands. 

1. The following command will automatically control and drive the car:

    ```
    python -m nn.carla_inference --model_path path/to/trained/model --auto_drive
    ```
2. The following commands will enable manual driving. The first command enables joystick driving, the second is calling inference with manual driving. 
    ```
    python -m util.joy

    ```
    ```
    python -m nn.carla_inference --model_path path/to/trained/model --drive
    ```
3. If you just want to test if a newly trained model is working properly, you can do:
    ```
    python -m nn.carla_inference --model_path path/to/trained/model --prototype
    ```

## Dataset Request

For the lidar dataset and pretrained models, please drop a mail with a subject prefix `[DRNDEF-Data]` at: mohd.omama@research.iiit.ac.in. For example,
- [DRNDEF-Data] Requesting LIDAR dataset for testing
- [DRNDEF-Data] Requesting pretrained models for testing
