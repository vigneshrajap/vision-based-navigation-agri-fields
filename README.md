# Vision-based-navigation-agri-fields


camera_data_collection - Package for collecting camera data from the real fields. 
                         
                         Supported Cameras: 
                         - FLIR and Basler Cameras.
                         - Realsense D435 Camera.


labelme - ROS package for manually annotating the RGB images using different classes, created by https://github.com/wkentaro/labelme.


librealsense.2.25 - Intel realsense SDK for running the realsense D435 camera, and realsense2_camera ROS package will be additionally required in the ROS side of things.

lane_detection_segnet.ipynb - file for training the annotated images using keras segmentation made by https://github.com/divamgupta/image-segmentation-keras. Can able to run the file in google colab if the training dataset is available.




