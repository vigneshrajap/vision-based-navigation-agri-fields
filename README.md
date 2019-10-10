# Vision-based-navigation-agri-fields

### Prerequisites

* Keras 2.0
* opencv for python
* Theano / Tensorflow / CNTK

lane_detection_segnet.py - Script for training and prediction. Some improvements like adding arguments for input, output images and folder to save the weights need to be done.

### Steps for annotating the images and preparing the agricultural field dataset for navigation:

	1) First of all, check the size of the input images from each datasets (covering all the variances throughout the cropping season). For instance, we had 848*480 or 640*360 image resolution in various cases. No matter what the size of the input images, all the images should have the SAME SIZE  (again SAME SIZE!!!) for using deep learning training and validations.

	2) If there are several datasets, then only a set of images from each dataset is needed for training. But the chosen images should able to cover all the generalizations of the environment so that it can learn the variance instead of memorizing it. In this case, we split each dataset into three stages in which two will be used for training and remaining one will be in testing as per the figure below. Make one folder for training and another for testing.

	3) Crop the input RGB images to same size (In the given case, every image is set to 640*320). There is also a reason that keras base models like vgg_segnet or resnet50_segnet usually require the image height and image width to be multiple of 32. And there is a special case, Google's mobilenet_segnet base model requires the image height and width to be 224*224. Any cropping techniques using third party will lead to losing the pixel information due to interpolation, therefore script "crop_images.py" is added which takes in the image as array and stores a subset of the image array into another output array which stores as image. In this way, no information is lost during cropping.

	4) Once the cropping is done, the images are ready for annotations. Here annotations are done using "labelme" which is attached here. The different objects in the input image will be assigned an unique class by creating a polygon (or other given basic shapes) around the object area. In this case, we have background, Crops and Lanes as classes. More can be added in the future. Once annotating is done, it will be saved as json file format. (Note: Manual annotations for large set of images are very cumbersome and time taking process which will be replaced by automatic annotations soon)

	5) Next step would be utilizing the JSON files and convert them into semantic annotated images has a set of steps. "labelme" has a provision to convert the JSON into VOC format which further gives more details about annotations as given by this link. (https://github.com/wkentaro/labelme/tree/master/examples/semantic_segmentation)	
	# It generates:
	#   - data_dataset_voc/JPEGImages
	#   - data_dataset_voc/SegmentationClass
	#   - data_dataset_voc/SegmentationClassVisualization
	./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt

        6) The SegmentationClass folder contains .npy files which has the information in the form of array that we want for the keras segmentation as annotated images. The script "annotated_images_from_npy.py" is added which gives the final annotated images and it has to be stored in annotations dataset folder. Now the dataset is ready for the training with different segmentation and base models.

        7) In order to visualize the ground truth, use "visualize_dataset.py" that takes in final annotated images and stores the images in same color pattern as training images. Therefore it is useful for better understanding.

Frogn_Dataset - Annotated Images from Frogn Fields

		Similar to training the dataset in https://github.com/divamgupta/image-segmentation-keras. You need to make two folders

    		Images Folder - For all the training images
    		Annotations Folder - For the corresponding ground truth segmentation images

		The filenames of the annotation images should be same as the filenames of the RGB images.
		The size of the annotation image for the corresponding RGB image should be same.


camera_data_collection - Package for collecting camera data from the real fields.

                         Supported Cameras:
                         - FLIR and Basler Cameras.
                         - Realsense D435 Camera.


GNSS_waypoint_navigation - Package for initiating GNSS RTK setup along with IMU and robot odometry. ROS ekf robot loclaization package is used in this package.

                          Supported Cameras:
                          - Septentrio Altus NR3.
                          - Xsens IMU Mti-710.

labelme - ROS package for manually annotating the RGB images using different classes, created by https://github.com/wkentaro/labelme.

For running the realsense camera D435, we stick to specific librealsense library, realsense-ros package and it's dependency ddynamic_reconfigure (soon it will be removed) are used as stable working versions.
- librealsense.2.25 - Intel realsense SDK for running the realsense D435 camera, and realsense2_camera ROS package will be additionally required in the ROS side of things.
- realsense_ros - ROS package containing camera launch file and description package for static tf and xacro file.
- ddynamic_reconfigure is a dependency by realsense_ros pacakge which will be removed by official Intel ROS maintainer soon. 

lane_detection_segnet.ipynb - file for training the annotated images using keras segmentation made by https://github.com/divamgupta/image-segmentation-keras. Can able to run the file in google colab if the training dataset is available.
