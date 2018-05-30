# RealTime3DPoseTracker-OpenPose
Real time 3D pose tracking and Hand Gesture Recognition using OpenPose, Python Machine Learning Toolkits, Realsense and Kinect libraries. 

INSTALLATION STEPS: 
OpenPose and PyOpenPose
Machine: 4 GPU, GeForce GTX 1080
OS: Ubuntu 16.04

1) Clone the OpenPose repository: 
	"git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose"
2) Check the currently integrated OpenPose version from PyOpenPose from the link: https://	github.com/FORTH-ModelBasedTracker/PyOpenPose

3) Reset the OpenPose version to this commit by: 
    git reset --hard #version
4) Download and install CMake GUI: sudo apt-get install cmake-qt-gui

5) INSTALL CUDA 8: 
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb"
sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda

6) INSTALL CUDNN5.1: Run from the openpose root directory: 
	sudo ubuntu/install_cudnn.sh  . 

7) Run from the OpenPose root directory: 
	sudo ubuntu/install_caffe_and_openpose_if_cuda8.sh
8) OpenPose need to be built with OpenCV3, In order to correctly install it follow the instructions here: https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04

9) Caffe prerequisites: By default, OpenPose uses Caffe under the hood. If you have not used Caffe previously, install its dependencies by running:
     sudo bash ./ubuntu/install_cmake.sh
10) - Open CMake GUI and select the OpenPose directory as project source directory, and a non-existing or empty sub-directory (e.g., build) where the Makefile files will be generated. If build does not exist, it will ask you whether to create it. Press Yes.
    -Press the Configure button, keep the generator in Unix Makefile, and press Finish.
    -Change the OpenCV directory to OpenCV/build folder that you previously generated while installing OpenCV3. 
    -Press the Generate button and proceed to building the project. You can now close CMake.
10) Create an “install” folder under the OpenPose root directory. 
11) Open a terminal in the root directory and type: 
cd build/
make -j`nproc` (i.e. I use -j8)

12) Change the installation folder under build/cmake_install.cmake using gedit to the install folder that you previously created such as: 
set(CMAKE_INSTALL_PREFIX "/home/burcak/Desktop/openpose/install")
13) If  you get access errors, just change the permissions to the file as: chmod 777 “fileName”
14) Back to the terminal and : 
	- cd build/ 
and then: 
	- sudo make install
15) Copy the MODELS folder in OpenPose root directory to this INSTALL folder.  If  you get access errors, just change the permissions to the file as: chmod 777 “folderName”
16) Set an environment variable named OPENPOSE_ROOT pointing to the openpose INSTALL folder.
17) Install python3.5 and pip3. 
18) Change current python version to 3.5: 
	gedit ~/.bashrc
	alias python=’/usr/bin/python3.5’
	source ~/.bashrc

19) Clone the PyOpenPose repository using : 
    git clone https://github.com/FORTH-ModelBasedTracker/PyOpenPose.git
20) Open a terminal at the root directory of PyOpenPose and type: 
	mkdir build 
	cd build 
	cmake .. -DWITH_PYTHON3=True
	make 

21) Add the folder containing PyOpenPose.so to your PYTHONPATH such as adding a line to your bash profile: 
    export PYTHONPATH=/home/burcak/Desktop/PyOpenPose/build/PyOpenPose.lib
22) In order to check if its working properly, run the scripts folder for python examples using PyOpenPose.




RealSense and Kinect libraries integration: 

i) Realsense: 
1) Remove any connected RealSense cameras first. 
2) Follow the Realsense Ubuntu installation steps at:         
	https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
3) CAREFUL! Instead of "sudo apt-get install cmake3 libglfw3-dev" for installing glfw libraries, use the following commands, otherwise you’ll get version mismatch error(see issue https://github.com/IntelRealSense/librealsense/issues/1525 for details) :
	sudo apt-get update
	sudo apt-get install build-essential cmake git xorg-dev libglu1-mesa-dev
	git clone https://github.com/glfw/glfw.git /tmp/glfw
	cd /tmp/glfw
	git checkout latest
	cmake . -DBUILD_SHARED_LIBS=ON
	make
	sudo make install
	sudo ldconfig
	rm -rf /tmp/glfw

4) Install the Python Wrapper PyRealSense by following the instructions here and update your Python PATH variable accordingly: 
	https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
	export PYTHONPATH=$PYTHONPATH:/home/burcak/Desktop/librealsense/wrappers/python

5) Install PyCharm and start a project by running ./pycharm.sh in the PyCharm bin folder. OR create an application shortcut on the desktop or wherever you want to. 

6) For the codes using the visualization of the skeleton only, install pygame for python3: sudo pip3 install pygame, OpenGL: "sudo apt-get install python3-opengl" and don’t forget to upgrade it otherwise some functions might not work: "pip3 install --upgrade PyOpenGL"

7) Create a Project in PyCharm and click and expand Project INterpreter tab, select Python3 and inherit all the packages that are installed. Install the other required packages such as python numpy, sklearn, matplotlib, pandas, etc. 


ii) Kinect: 
1) Follow the installation steps:  https://github.com/r9y9/pylibfreenect2 and http://r9y9.github.io/pylibfreenect2/latest/installation.html
2) Don’t forget to add the pylibfreenect2 library to your PYTHONPATH and the other related installation paths, as an example, on my computer they are set as: 
    export LD_LIBRARY_PATH=”/home/burcak/Desktop/libfreenect2/build/lib”
    export LIBFREENECT2_INSTALL_PREFIX=”/home/burcak/freenect2”    


