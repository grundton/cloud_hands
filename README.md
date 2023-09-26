# About
Cloud Hands is granular synthesizer instrument/installation using machine vision.
A python script making use of the [MediaPipe](https://developers.google.com/mediapipe) library estimates hand landmarks from live video input which are then used as parameters for a granular synthesis engine written in Pure Data.


# Installation 
Download the zip file or clone the repository

Before installing any dependencies, I recommend using a seperate virtual environment
```
conda create -n cloudhands python=3.9
```
To install the required libraries for cloud hands, simply open a terminal at the root directory and execute
```
pip install -r requirements.txt
```
# Getting started
Find some .wav or .aiff files you want to use and put them inside the samples folder.

# Usage
Follow these steps to start cloud hands.

### 1. Start cam2osc_grain.py from the terminal
Open a terminal window at the root directory.

First activate the correct virtual environment, for example ```cloudhands```.
```
conda activate cloudhands
```
Then execute the cam2osc.py script.
```
python python_script/Cam2Osc_grain.py
```
You might need to allow access to your camera.
You can close the application by pressing q inside the active video window or by interrupting the process inside the terminal.


### 2. Open cloud_hands.pd
2. a) activate the [s all_on]-toggle
2. b) turn up the black master volume vertical slider
2. c) activate the horizontal radio toggles (hradio) below the [r b1] and [r b2] receive objects by clicking into one of the available boxes (aka selecting a sample and a playback speed)

At this point you should be able to hear sound whenever your hand is detected on screen. You should be able to scroll through the selected sample by moving your hand/hands in a horizontal motion across the frame.

### 3. Open Wekinator
3. a) Open the Wekinator project from this path: WEK_sample_speed > WEK_sample_speed.wekproj 
3. b) Press the "Run button on the left side

At this point you should be able to switch between samples and playback speeds simply by holding the ASL sign for [I love you](https://www.lifeprint.com/asl101/images-signs/i_love_you.jpg) with your left hand over one of the respective boxes on top of the camera frame

# Customization
You can pass some arguments to the python script in order to customize the behavior. 

## Speed
In some cases, especially on older laptops, the processing of the images cannot keep up with the cameras framerate causing the whole system to lag and feel less responsive.
Changing some of the arguments can result in a more pleaseant experience.
For faster processing i recommend
1. Increasing the scaling factor. A scaling factor of 2 halves each dimension of the video feed before processing.
```
python python_script/Cam2Osc_grain.py -sf 2
```

2. Disabling visual feedback. For some applications, visual feedback is not necessary. In this case turn off the video rendering after processing.
```
python python_script/Cam2Osc_grain.py -off
```
You can terminate the application from the terminal, but not from the video window (since there is none).

## Debugging
In order to evaluate the speed benefits of certain configurations, you can print out the inter-frame time.
For example compare
```
python python_script/Cam2Osc_grain.py -t
```
against 
```
python python_script/Cam2Osc_grain.py -sf 2 -off -t
```

## Video Device
If you have multiple cameras hooked up to your machine, you can specify which one to use via the ```-vd``` argument. It takes an integer input to specify the camera it should access. To access the fourth camera, you can execute
```
python python_script/Cam2Osc_grain.py -vd 3
```

## Color
Customizing the color might be important in some applications. To change it, you can specify the color by supplying the color in BGR convention.
This one will turn every drawn line and text blue:
```
python python_script/Cam2Osc_grain.py -c 255 0 0 
```




