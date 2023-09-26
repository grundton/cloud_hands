# grain
# grain_2

Workflow:

### 1. Open Cam2Osc_grain.py
1. a) in line 329: change your input device 
1. b) run the code

At this point a window with live camera feed should open. If there is a hand within the frame, green positional markers should appear at the correct positions.

### 2. Open cloud_hands.pd
2. a) activate the [s all_on]-toggle
2. b) turn up the black master volume vertical slider
2. c) activate the horizontal radio toggles (hradio) below the [r b1] and [r b2] receive objects by clicking into one of the available boxes (aka selecting a sample and a playback speed)

At this point you should be able to hear sound whenever your hand is detected on screen. You should be able to scroll through the selected sample by moving your hand/hands in a horizontal motion across the frame.

### 3. Open Wekinator
3. a) Open the Wekinator project from this path: WEK_sample_speed > WEK_sample_speed.wekproj 
3. b) Under "Models", select the number box to the right of "outputs-1 (v2)" and change it to 3
3. c) Press the "Run button on the left side

At this point you should be able to switch between samples and playback speeds simply by holding the ASL sign for [I love you](https://www.lifeprint.com/asl101/images-signs/i_love_you.jpg) with your left hand over one of the respective boxes on top of the camera frame

# cloud_hands
