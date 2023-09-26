
# IMPORTS
import mediapipe as mp
import cv2
import numpy as np
import time
from oscpy.client import OSCClient
import argparse

# arg parse
parser = argparse.ArgumentParser()
parser.add_argument("-sf", "--scaling_factor", type=float, help="Scale down size of window. Affects processing and video output. Range 1-10.", default = 1, choices = range(1, 11))
parser.add_argument("-off", "--screen_off", action = "store_false", help="Disable viewing video output by disabling cv2.imshow().")
parser.add_argument("-c", "--color", nargs = "+", type = int, help="Choose color in BGR format (B, G, R).", default = (20, 255, 20))
parser.add_argument("-t", "--timer", action = "store_true", help="Print inter-frame time.")
parser.add_argument("-dk", "--dark", action = "store_true", help="Turn video feed black after processing.")
parser.add_argument("-vd", "--video_capture", type=float, help="Float representing camera device connected to machine. 0 probably built in camera.", default = 0)
args = vars(parser.parse_args())
print(args)

scaling_factor = args["scaling_factor"]
screen_on = args["screen_off"]
color = tuple(args["color"])
timer = args["timer"]
dark = args["dark"]
video_capture = args["video_capture"]


# process arguments
scaling_factor = max(min(10, scaling_factor), 1) #clamp scaling factor to 1-10
try:
    color
except NameError:
    color = (20, 255, 20)    

if len(color) != 3:
    print("Please use (B, G, R) format. Continue using default value :(20, 255, 20)")
    color = (20, 255, 20) 

#Get time at start
time_old = time.perf_counter()

# setup osc connection
OSC_HOST ="127.0.0.1" #127.0.0.1 is for same computer
OSC_PORT = 8000
OSC_CLIENT = OSCClient(OSC_HOST, OSC_PORT)
OSC_PORT_WEK = 6448
OSC_CLIENT_WEK = OSCClient(OSC_HOST, OSC_PORT_WEK)

# media pipe objects
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

color_set = color
left_hand_score = 0
right_hand_score = 0

# get labels for hands

def get_label(index, hand, results):
    output = None
    
    if index == 0:
        label = results.multi_handedness[0].classification[0].label
        score = results.multi_handedness[0].classification[0].score
        text = '{} {}'.format(label, round(score, 2))
        coords = tuple(np.multiply(
                        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                        [cam_width,cam_height]).astype(int))
        
        if label == "Left":
            left_hand_score = score
        else:
            right_hand_score = score

        output = text, coords
        return output
    
    if index == 1:
        label = results.multi_handedness[1].classification[0].label
        score = results.multi_handedness[1].classification[0].score
        text = '{} {}'.format(label, round(score, 2))
        coords = tuple(np.multiply(
                        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                        [cam_width,cam_height]).astype(int))

        if label == "Left":
            left_hand_score = score
        else:
            right_hand_score = score

        output = text, coords
        return output
    

#making variables globally accessable
def bufferfunc():
    bufferfunc.dist_finger = np.array([0,0,0])
    bufferfunc.dist_finger2 = np.array([0,0,0])
    bufferfunc.dist_hands = np.array([0,0,0])
    bufferfunc.count_hands = 0
    bufferfunc.wek_left = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) #12 positions für 5 finger + wrist
    bufferfunc.wek_right = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) #12 positions für 5 finger + wrist #not used

bufferfunc()
        
# draw finger position
def draw_finger_position(image, results, joint_list):
    
    #BUFFER Variable
    buff = np.array([0,0])
    bufferfunc.count_hands = 0
     
    
    thumb_pos_0 = None
    thumb_pos_1 = None
    index_pos_0 = None
    index_pos_1 = None
    pinky_pos_0 = None
    pinky_pos_1 = None

    i = 0
    # Loop through hands and send OSC
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        
        if i == 0:
            hand_label = results.multi_handedness[0].classification[0].label
        if i == 1:
            hand_label = results.multi_handedness[1].classification[0].label
        
        thumb_pos = None
        index_pos = None
        pinky_pos = None

        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            a_depth = (hand.landmark[joint[0]].z)*(-1)*100
            if a_depth < 0 : a_depth = 0
            if a_depth > 255: a_depth = 255

            txt_a = str(round(a[0], 2)) + ", " + str(round(a[1], 2)) + ", " + str(round(a_depth, 2))
                
            cv2.circle(image, center = tuple(np.multiply(a, [cam_width, cam_height]).astype(int)), 
                        radius = int(a_depth * 2), color = color_set, thickness = 1*int(a_depth/3))
            
            if hand_label == "Left":
                hand_ind = "0"
            if hand_label == "Right":
                hand_ind = "1"

            joint_ind = joint_list.index(joint)

            string_path = '/'+'h'+str(hand_ind)+'/'+'f'+str(joint_ind)+'/x'
            ruta = string_path.encode()
            if (buff[0] != a[0]):
                OSC_CLIENT.send_message(ruta, [float(a[0])])
            string_path = '/'+'h'+str(hand_ind)+'/'+'f'+str(joint_ind)+'/y'
            ruta = string_path.encode()
            if (buff[1] != a[1]):
                OSC_CLIENT.send_message(ruta, [float(a[1])])
                
            
            buff = a

            if joint == [4,3,2]:
                thumb_pos = tuple(np.multiply(a, [cam_width, cam_height]).astype(int))
                
            if joint == [8,7,6]:
                index_pos = tuple(np.multiply(a, [cam_width, cam_height]).astype(int))
                
            if joint == [20,19,18]:
                pinky_pos = tuple(np.multiply(a, [cam_width, cam_height]).astype(int))
                

            cv2.putText(image, txt_a, tuple(np.multiply(a, [cam_width, cam_height]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA)
            
        # LEFT HAND, PUT TRIANGLE AND SEND VIA OSC            
        if hand_label == "Left":
            
            thumb_pos_0 = thumb_pos
            index_pos_0 = index_pos
            pinky_pos_0 = pinky_pos
            
            cv2.putText(image,  str(thumb_pos), (int(0.126*cam_width), int(0.14*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image, str(index_pos), (int(0.126*cam_width), int(0.16*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image,  str(pinky_pos), (int(0.126*cam_width), int(0.18*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

            dist_thumb_index = np.linalg.norm(np.array((thumb_pos_0[0], thumb_pos_0[1]))-np.array((index_pos_0[0], index_pos_0[1])))
            dist_index_pinkie = np.linalg.norm(np.array((index_pos_0[0], index_pos_0[1]))-np.array((pinky_pos_0[0], pinky_pos_0[1])))
            dist_pinkie_thumb = np.linalg.norm(np.array((pinky_pos_0[0], pinky_pos_0[1]))-np.array((thumb_pos_0[0], thumb_pos_0[1])))

            cv2.putText(image,   str(round(dist_thumb_index, 2)), (int(0.126*cam_width), int(0.20*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image,  str(round(dist_index_pinkie, 2)), (int(0.126*cam_width), int(0.22*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image,  str(round(dist_pinkie_thumb, 2)), (int(0.126*cam_width), int(0.24*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

            dist_array = np.array([dist_thumb_index,dist_index_pinkie,dist_pinkie_thumb])     

            if not ((np.array_equal(bufferfunc.dist_finger, dist_array))):
                string_path = '/'+str("h0")+'/'+'dist_ti'
                ruta = string_path.encode()
                OSC_CLIENT.send_message(ruta, [float(dist_thumb_index)])

                string_path = '/'+str("h0")+'/'+'dist_ip'
                ruta = string_path.encode()
                OSC_CLIENT.send_message(ruta, [float(dist_index_pinkie)])

                string_path = '/'+str("h0")+'/'+'dist_pt'
                ruta = string_path.encode()
                OSC_CLIENT.send_message(ruta, [float(dist_pinkie_thumb)])

            bufferfunc.dist_finger = dist_array

            bufferfunc.count_hands = bufferfunc.count_hands +1



            wrist = np.array([hand.landmark[0].x, hand.landmark[0].y]) 
            thumb = np.array([hand.landmark[4].x, hand.landmark[4].y]) 
            index = np.array([hand.landmark[8].x, hand.landmark[8].y]) 
            middle = np.array([hand.landmark[12].x, hand.landmark[12].y]) 
            ring = np.array([hand.landmark[16].x, hand.landmark[16].y]) 
            pinky = np.array([hand.landmark[20].x, hand.landmark[20].y]) 
            bufferfunc.wek_left  = np.concatenate((wrist, thumb, index, middle, ring, pinky))

        # RIGHT HAND, PUT TRIANGLE AND SEND VIA OSC
        if hand_label == "Right":
            
            thumb_pos_1 = thumb_pos
            index_pos_1 = index_pos
            pinky_pos_1 = pinky_pos

            cv2.putText(image,  str(thumb_pos), (int(0.186*cam_width), int(0.14*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image, str(index_pos), (int(0.186*cam_width), int(0.16*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image,  str(pinky_pos), (int(0.186*cam_width), int(0.18*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

            dist_thumb_index1 = np.linalg.norm(np.array((thumb_pos_1[0], thumb_pos_1[1]))-np.array((index_pos_1[0], index_pos_1[1])))
            dist_index_pinkie1 = np.linalg.norm(np.array((index_pos_1[0], index_pos_1[1]))-np.array((pinky_pos_1[0], pinky_pos_1[1])))
            dist_pinkie_thumb1 = np.linalg.norm(np.array((pinky_pos_1[0], pinky_pos_1[1]))-np.array((thumb_pos_1[0], thumb_pos_1[1])))

            cv2.putText(image, str(round(dist_thumb_index1, 2)), (int(0.186*cam_width), int(0.20*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image, str(round(dist_index_pinkie1, 2)), (int(0.186*cam_width), int(0.22*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
            cv2.putText(image, str(round(dist_pinkie_thumb1, 2)), (int(0.186*cam_width), int(0.24*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

            dist_array2 = np.array([dist_thumb_index1,dist_index_pinkie1,dist_pinkie_thumb1])     

            if not ((np.array_equal(bufferfunc.dist_finger2, dist_array2))):
                string_path = '/'+str("h1")+'/'+'dist_ti'
                ruta = string_path.encode()
                OSC_CLIENT.send_message(ruta, [float(dist_thumb_index1)])

                string_path = '/'+str("h1")+'/'+'dist_ip'
                ruta = string_path.encode()
                OSC_CLIENT.send_message(ruta, [float(dist_index_pinkie1)])

                string_path = '/'+str("h1")+'/'+'dist_pt'
                ruta = string_path.encode()
                OSC_CLIENT.send_message(ruta, [float(dist_pinkie_thumb1)])

            bufferfunc.dist_finger2 = dist_array2

            bufferfunc.count_hands = bufferfunc.count_hands +1


            #wrist = np.array([hand.landmark[0].x, hand.landmark[0].y]) 
            #thumb = np.array([hand.landmark[4].x, hand.landmark[4].y]) 
            #index = np.array([hand.landmark[8].x, hand.landmark[8].y]) 
            #middle = np.array([hand.landmark[12].x, hand.landmark[12].y]) 
            #ring = np.array([hand.landmark[16].x, hand.landmark[16].y]) 
            #pinky = np.array([hand.landmark[20].x, hand.landmark[20].y]) 
            #bufferfunc.wek_right  = np.concatenate((wrist, thumb, index, middle, ring, pinky))

        cv2.line(image, (thumb_pos), (index_pos), color_set, thickness = 2)
        cv2.line(image, (index_pos), (pinky_pos), color_set, thickness = 2)
        cv2.line(image, (pinky_pos), (thumb_pos), color_set, thickness = 2)

        i = i+1

    # Check for both using isinstance, to work around the "thumb_pos_0[i] -> NoneType not subscriptable" error that appears, when BOTH_HANDS happens before one of the hands 
    if (isinstance(thumb_pos_0, (str, list, tuple)))&(isinstance(thumb_pos_1, (str, list, tuple))):
        #print("both", )
       

        cv2.line(image, thumb_pos_0, thumb_pos_1, color_set, thickness = 2)
        cv2.line(image, index_pos_0, index_pos_1, color_set, thickness = 2)
        cv2.line(image, pinky_pos_0, pinky_pos_1, color_set, thickness = 2)

        thumb_0_x_y = np.array((thumb_pos_0[0], thumb_pos_0[1]))
        thumb_1_x_y = np.array((thumb_pos_1[0], thumb_pos_1[1]))
        index_0_x_y = np.array((index_pos_0[0], index_pos_0[1]))
        index_1_x_y = np.array((index_pos_1[0], index_pos_1[1]))
        pinky_0_x_y = np.array((pinky_pos_0[0], pinky_pos_0[1]))
        pinky_1_x_y = np.array((pinky_pos_1[0], pinky_pos_1[1]))

        dist_thumb = np.linalg.norm(thumb_0_x_y - thumb_1_x_y)
        dist_index = np.linalg.norm(index_0_x_y - index_1_x_y)
        dist_pinky = np.linalg.norm(pinky_0_x_y - pinky_1_x_y)

        middle_thumb = tuple(np.multiply(((thumb_0_x_y + thumb_1_x_y)/2), [1, 1]).astype(int))
        middle_index = tuple(np.multiply(((index_0_x_y + index_1_x_y)/2), [1, 1]).astype(int))
        middle_pinky = tuple(np.multiply(((pinky_0_x_y + pinky_1_x_y)/2), [1, 1]).astype(int))
        
        cv2.putText(image, str(dist_thumb), (middle_thumb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image,  str(dist_thumb), (int(0.126*cam_width), int(0.04*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, str( dist_index), (middle_index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image,  str(dist_index), (int(0.126*cam_width), int(0.06*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, str( dist_pinky), (middle_pinky), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image,  str(dist_pinky), (int(0.126*cam_width), int(0.08*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

        dist_array_hands = np.array([dist_thumb,dist_index,dist_pinky])     

        if not ((np.array_equal(bufferfunc.dist_finger, dist_array_hands))):
            string_path = '/'+str("both")+'/'+'tt'
            ruta = string_path.encode()
            OSC_CLIENT.send_message(ruta, [float(dist_thumb)])

            string_path = '/'+str("both")+'/'+'ii'
            ruta = string_path.encode()
            OSC_CLIENT.send_message(ruta, [float(dist_index)])

            string_path = '/'+str("both")+'/'+'pp'
            ruta = string_path.encode()
            OSC_CLIENT.send_message(ruta, [float(dist_pinky)])

        bufferfunc.dist_finger = dist_array


    # WEKINATOR INPUTS ____________________________________________
        
    ruta_wek = "/wek/inputs"
    #message as array
    message_wek = bufferfunc.wek_left # np.append(bufferfunc.wek_left, bufferfunc.wek_right)
    #print(message_wek)
    OSC_CLIENT_WEK.send_message(ruta_wek.encode(), message_wek)

    

    

    return image


# joint List 
joint_list = [[4,3,2], [8,7,6], [20,19,18]]

# Video Capture 
## this is where you choose your webcam. try 0, 1, etc. 
cap = cv2.VideoCapture(video_capture)


# camera parameters
cam_width  = cap.get(3)  # float `width`
cam_height = cap.get(4)  # float `height`

# camera parameters
print(cam_width, " ", cam_height)

# call bufferfunc to make variable accessible
bufferfunc()


cam_width, cam_height = cam_width/scaling_factor, cam_height/scaling_factor

with mp_hands.Hands(max_num_hands = 2, min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity = 1) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        if timer:
            time_now = time.perf_counter()
            print(time_now - time_old)
            time_old = time_now
        
        #resize
        frame = cv2.resize(frame, (int(cam_width), int(cam_height)), interpolation = cv2.INTER_AREA)

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if dark:
            image[:,:,:] = 0 
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=color_set, thickness=2, circle_radius=0),)
                
                
                
                # Render left or right detection
                if get_label(num, hand, results):
                    
                    
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, color_set, 2, cv2.LINE_AA)
                    if text[0] == "L":
                        cv2.putText(image, text[-4:],(int(0.126*cam_width), int(0.10*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
                    if text[0] == "R":
                        cv2.putText(image, text[-4:], (int(0.126*cam_width), int(0.12*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
                    

            # Draw position to image from joint list
            draw_finger_position(image, results, joint_list)


        # default text
        cv2.putText(image, "distance thumb: ", (int(0.026*cam_width), int(0.04*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "distance index: ", (int(0.026*cam_width), int(0.06*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "distance pinkie: ", (int(0.026*cam_width), int(0.08*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "Left " , (int(0.026*cam_width), int(0.10*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "Right " , (int(0.026*cam_width), int(0.12*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "position thumb: ", (int(0.026*cam_width), int(0.14*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "position index: ", (int(0.026*cam_width), int(0.16*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "position pinkie: ", (int(0.026*cam_width), int(0.18*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )  
        cv2.putText(image, "dist thumb - index: ", (int(0.026*cam_width), int(0.20*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "dist index - pinkie: " , (int(0.026*cam_width), int(0.22*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "dist pinkie - thumb: ", (int(0.026*cam_width), int(0.24*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

        cv2.putText(image, "change speed ", (int(0.57*cam_width), int(0.04*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "change sample ", (int(0.37*cam_width), int(0.04*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "hold left ASL ILY for 2 sec ", (int(0.54*cam_width), int(0.07*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "hold left ASL ILY for 2 sec ", (int(0.34*cam_width), int(0.07*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "CLOUD HANDS by TIM-TAREK GRUND ", (int(0.8*cam_width), int(0.92*cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.rectangle(image, (int(0.3*cam_width), int(0.0*cam_height)), (int(0.5*cam_width),int(0.1*cam_height)), color_set)
        cv2.rectangle(image, (int(0.7*cam_width), int(0.0*cam_height)), (int(0.5*cam_width),int(0.1*cam_height)), color_set)


        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)

        
        if screen_on:
            cv2.imshow('Hand Tracking', image)

        # Quit application by pressing 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)