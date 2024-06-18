import jetson.inference
import jetson.utils
import math

import argparse
import sys

parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# set font to print text on screen
font = jetson.utils.cudaFont(size=40)
font2 = jetson.utils.cudaFont(size=49)

def draw_flags():
    left_wrist_idx = pose.FindKeypoint(9)
    left_elbow_idx = pose.FindKeypoint(7)
    left_elbow = pose.Keypoints[left_elbow_idx]
    left_wrist = pose.Keypoints[left_wrist_idx]
    left_diff_el_wr_x = left_wrist.x - left_elbow.x
    left_diff_el_wr_y = left_wrist.y - left_elbow.y

    right_wrist_idx = pose.FindKeypoint(10)
    right_elbow_idx = pose.FindKeypoint(8)
    right_elbow = pose.Keypoints[right_elbow_idx]
    right_wrist = pose.Keypoints[right_wrist_idx]
    right_diff_el_wr_x = right_wrist.x - right_elbow.x
    right_diff_el_wr_y = right_wrist.y - right_elbow.y
    #Flags
    jetson.utils.cudaDrawLine(img, (left_wrist.x + left_diff_el_wr_x, left_diff_el_wr_y + left_wrist.y + 20), (left_wrist.x + 2 * left_diff_el_wr_x , 2 * left_diff_el_wr_y + left_wrist.y + 20), (255, 0, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (right_wrist.x + right_diff_el_wr_x, right_wrist.y + right_diff_el_wr_y + 20), (right_wrist.x + 2 * right_diff_el_wr_x, 2 * right_diff_el_wr_y + right_wrist.y + 20), (255, 255, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (left_wrist.x + left_diff_el_wr_x, left_diff_el_wr_y + left_wrist.y + 35), (left_wrist.x + 2 * left_diff_el_wr_x , 2 * left_diff_el_wr_y + left_wrist.y + 35), (255, 0, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (right_wrist.x + right_diff_el_wr_x, right_wrist.y + right_diff_el_wr_y + 35), (right_wrist.x + 2 * right_diff_el_wr_x, 2 * right_diff_el_wr_y + right_wrist.y + 35), (255, 255, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (left_wrist.x + left_diff_el_wr_x, left_diff_el_wr_y + left_wrist.y + 50), (left_wrist.x + 2 * left_diff_el_wr_x , 2 * left_diff_el_wr_y + left_wrist.y + 50), (255, 0, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (right_wrist.x + right_diff_el_wr_x, right_wrist.y + right_diff_el_wr_y + 50), (right_wrist.x + 2 * right_diff_el_wr_x, 2 * right_diff_el_wr_y + right_wrist.y + 50), (255, 255, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (left_wrist.x + left_diff_el_wr_x, left_diff_el_wr_y + left_wrist.y + 65), (left_wrist.x + 2 * left_diff_el_wr_x , 2 * left_diff_el_wr_y + left_wrist.y + 65), (255, 0, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (right_wrist.x + right_diff_el_wr_x, right_wrist.y + right_diff_el_wr_y + 65), (right_wrist.x + 2 * right_diff_el_wr_x, 2 * right_diff_el_wr_y + right_wrist.y + 65), (255, 255, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (left_wrist.x + left_diff_el_wr_x, left_diff_el_wr_y + left_wrist.y + 20), (left_wrist.x + left_diff_el_wr_x , left_diff_el_wr_y + left_wrist.y + 65), (255, 0, 0, 255), 15)
    jetson.utils.cudaDrawLine(img, (right_wrist.x + right_diff_el_wr_x, right_wrist.y + right_diff_el_wr_y + 20), (right_wrist.x + right_diff_el_wr_x, right_diff_el_wr_y + right_wrist.y + 65), (255, 255, 0, 255), 15)

    #Stick
    jetson.utils.cudaDrawLine(img, (left_wrist.x, left_wrist.y), (left_wrist.x + 2.2 * left_diff_el_wr_x, 2.2 * left_diff_el_wr_y + left_wrist.y), (0, 0, 0, 255), 5)
    jetson.utils.cudaDrawLine(img, (right_wrist.x, right_wrist.y), (right_wrist.x + 2.2 * right_diff_el_wr_x, 2.2 * right_diff_el_wr_y + right_wrist.y), (0, 0, 0, 255), 5)

def is_gesture(pose):
    left_wrist_idx = pose.FindKeypoint(9)
    left_elbow_idx = pose.FindKeypoint(7)
    left_shoulder_idx = pose.FindKeypoint(5)

    right_wrist_idx = pose.FindKeypoint(10)
    right_elbow_idx = pose.FindKeypoint(8)
    right_shoulder_idx = pose.FindKeypoint(6)

    if left_shoulder_idx < 0 or left_elbow_idx < 0 or left_wrist_idx < 0 or right_shoulder_idx < 0 or right_elbow_idx < 0 or right_wrist_idx < 0:
        return(False)
    
    left_elbow = pose.Keypoints[left_elbow_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]
    left_wrist = pose.Keypoints[left_wrist_idx]

    left_diff_sh_el_x = left_shoulder.x - left_elbow.x
    left_diff_sh_el_y = left_shoulder.y - left_elbow.y

    left_diff_el_wr_x = left_elbow.x - left_wrist.x 
    left_diff_el_wr_y = left_elbow.y - left_wrist.y 

    if left_diff_el_wr_x != 0:
        k = left_diff_sh_el_x / left_diff_el_wr_x
        k = abs(k)
    else: 
        k = 1

    new_left_diff_el_wr_x = left_diff_el_wr_x * k
    new_left_diff_el_wr_y = left_diff_el_wr_y * k

    left_com_y = left_diff_sh_el_y - new_left_diff_el_wr_y

   
    if -200 < left_com_y < 200:
        left = True
    else:
        left = False

    #jetson.utils.cudaDrawRect(img, (left_shoulder.x - 150,left_shoulder.y - 150,left_shoulder.x + 150,left_shoulder.y + 150), (0,255,0,100))
    #jetson.utils.cudaDrawCircle(img, (new_left_diff_el_wr_x + left_elbow.x,new_left_diff_el_wr_y + left_elbow.y), 10, (255,0,255,255))


    right_elbow = pose.Keypoints[right_elbow_idx]
    right_shoulder = pose.Keypoints[right_shoulder_idx]
    right_wrist = pose.Keypoints[right_wrist_idx]

    right_diff_sh_el_x = right_shoulder.x - right_elbow.x
    right_diff_sh_el_y = right_shoulder.y - right_elbow.y

    right_diff_el_wr_x = right_elbow.x - right_wrist.x 
    right_diff_el_wr_y = right_elbow.y - right_wrist.y 

    if right_diff_el_wr_x != 0:
        k = right_diff_sh_el_x / right_diff_el_wr_x
        k = abs(k)
    else: 
        k = 1
    new_right_diff_el_wr_x = right_diff_el_wr_x * k
    new_right_diff_el_wr_y = right_diff_el_wr_y * k

    right_com_y = right_diff_sh_el_y - new_right_diff_el_wr_y

   
    if -200 < right_com_y < 200:
        right = True
    else:
        right = False

    #jetson.utils.cudaDrawRect(img, (right_shoulder.x - 150,right_shoulder.y - 150,right_shoulder.x + 150,right_shoulder.y + 150), (0,255,0,100))
    #jetson.utils.cudaDrawCircle(img, (new_right_diff_el_wr_x + right_elbow.x,new_right_diff_el_wr_y + right_elbow.y), 10, (255,0,255,255))

    if right and left:
        return(True)

def armsAngles(pose):
    left_wrist_idx = pose.FindKeypoint(9)
    left_shoulder_idx = pose.FindKeypoint(5)

    right_wrist_idx = pose.FindKeypoint(10)
    right_shoulder_idx = pose.FindKeypoint(6)

    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]

    right_wrist = pose.Keypoints[right_wrist_idx]
    right_shoulder = pose.Keypoints[right_shoulder_idx]
    
    delta_x_left = left_shoulder.x - left_wrist.x
    delta_y_left = left_shoulder.y - left_wrist.y

    delta_x_right = right_shoulder.x - right_wrist.x
    delta_y_right = right_shoulder.y - right_wrist.y

    left_angle = (math.degrees(math.atan2(delta_y_left, delta_x_left)) + 90) % 360 
    right_angle = (math.degrees(math.atan2(delta_y_right, delta_x_right))  + 90) % 360 

    return left_angle, right_angle

def drawProgress(img, progress):
    radius = 50
    padding = 10
    jetson.utils.cudaDrawCircle(img, (1280-(radius+padding),radius + padding), radius, (255,0,0,100))
    if(trust > 0):
        jetson.utils.cudaDrawCircle(img, (1280-(radius+padding),radius + padding), radius*progress, (0,255,0,255))

def detectLetter(pose):
    left_angle, right_angle = armsAngles(pose)
            
    left_angle = round((left_angle)/45 )*45 
    right_angle = round((right_angle)/45)*45 

    if(left_angle == 360):
        left_angle = 0
    if(right_angle == 360):
        right_angle = 0

    for letter in flag_semaphore:
        left = flag_semaphore[letter]["left"]
        right = flag_semaphore[letter]["right"]
        if(left_angle == left and right_angle == right):
            break
    return letter.upper()

flag_semaphore = {
    'a': {'left': 0, 'right': 45},
    'b': {'left': 0, 'right': 90},
    'c': {'left': 0, 'right': 135},
    'd': {'left': 0, 'right': 180},
    'e': {'left': 225, 'right': 0},
    'f': {'left': 270, 'right': 0},
    'g': {'left': 315, 'right': 0},
    'h': {'left': 45, 'right': 90},
    'i': {'left': 45, 'right': 135},
    'j': {'left': 270, 'right': 180},
    'k': {'left': 180, 'right': 45},
    'l': {'left': 225, 'right': 45},
    'm': {'left': 270, 'right': 45},
    'n': {'left': 315, 'right': 45},
    'o': {'left': 135, 'right': 90},
    'p': {'left': 180, 'right': 90},
    'q': {'left': 225, 'right': 90},
    'r': {'left': 270, 'right': 90},
    's': {'left': 315, 'right': 90},
    't': {'left': 180, 'right': 135},
    'u': {'left': 225, 'right': 135},
    'v': {'left': 315, 'right': 180},
    'w': {'left': 270, 'right': 225},
    'x': {'left': 225, 'right': 315},
    'y': {'left': 270, 'right': 135},
    'z': {'left': 270, 'right': 315},
    ' ': {'left': 0, 'right': 0},
    '-': {'left': 315, 'right': 135},
}

trust = 0
i = 0
message= ""
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)

    for pose in poses:
        if pose.ID == 0:

            if is_gesture(pose):
                frames_to_detect = net.GetNetworkFPS() * 2.5

                letter = detectLetter(pose)

                if i == 0:
                    letter_first = letter
                
                if letter == letter_first:
                    i += 1
                else:
                    i -= 1
                
                if i > frames_to_detect:
                    if(letter == "-"):
                        message = message[:-1]
                    else:
                        message += letter
            
                    i = 0
                
                trust = i/frames_to_detect

                drawProgress(img, trust)

                font2.OverlayText(img, img.width, img.height, f"{letter}    ", 1280-68, 47, (0,0,0,255), (0,0,0,0))
            draw_flags()
            
    font.OverlayText(img, img.width, img.height, f"Message: {message}", 5, 5, (255,255,255,255), font.Gray40)

            

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0} FPS".format(opt.network, net.GetNetworkFPS()))

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break