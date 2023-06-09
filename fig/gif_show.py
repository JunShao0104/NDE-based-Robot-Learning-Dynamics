from PIL import Image, ImageSequence
import cv2
import numpy
 
pic_name = "obstacle_avoidance_pushing_visualization_ode_multi_step_3.gif"
im = Image.open(pic_name)
 
for frame in ImageSequence.Iterator(im):
    frame = frame.convert('RGB')
    cv2_frame = numpy.array(frame)
    show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(pic_name, show_frame)
    cv2.waitKey(50)