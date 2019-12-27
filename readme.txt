#Description

This implementation aims to measure the elbow angle from a webcam and write it to the serial port.

Please see this video to get a rough idea of what my program does:
https://photos.app.goo.gl/Ha8mX6wqMgksg8gQ8

This is how my programs operates:
1. Extract everything which matches the skin color to a bipolar image.
2. Find the biggest contours (if the number of contours in the frame is > 3, then it will only use the 3 biggest).
3. Extract the left most or the right most contour depending on whether the user last pressed 'l' or 'r'.
4. Then find the topmost point of that countour.
5. Measure the angle between the highest point and the centroid and the x-axis on that contour.

A lot of inspiration was taken from this project: https://github.com/PierfrancescoSoffritti/Handy

The code for measuring the angle between given points was extracted from here: 
    https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd

If you are new to OpenCV and need to learn more about the frame work, please refer to this website:
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html

#Instructions:
##Notes
1. This code has only been tested on a linux OS. There might be some issues with serial port connectivity on other OS etc.
2. Make sure you have a white background, that you are wearing a shirt and nothing in the frame (not even you shirt) matches your skin color.
3. Make sure that the arm that you use doesn't go out of the image frame. It needs to stay inside the frame all of time for elbow angles to be measured.
4. The program can only handle 1 arm or 1 face and 1 arm or 1 face 2 arms. Anything else will you issues.

##How to run
1. Connect an FTDI to your computer and into the cat. 
2. Run the program by typing into the terminal: python3 vision_commander.py (You might need to resolve some issues with serial port connectivity at this stage).
3. Now you will see 2 different screens on your window. Make sure your skin color stands out in them, that it is significantly different.
4. Decide which arm you want to give instructions by (left or right). Flex and unflex the elbow, making sure that your arm doesn't go out of the image frame. 
5. Bring your elbow into the box on the image. Make sure that the color tone of the skin has the greatest resemblance to the majority of your arm. Press "r" or "l" depending on which arm you chose.
6. The bottom-left of the image shows you the angle. Try to get a 0, -70, and 70 onto it without going out of the frame. Adjust your elbow's position if it does.
7. Press "r" or "l" to switch between the arms.
8. Press "q" to close the program.


##Potential issues:
1. Serial port issues: Please read through the code of Py_commander.py. It is very self-explanatory and you should be able to figure it out.

2. Comptuter vision related:
    If the program is having a hard time differentiating your skin from the surroundings, then you could play around with the following in the vision_commander.py file:

-> skin_col_range(frame,coordinates): Play around with the variables Color_wavelength_range, Saturation_brightness. You might need to understand how an HSV system operates. Please go here and play around with the HSV color scheme: http://colorizer.org/.

-> angle_writer(cap,key,skin_range,port): You could also also increase or decrease the kernels for gaussian blurring or the morphological transformations. 
    If the area of contours is too thick or thin, then alter the number of iterations for dilations.
