# Authors
-Tadeáš Těhan, Gymnázium Ostrava-Zábřeh; tadeastehan@gmail.com
-Mikuláš Voňka, Gymnázium Kladno; vonka.mikulas@seznam.cz
-Jan Pavel Šafrata, Gymnázium Evolution; honza@klan.cz
# Abstract
In recent years, the field of artificial intelligence and machine learning has experienced a dramatic increase in interest and innovation. This surge opens new possibilities in various disciplines, including computer vision. In our project, we focused on using the NVIDIA Jetson Nano device, which is equipped with a powerful graphics card optimized for artificial intelligence and computer vision applications. The goal of our work was to recognize the signals of a flag semaphore based on key points on the body using the Jetson Nano. After recognizing individual letters, we were able to decode the transmitted message. From our experiments, we gained valuable insights and discussed the practical advantages and limitations of this device. The results of our work can be found in [GitHub repository](https://github.com/tadeastehan/flag_semaphore_jetson).
# Introduction
The device we used to test the entire project was a small single-board computer, the [NVIDIA Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/) with a GPU. This aspect makes our project portable, and thanks to GPU-accelerated computations, it is also sufficiently fast. To speed up computations, we utilized NVIDIA's [CUDA technology](https://developer.nvidia.com/cuda-toolkit), which allows for efficient processing of parallel tasks. NVIDIA also provides a [Software Development Kit](https://github.comdusty-nv/jetson-inference) for deep learning, which significantly facilitated the development of our application. Based on our experiments, we analyzed the results and discussed in detail the advantages and limitations of this device for practical use.
For recognizing key points on the body, we used a pre-trained model, Pose-ResNet18-Body, which can identify 18 key points on the human body, as shown in the image below. This model allowed us to track hand positions and effectively recognize individual semaphore [flag signals](https://en.wikipedia.org/w/index.php?title=Flag_semaphore&oldid=1228246342). We developed the entire system using the Python programming language, which offers a specialized library for working with this device.
![Key points on the human body detected by the neural network.](images/body_skeleton.png)
# Code Implementation
Our first task was to enable the program to recognize gestures. This involved identifying when the arms are in a straight position, indicating an attempt at a gesture. We explored various methods to define this condition and eventually found an optimal solution. We calculated the differences in the $$x$$ and $$y$$ coordinates between the shoulder and elbow points, as well as between the elbow and wrist points for both arms. These differences provided us with vectors of varying lengths. When a gesture was performed, these vectors were aligned in the same direction. By scaling the vectors so that their $$x$$ coordinates matched, we ensured that if the vectors were pointing in the same direction, their $$y$$ coordinates would be the same. Then we measured the difference between the $$y$$ coordinates of the two vectors, which should be close to zero if the gesture was being performed.
Next, we needed to determine the angles the arms made with the vertical axis. We achieved this by calculating the differences in the x and y coordinates between the shoulder and wrist points and then using the two-argument $$atan2$$ function to find the angle of the arm relative to the vertical axis. We then assigned the angle values for both arms to each character of the semaphore alphabet.
We also developed a condition for accurately detecting the letter the person was trying to signal. Since relying on a single frame could lead to errors, we implemented a more robust approach. We defined a variable $$i = 0$$ and a reference variable $$r$$ as the letter from the first frame. In the algorithm, the value of $$r$$ changes only when $$i$$ equals zero. Subsequent letters from consecutive frames were compared to the value of $$r$$. If they matched, we increased $$i$$ by 1; if they differed, we decreased $$i$$ by 1. When $$i$$ reaches a predefined threshold of $$2.5 \times f$$, where $$f$$ represents the number of video frames per second, which occurs after 2.5 seconds of signalling the same letter, the program confirms the letter and prints it. After reaching this threshold, the variable $$i$$ is reset, the reference variable $$r$$ changes to the current letter, and the detection of the next letter in the message begins.
```python
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
```
## Graphical Interface
The graphical interface, shown in the video below, includes a red circle in the upper right corner that gradually fills with green depending on the variable i. If we hold a position for a sufficient amount of time, the circle fills with green, and the letter is added to the message in the upper left corner. The specific letter detected by the program is also displayed in the centre of this circle. The semaphore flag characters include a space and a backspace character, allowing us to delete letters. The graphical interface also shows the skeleton of recognized points on the human body, holding virtual flags for better imitation of reality.
:::youtube[NVIDIA Jetson Nano Flag Semaphore]{#jYlftoufqjc}
# Results and Discussion
The program we developed successfully detects most semaphore flag characters as shown in the whole alphabet picture below. One significant limitation occurred when the arm was positioned directly above the head. The neural network used has difficulty recognizing key points on the arm in this position. To address this, we adjusted the gesture detection condition so that it does not activate at certain angles above the head.
Another issue arose when the arm crossed the body and pointed to the opposite side. This positioning sometimes caused the program to incorrectly identify key points on the arm, leading to the misrecognition of certain letters. We implemented additional corrective measures to increase accuracy, but there is still room for improvement.
These limitations highlighted the complexity of gesture recognition and the need for continuous refinement. Future work could involve improving the algorithm to better handle extreme arm positions and exploring more advanced machine learning models. Training a custom neural network specifically for these purposes could lead to improved accuracy.
![Image that shows every letter recognized by our implementation.](https://github.com/user-attachments/assets/fab6609f-63b8-4ecc-b5e7-29f75a40d933)
