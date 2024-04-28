# CS5330 Spring 2023 Project 3
## Bryan Teck Yean Ang

## Links to video
Video for task 9 demonstrating the system classifying objects:
https://drive.google.com/file/d/186RCFxFdpr6KZQ04bO0Xz8Kxu__MRHLA/view?usp=share_link


## IDE and OS

The project code was written, compiled and ran on Windows 11 and Visual Studio.

## Running exes

After compiling the code, the exe can be run on a console with no additional commands.


Once the program is running, different keystrokes can be pressed to enable different views of the video feed. If a view is already enabled and the keystroke is pressed again, it will be toggled off. These views corresponds to the different parts of the process pipeline as required in the tasks.

Pressing spacebar will also allow the user to save the current image features along with a label to a text file (`db.txt`). The program will be put on hold until a label name is entered in the console. 

The following is a list of commands:
| Keystroke | Action |
|---|---|
| spacebar | Save features |
| t | Threshold Image|
| c | Cleaned up image |
| r | Region Map |
| d | Object of interest + bounding box and axis of least moment |
| q | quit |

## Extensions
__GUI__: The GUI has been extended to be able to show all different steps of the pipeline with various button toggles. 2 different features (bounding box and axis) were shown for the feature view.

__12 Objects__: Additional 2 objects were added

__Extra implementations__: The region growing and calculations for moments around axis of least central moment was implemented from scratch.

__Unknown classification thresholding__: A threshold of 2 standard deviations  units was used to allow the system to classify unknowns. The formula used was to take the minimum distance to a data label in k-nearest, and divide by the sum of all feature standard deviations.

## Time travel days
Two time travel days were used, extending the deadline to Monday, 26th Feb 2023 00:00 PST.