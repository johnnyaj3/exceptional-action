# exceptional-action
## Data Using
To collect the video data, smartphone: iPhone 11, was utilized to capture the video of action as 30 frames per second (FPS) and in the resolution of 1920×1080. The file contains 9 different video clips collected. Video (1) to (3) demonstrate the repeated wiping action by three different people. The differences of the three videos are the speed of wiping, the size of human body shape of the three people and the times of the action executed. Video (4) to (9) simulate the actions behavior in a general working environment, such as assembling carts, screwing, hammering, loading carts and touching button. Each video clip has different range of interested (ROI) and different executing time of each action. The exceptional actions conducted in video (4) to (9) were the behaviors of intentionally stopping, drinking water, resting, stretching the arm and so on to present the real-world field operating.
## Code for framework
In this research, an Entropy Signal Clustering (ESC) was proposed to detect the exceptional actions from a video in "worker_dataset" folder.
Please download the pre-working file: "repnet_ckpt"[^1], "model.py"[^2] and "utils_hs.py"[^3]. Thanks to the contributions of the aboved authors, this proposed framework applies and updates the code for exceptional action exceptional actions.
There are three code of the proposed framework. 
1) TSM_computation: Utilizing the video from worker_dataset and applying [^1] and [^2] to generate TSM for each video, save the TSM as ".npz" file to "TSM" folder.
2) TSM to two entropy signal: Utilizing the TSM file to generate two entropy signal. Saving the signals as ".npy" file to "entropy" folder.
3) Exceptional action detection: Utilzing the two entropy signal form ".npy" file to apply signal filtering and LRRDSKEAM to detect exceptional action by frame. 


[^1]:Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A. (2020) 'Counting out time: Class agnostic video repetition counting in the wild' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10387-10396. Available at: https://doi.org/10.48550/arXiv.2006.15418.
[^2]:https://gist.github.com/mills-nick/36177d6aaeee56b6242b11720451d0ac
[^3]:Sutrisno, H., & Yang, C.-L. (2021, July 18-21). Discovering defective products based on multivariate sensors data using local recurrence rate and robust k-means clustering. (Paper presented at the ICPR Taichung, Taiwan)

