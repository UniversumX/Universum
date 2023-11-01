# Universum
Universum Repository

The Universum Project, a collaborative initiative led by seven students and NeuroTech, seeks to revolutionize proprioception modeling with significant implications for virtual reality, neurorehabilitation, robotics, and sports science. We focus on predicting real-time lumbar tilt perception, a critical aspect of proprioception, using cutting-edge computer vision for pose estimation and advanced machine learning techniques. Our methodology involves processing denoised EEG data collected from an 8-channel brain-computer interface (BCI) headset, augmented with other EEG recording methods. The core objective is to develop a robust, real-time model that predicts lumbar tilt perception exclusively from limited-channel EEG data, which undergoes advanced noise reduction and pattern recognition techniques. A successful implementation of this project signifies that a comprehensive and dynamic understanding of human proprioception and kinesthesia can be achieved through the sophisticated analysis of EEG data using the latest machine learning methodologies.

# Cognitive and Biological Considerations:
Our research is anchored in comprehending the neurophysiological processes of proprioception. We collaborate closely with neuroscientists to identify the neural underpinnings of lumbar tilt perception, which will guide the refinement of our 8-channel BCI headset and enhance our EEG recording methods.

# Engineering Approach:
Our engineering strategy involves amalgamating pose estimation algorithms and machine learning techniques to effectively denoise and interpret EEG data. We intend to deploy Convolutional Neural Networks (CNNs) for pose estimation, while Autoencoders, LSTM networks, and Transformer models will be utilized for EEG data denoising and real-time tilt prediction.

# Research Steps:

Data Collection: Participants will perform lumbar tilt tasks while wearing an 8-channel BCI headset. Simultaneously, a CNN-based pose estimation algorithm will establish ground truth data.

Data Preprocessing: Collected EEG data will be subjected to rigorous preprocessing, including signal filtering, normalization, and segmentation, to prepare optimal input for our machine learning models.

Model Development: Autoencoders will denoise EEG data. LSTM networks and Transformer models will subsequently predict real-time lumbar tilt based on the denoised data.

Model Evaluation: Developed models' performance will be assessed using metrics such as RMSE and R² values.

Model Interpretation: We will analyze feature importance, attention weights, or saliency maps from the final model to understand neural mechanisms and patterns contributing to accurate lumbar tilt prediction.

# Expected Outcomes: 
The Universum Project aims to develop a model capable of accurately predicting an individual's real-time lumbar tilt using denoised EEG data from an 8-channel BCI. Project success will be gauged by the model's predictive accuracy, neural insight gained into proprioception, and its potential applications in advanced prosthetics, immersive virtual reality, and neuroscience research.

# Strategies for Limited Data:
Recognizing the challenges of limited data, we will:

Augment data to increase the dataset size through transformations such as rotation, translation, and scaling.
Employ feature selection and digital signal processing techniques to focus on the most relevant dataset features, thereby reducing noise and improving model performance.
Use ensemble learning techniques to combine predictions from multiple machine learning models, enhancing overall prediction accuracy.
The Universum Project team is confident in these strategies' ability to effectively address limited data challenges, leading to the successful development of a reliable, real-time model for human lumbar tilt perception.

