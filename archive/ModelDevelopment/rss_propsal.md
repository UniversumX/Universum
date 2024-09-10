Title, Author(s)

Model Development Proposal

## Introduction (2%): Introduce your problem and the overall plan for approaching your problem

The problem that is being solve is a model that is able to take in the data that comes from the BCI and classify it in real-time. This problem comes down to an optimization problem where we are trying to choose a model architecture and hyperparameters that optimize the F-score. The current SOA uses a Transformer to solve this problem. My idea of how to approach this problem is to just copy what the EEGFormer paper does, and then use other ML techniques to improve the results. This could include changing the architecture, as well as some fancier techniques like meta-learning. The goal is to make sure that this model is able to generalize with a small number of training examples.

After a few ideas are generated for how the model could be improved, it will be important to test what different components improve the model's performance. The Optuna library can be used for this.

## Problem Statement (2%): Describe your problem precisely specifying the dataset to be used, expected results and evaluation

## Literature Review (2%): Describe important related work and their relevance to your project

## Technical Approach (2%): Describe the methods you intend to apply to solve the given problem

## Intermediate/Preliminary Results (2%): State and evaluate results of your study or other related studies/implementations

# Steps for Model Development

Figure out input size from

## Software/Tools

For this project, you obviously will need some sort of code editor/ide to work with it. We might make a database using something like MySQL or something, so that might be another potential tool.

For specific libraries, we will be using PyTorch (cuz its cool), Optuna (for model optimization), and probably scikit-learn (for evaluation metrics).

## Model Components (heavily inspired from EEGFormer paper)

From the EEGFormer paper, their model has 3 separate components, each of which can have their own improvements.

#### 1. Feature Extractor (optional)

This was a 1D CNN that swept across the input data and excracted specific features. The purpose of this, I can imagine, is similar to how neuoscientists look at EEG data and look for little scribbles that mean different things. So the specific purpose of this component is to encode the little blips in the EEG data and then send them further into the model to figure out how they relate to each other.

#### 2. Encoder

Imma be so fr, this is the part I understand the least, but here goes: The input to the encoder is the output from the previous module plus a position encoding. There are three submodules of the encoder, one for regional data, one for spatial data, and one for temporal data. I have no clue what this means

#### 3. Decoder

The decoder for thier paper was a basic CNN to take the embedding output from the encoder and then feed it to a full-connected layer to classify.

## Training

Model traning will be pretty simple, take the data we have, preprocess it, and then feed it into our model. We can also create some synthetic data to improve the robustness of our model. Some things to do is to add noise (multiple levels of noise).

## Validation

**Experiments**: We will need to run a lot of experiments to figure out which model architecture is the best. This will be done using Optuna, which will allow us to run a lot of experiments in parallel. We will need some clusters to run these experiments on, so we need to learn more about that.

**Benchmarks**: If we were inventing a new model architecture, then it would be smart to try it out on a lot of different benchmarks, but for this project if we just want to maximize our predictive power with our model, then we probably just need to use the BCI to get data from a represenative sample of the population, and use that as the validation data for this project.

**Robustness**: We will need to test the robustness of the model to different types of noise. This will be done by adding different types of noise to the input data and seeing how the model's performance changes. This will be done using the `torchattacks` library and any other potential adversarial model we create. (Here is more info on torchattacks, idk much about it: https://arxiv.org/pdf/2010.01950.pdf)

**Generalization**: This model should generalize to other people and also be able to generalize over a period of time. In one paper we read, the model had different accuracy on the same person in between sessions, so making the model generalize will be super-important so we have a model that can predict your perception, not a model that can predict the how your perception used to be. This can be done with cross-validation as well as regularization techniques like L1 or L2 regularization.

## Explainability

A motivation of this project is to actually learn something about the certain brain regions/EEG data that we get. I think exploring the field of ML explainability will allow us to get these insights. From my limited knoweldge of the field, I know there is one thing we can do:

To figure out which brain regions are the most important for the model's predictions, we can figure out how the model's performance changes when we shuffle around the input from one of the channels. If that channel doesn't contain any useful data, the model should be able to extract that and not change its predictions, but if that channel has useful data, then its F1-score should drastically decrease. We can do other versions of this, we can randomize muliple channels and see if the information they mutally carry is important, etc.

## Future Ideas

#### Future Project--Reduce Cost of Labeling Data

So for this project, the data is very easy to label, but for future projects it might be hard to label. I propose one experiment where we use a specific training method. We will start out with having some of our data be unlabeled and a small fraction being labeled. We will then train the best model architecture (that we find through experimentation) on this. We will use that model to classify the unlabeled data, and if it is very confident for 1 specific label, then we will pseudo-label that data with the model's prediction. This should give us more labeled data. We then train another model on this new dataset, and rinse and repeat. We can evalute the performance of this model. If it is similar to our original model's performance, then we can probably save some time in the future labeling data if it is costly.

#### Decreasing Model Scale

We can use model distillation techniques to reduce the size of our final model. We can create a teacher and student model. The teacher model is the original beefy model and the student model is a randomly-intiaizlied network. We then train the student to output the same outputs as the teacher model. This in practice has shown to decrease the number of hyperparameters a model need by 1-2 order of magnitude without hindering performance.
