# CharacterRecognition

### Introduction
------
Recognize given a scanned picture with a character through a CNN using Keras' Functional API, moreover we had
to generate our own dataset to train the Neural Network. The following sections describes the model that was
built, the dataset generated and other choices made during development.


### Dataset
------
Since in this proejct dataset to train the model were not given we had to create our own. hence by taking the
possible fonts and, using the [PIL package](https://pillow.readthedocs.io/en/stable/index.html), each possible character has been made. At the end we have a total of 94 characters and 11 fonts, and from this 11 fonts there is variations of how the font could be: normal (N), **bold** (B), *italic* (I), **_bold and italic_** (BI) at the same time. Considering these combinations gives us a total of 28 possible ways of drawing each character.<br/>

Initially each character has been drawn once, using a 64x64 canvas on a white background, with the character roughly centered on the canvas. This process, coded on the file `data_generator.py`, generates a total of 94 * 28 = 2632 images. But, this is not enough to train our Neural Network. To create more data and at the same time augment the already existing data, the following procedures was applied (for each character we run this process a number of times - more on this later):<br/>
- Add a random noise with probability of 0.6 - Here random noise is just white pixels on the drawn character.
- Add a random rotation between -60&deg; and 60&deg; with probability 0.6.
- Add a random translation in the x,y axis with probability 1.

The first two steps guarantees diversity on the data, while the third one, besides also guaranteeing diversity, make us sure that no new image will be completely the same. The problem with the initial generated data is that the dataset is unbalanced on the fonts classes. We have 11 different fonts, but the fonts with B, I, and B_I have more images then fonts with only N. To tackle this problem, while augment the data we generate more new images for the fonts that don't have B, I, and B_I, this process has benn done in a way that the final dataset will be balanced on the fonts classes, obviously is also balanced on the characters classes, but it is not balanced on the bolds and italics classes, the process to generate this data is coded on the `data_augmentation.py` file. Here is a quick summary of the number of training instances:<br/>
- Fonts: 16920 images each.
- Characters: 1980 images each.
- Bolds: 59220 are bold, and 126900 are not.
- Italics: 50760 are italic and 135360 are not.
at the end a total of 186120 images for training has been made.

### CNN Model
------
To tackle this problem we created a Convolutional Neural Network using the Keras Functional API. Since the task is to predict four different outputs we decide to tackle this problem separetely, hence, we initially create two different branches on our network, these two branches are eventually merged and then split again on two new branches, each branch of the 4 created will predict one of the outputs. The reasons for creating these different branches are: this generates a visually 'complex' model, but each branch is simple on its own, which helps to avoid overfitting, and at the same time we have freedom to tune the best configuration of each branch for each output.<br/>
Between these four branches, we have four different Network configurations, the idea was to use more complex models to outputs that we judged were more complex, and simpler models for outputs that we judged were simpler. Also, where to place each output was something that we decided observing the results from the Network. You can check the model.png file to visually check the network configuration, as we will not reproduce it here because it is too big.<br/>
![alt text](https://github.com/AAbasinejad/CharacterRecognition/blob/master/model.png)
