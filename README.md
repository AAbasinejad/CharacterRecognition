# CharacterRecognition

### Introduction
------
Recognize given a scanned picture with a character through a CNN using Keras' Functional API, moreover we had
to generate our own dataset to train the Neural Network. The following sections describes the model that was
built, the dataset generated and other choices made during development.


### The Dataset
------
Since in this proejct dataset to train the model were not given we had to create our own. hence by taking the
possible fonts and, using the [PIL package](https://pillow.readthedocs.io/en/stable/index.html), each possible character has been made. At the end we have a total of 94 characters and 11 fonts, and from this 11 fonts there is variations of how the font could be: normal (N), **bold** (B), *italic* (I), **_bold and italic_** (BI) at the same time. Considering these combinations gives us a total of 28 possible ways of drawing each character.<br/>

Initially each character has been drawn once, using a 64x64 canvas on a white background, with the character roughly centered on the canvas. This process, coded on the file `data_generator.py`, generates a total of $94 \times 28 = 2632$ images. But, this is not enough to train our Neural Network. To create more data and at the same time augment the already existing data, the following procedures was applied (for each character we run this process a number of times - more on this later):<br/>
- Add a random noise with probability of 0.6 - Here random noise is just white pixels on the drawn character.
- Add a random rotation between -60&deg; and 60&deg; with probability 0.6.
- Add a random translation in the x,y axis with probability 1.
