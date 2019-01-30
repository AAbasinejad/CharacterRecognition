# CharacterRecognition

### Introduction
------
Recognize given a scanned picture with a character through a CNN using Keras' Functional API, moreover we had
to generate our own dataset to train the Neural Network. The following sections describes the model that was
built, the dataset generated and other choices made during development.


### The Dataset
------
Since in this proejct dataset to train the model were not given we had to create our own. hence by taking the
possible fonts and, using the [PIL package](https://pillow.readthedocs.io/en/stable/index.html), each possible character has been made. We have a total of 94 characters and 11 fonts, from this 11 fonts we have variations of how the font could be: normal (N),
bold (B), italic (I), bold and italic (BI) at the same time. Considering these combinations we have a total of 28
possible ways of drawing each character.
