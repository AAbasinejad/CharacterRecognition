import numpy as np
import pickle

from contextlib import redirect_stdout

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, MaxPooling2D, Concatenate


'''
SCORE ON VALIDATION

char: 0.8802
font: 0.6237
italic: 0.8577
bolds: 0.8615

average:


SCORE ON TEST

char: 0.626012
font: 0.258292
italic: 0.739884
bolds: 0.792203

average: 0.572

'''


if __name__ == "__main__":

    np.random.seed(123)

    ## Read data and labels
    print("Reading data...")
    X_train = np.load("../data/X_data_augmented.npy")
    Y_fonts = np.load("../data/Y_fonts_augmented.npy")
    Y_chars = np.load("../data/Y_chars_augmented.npy")
    Y_bold = np.load("../data/Y_bold_augmented.npy")
    Y_italic = np.load("../data/Y_italic_augmented.npy")
    
    # Get a random permutation of data and labels to have a better train
    # When using the validation_split parameter on the fit method it gets the 
    # data sequentially, so the validation result is biased.
    # If we get a random permutation of the data before calling fit, the results
    # will be more reliable
    i = np.random.permutation(np.arange(X_train.shape[0]))
    X_train = X_train[i]
    Y_fonts = Y_fonts[i]
    Y_chars = Y_chars[i]
    Y_bold = Y_bold[i]
    Y_italic = Y_italic[i]

    # One hot encode chars and fonts
    lb_chars = LabelBinarizer()
    lb_fonts = LabelBinarizer()
    Y_fonts = lb_fonts.fit_transform(Y_fonts)
    Y_chars = lb_chars.fit_transform(Y_chars)
    num_fonts = len(Y_fonts[0])
    num_chars = len(Y_chars[0])
    
    print("Finished reading data...")

    # Save one hot encodes to do a inverse transformation after predicting
    with open('../data_out/fonts_encoder.pkl', 'wb') as f:
        pickle.dump(lb_fonts, f)
    with open('../data_out/chars_encoder.pkl', 'wb') as f:
        pickle.dump(lb_chars, f)

    # This neural network starts with two branches.
    # One branch predicts bolds and the other one predicts fonts

    # These two branches are merged and serve as input to two new branches.
    # One branch predicts italics and the other one predicts character


    # Input Layer
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)

    # First Branch - First Level
    # 1º Hidden Layer
    # CONV32 => POOL2 => DROPOUT(0.25)
    conv1 = Convolution2D(32, 3,  padding='same', activation = 'relu')(inp)
    pooling1 = MaxPooling2D(pool_size=2)(conv1)
    dropout1 = Dropout(0.25)(pooling1)

    # 2º Hidden Layer
    # CONV64 => POOL2 => DROPOUT(0.25)
    conv2 = Convolution2D(64, 3, padding='same', activation = 'relu')(dropout1)
    pooling2 = MaxPooling2D(pool_size=2)(conv2)
    dropout2 = Dropout(0.25)(pooling2)

    # Output Layer - BOLD
    flat1 = Flatten()(dropout2)
    dropout3 = Dropout(0.5)(flat1)
    out3 = Dense(1, activation='sigmoid', name="output_bold")(dropout3)  

    ########### END FIRST BRANCH - FIRST LEVEL ##################

    # Second Branch - First Level
    # 1º Hidden Layer
    # CONV32 => POOL2 => DROPOUT(0.25)
    conv8 = Convolution2D(32, 3, padding='same', activation = 'relu')(inp)
    pooling8 = MaxPooling2D(pool_size=2)(conv8)
    dropout11 = Dropout(0.25)(pooling8)

    # 2º Hidden Layer
    # CONV64 => POOL2 => DROPOUT(0.25)
    conv9 = Convolution2D(64, 3, padding='same', activation = 'relu')(dropout11)
    pooling9 = MaxPooling2D(pool_size=2)(conv9)
    dropout12 = Dropout(0.25)(pooling9)

    # 3º Hidden Layer
    # DENSE128 => DROPOUT(0.5)
    flat4 = Flatten()(dropout12)
    dense4 = Dense(128, activation="relu")(flat4)
    dropout14 = Dropout(0.5)(dense4)
    
    # Output Layer - FONTS
    out1 = Dense(num_fonts, activation='softmax', name="output_fonts")(dropout14)
    
    ########### END SECOND BRANCH - FIRST LEVEL ##################

    # Merge two branches
    merged = Concatenate()([dropout2, dropout12])

    # First Branch - Second Level
    # 1º Hidden Layer
    # CONV32 => POOL2 => DROPOUT(0.25)
    conv3 = Convolution2D(32, 3, padding='same', activation = 'relu')(merged)
    pooling3 = MaxPooling2D(pool_size=2)(conv3)
    dropout4 = Dropout(0.25)(pooling3)

    # 2º Hidden Layer
    # CONV64 => POOL2 => DROPOUT(0.25)
    conv4 = Convolution2D(64, 3, padding='same', activation = 'relu')(dropout4)
    pooling4 = MaxPooling2D(pool_size=2)(conv4)
    dropout5 = Dropout(0.25)(pooling4)

    # 3º Hidden Layer
    # CONV128 => POOL2 => DROPOUT(0.25)
    conv41 = Convolution2D(128, 3, padding='same', activation = 'relu')(dropout5)
    pooling41 = MaxPooling2D(pool_size=2)(conv41)
    dropout51 = Dropout(0.25)(pooling41)

    # Output Layer - ITALICS
    flat2 = Flatten()(dropout51)
    dropout6 = Dropout(0.5)(flat2)
    out4 = Dense(1, activation='sigmoid', name="output_italics")(dropout6) 

    ########### END FIRST BRANCH - SECOND LEVEL ##################

    # Second Branch - Second Level
    # 1º Hidden Layer
    # CONV32 => POOL2 => DROPOUT(0.25)
    conv5 = Convolution2D(32, 3, padding='same', activation = 'relu')(merged)
    pooling5 = MaxPooling2D(pool_size=2)(conv5)
    dropout7 = Dropout(0.25)(pooling5)

    # 2º Hidden Layer
    # CONV64 => POOL2 => DROPOUT(0.25)
    conv6 = Convolution2D(64, 3, padding='same', activation = 'relu')(dropout7)
    pooling6 = MaxPooling2D(pool_size=2)(conv6)
    dropout8 = Dropout(0.25)(pooling6)

    # 3º Hidden Layer
    # CONV128 => POOL2 => DROPOUT(0.25)
    conv7 = Convolution2D(128, 3, padding='same', activation = 'relu')(dropout8)
    pooling7 = MaxPooling2D(pool_size=2)(conv7)
    dropout9 = Dropout(0.25)(pooling7)

    # 4º Hidden Layer
    # DENSE128 => DROPOUT(0.5)
    flat3 = Flatten()(dropout9)
    dense3 = Dense(128, activation="relu")(flat3)
    dropout10 = Dropout(0.5)(dense3)

    # Output Layer - CHARS
    out2 = Dense(num_chars, activation='softmax', name="output_chars")(dropout10)

    # Compile and fit the model
    model = Model(inputs=inp, outputs=[out1, out2, out3, out4])
    model.compile(optimizer='adam',
                loss=['categorical_crossentropy', 'categorical_crossentropy', 
                    "binary_crossentropy", "binary_crossentropy"],
                metrics=['accuracy'])
    model.fit(X_train, [Y_fonts, Y_chars, Y_bold, Y_italic], epochs=10, batch_size=64, validation_split=0.1)
    
    # Save model summary    
    with open('../data_out/model/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Save model and model weights
    model_json = model.to_json()
    with open("../data_out/model/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("../data_out/model/model.h5")
    
    print("Finished")