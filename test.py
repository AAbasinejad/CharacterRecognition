import sys
import pickle
import numpy as np
import pandas as pd
from keras.models import model_from_json

if __name__ == "__main__":

    # Get input and output files
    x_data_filename = sys.argv[1]
    output_filename = "../data_out/test/"+sys.argv[2]+".csv"
    
    # Load test data
    X_data = np.load(x_data_filename)
    
    ## Load model
    json_file = open('../data_out/model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../data_out/model/model.h5")

    ## Load encoders
    lb_fonts = pickle.load(open("../data_out/fonts_encoder.pkl", "rb"))
    lb_chars = pickle.load(open("../data_out/chars_encoder.pkl", "rb"))

    ## Predict
    preds = model.predict(X_data)

    ## Format output
    preds[2][np.where(preds[2] >= 0.5)] = 1
    preds[2][np.where(preds[2] < 0.5)] = 0

    preds[3][np.where(preds[3] >= 0.5)] = 1
    preds[3][np.where(preds[3] < 0.5)] = 0

    # Get label names
    fonts = lb_fonts.inverse_transform((preds[0] == preds[0].max(axis=1, keepdims=1)))
    chars = lb_chars.inverse_transform((preds[1] == preds[1].max(axis=1, keepdims=1)))
    bold = preds[2].reshape(preds[2].shape[0])
    italic = preds[3].reshape(preds[3].shape[0])

    # Format output
    rows = np.dstack((chars, fonts, bold, italic)).reshape(fonts.shape[0], 4)
    
    # Save output as csv file
    print("Saving {}".format(output_filename))
    df = pd.DataFrame(rows)
    df.to_csv(output_filename, header=False, index=False)
    print("Finished")