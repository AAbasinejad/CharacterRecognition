import numpy as np
import string

from PIL import Image, ImageDraw, ImageFont
from os import listdir



if __name__ == "__main__":

    ## Directory with fonts data
    font_dir = "../data/fonts/"
    ## Get all fonts - 11 fonts in total, 28 if we count bold and italic versions
    files = [x for x in listdir(font_dir) if x.split(".")[1] in ["ttf", "otf"]]
    ## Get all possible chars - Total of 94
    chars = string.printable[:-6]
    ## Background color for the images - Always white
    background_color = 255
    ## Fonts size
    font_size = 44
    ## Size of the canvas where to draw image - Images are always 64x64
    image_size = 64
    # Number of images that will be generated 
    # So the total is 94*28
    nsamples = len(files)*len(chars)

    ## nparray to save image data
    np_data = np.zeros(shape=(nsamples, 64, 64, 1), dtype="float")
    ## nparrays to save labels - one for each output - font, char, bold and italic
    np_fonts = np.empty(shape=(nsamples), dtype=object)
    np_chars = np.empty(shape=(nsamples) , dtype=str)
    np_bold = np.zeros(shape=(nsamples))
    np_italic = np.zeros(shape=(nsamples))

    count = 0
    ## Loop through fonts
    for font in files:
        filename = font_dir + font
        # Get font name
        font_name, _, bold_italic = font.split(".")[0].partition("_")
        # Get if it is bold
        is_bold = 1 if bold_italic == "B" or bold_italic == "B_I" else 0 
        # Get if it is italic
        is_italic = 1 if bold_italic == "I" or bold_italic == "B_I" else 0
        # Loop through all characters
        for char in chars:
            # Define white canvas
            canvas = Image.new("L", (image_size, image_size), 255)
            # Draw canvas
            draw = ImageDraw.Draw(canvas)
            # Define font and font size
            font = ImageFont.truetype(filename, font_size)
            # Get font width and height
            (font_width, font_height) = font.getsize(char)
            # Define coordinates where to draw the character
            x = (image_size - font_width)/2
            y = (image_size - font_height)/2
            # Draw the char using the given font
            draw.text((x, y), char, font=font)
            # Put image as array on nparray
            np_data[count] = np.asarray(canvas).reshape((64,64,1))
            # Save labels
            np_bold[count] = is_bold
            np_italic[count] = is_italic
            np_fonts[count] = font_name
            np_chars[count] = char
            count += 1
    
    # Normalize data to be between 0 and 1
    np_data = np_data/255
    # Save nparrays
    np.save("../data/X_data.npy", np_data)
    np.save("../data/Y_bold.npy", np_bold)
    np.save("../data/Y_italic.npy", np_italic)
    np.save("../data/Y_fonts.npy", np_fonts)
    np.save("../data/Y_chars.npy", np_chars)