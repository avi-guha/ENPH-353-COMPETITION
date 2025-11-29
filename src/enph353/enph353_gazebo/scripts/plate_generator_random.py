#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import os
import random
import string

from PIL import Image, ImageFont, ImageDraw

def random_word_fixed_length(length=12):
    '''Generate a random alphabetic word (with spaces allowed) of exact length'''
    if length < 3:
        length = 3  # minimum sensible length

    # first character cannot be a space
    word = random.choice(string.ascii_uppercase)
    
    for _ in range(length - 1):
        word += random.choice(string.ascii_uppercase + ' ')

    return word[:length]  # ensure exactly the required length

def loadCrimesProfileCompetition():
    '''
    @brief returns a set of clues for one game and save them to plates.csv

    @retval clue dictionary of the form 
                [size:value, victim:value, ...
                 crime:value,time:value,
                 place:value,motive:value,
                 weapon:value,bandit:value]
    '''
    key_list = ['size','victim','crime','time','place','motive','weapon','bandit']

    clues = {}

    # Save the clues to plates.csv
    with open(SCRIPT_PATH + "plates.csv", 'w') as plates_file:
        csvwriter = csv.writer(plates_file)

        for key in key_list:
            value = random_word_fixed_length(12)
            clues[key] = value.upper()
            csvwriter.writerow([key, value.upper()])

    return clues

# Find the path to this script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
TEXTURE_PATH = '../media/materials/textures/'

banner_canvas = cv2.imread(SCRIPT_PATH+'clue_banner.png')
PLATE_HEIGHT = 600
PLATE_WIDTH = banner_canvas.shape[1]
IMG_DEPTH = 3

clues = loadCrimesProfileCompetition()

i = 0
for key, value in clues.items():
    entry = key + "," + value
    print(entry)

    # Generate plate

    # Convert into a PIL image (to use monospaced font)
    blank_plate_pil = Image.fromarray(banner_canvas)
    draw = ImageDraw.Draw(blank_plate_pil)
    font_size = 90
    monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 
                                    font_size)
    font_color = (255,0,0)
    draw.text((250, 30), key, font_color, font=monospace)
    draw.text((30, 250), value, font_color, font=monospace)

    # Convert back to OpenCV image and save
    populated_banner = np.array(blank_plate_pil)
    cv2.imwrite(os.path.join(SCRIPT_PATH+TEXTURE_PATH+"unlabelled/",
                                "plate_" + str(i) + ".png"), populated_banner)
    i += 1

