#Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd

# Adjust for your folder of music
path = '/Users/coltonbenson/Documents/Intro to ML - Final Project/Music'

pipe = pipeline("audio-classification", model="lewtun/distilhubert-finetuned-music-genres-small")

# Import Data
music_dta_3 = pd.read_csv('music_with_labels_3_genres.csv')

# Intialize Values
correct = 0
ind = 0
pred_pop = 0
pred_rock = 0
pred_hh = 0
correct_pop = 0
correct_rock = 0
correct_hh = 0
pred_pop_act_hh = 0
pred_pop_act_rock = 0
pred_hh_act_pop = 0
pred_hh_act_rock = 0
pred_rock_act_pop = 0
pred_rock_act_hh = 0

# Loop Through Elements
for i in music_dta_3['Audio']:
    output = pipe(path + "/" + i)
    filter = list()
    
    # Filter for predictions of relevance
    for n in output:
        if (n['label'] in ['Rock', 'Pop', 'Hip-Hop']):
            filter.append(n)
    # Determine prediction (most likely of three choices is first in list)
    pred = filter[0]['label'].lower().replace("-", "")
    actual = music_dta_3['Label'][ind]
    
    # Add to correct total if prediction matches label
    if (actual == pred):
        correct += 1

    # Rock prediction
    if (pred == 'rock'):
        pred_rock += 1
        if (actual == 'rock'):
            correct_rock += 1
        elif (actual == 'pop'):
            pred_rock_act_pop += 1
        else:
            pred_rock_act_hh += 1
    # Pop prediction
    elif (pred == 'pop'):
        pred_pop += 1
        if (actual == 'pop'):
            correct_pop += 1
        elif (actual == 'rock'):
            pred_pop_act_rock += 1
        else:
            pred_pop_act_hh += 1
    # Hiphop prediction
    else:
        pred_hh += 1
        if (actual == 'hiphop'):
            correct_hh += 1
        elif (actual == 'rock'):
            pred_hh_act_rock += 1
        else:
            pred_hh_act_pop += 1
    ind += 1

# Print Outcome
print(correct) # 144
print(f"Classification Accuracy: {correct/300}") # 48%
print(f"Correct Pop: {correct_pop}") # 77
print(f"Correct HH: {correct_hh}") # 1
print(f"Correct Rock: {correct_rock}") # 66
print(f"Predict Pop, Actual HH: {pred_pop_act_hh}") # 99
print(f"Predict Pop, Actual Rock: {pred_pop_act_rock}") # 29
print(f"Predict HH, Actual Pop: {pred_hh_act_pop}") # 1
print(f"Predict HH, Actual Rock: {pred_hh_act_rock}") # 5
print(f"Predict Rock, Actual HH: {pred_rock_act_hh}") # 0
print(f"Predict Rock, Actual Pop: {pred_rock_act_pop}") # 22
