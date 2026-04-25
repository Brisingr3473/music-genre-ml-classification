# Import Libraries
import pandas as pd
import os

# Change to your path
path = '/Users/Tarush/Desktop/STA 395/ML_Music_Genre-main'

# Read in data
file_names = sorted(os.listdir(path))
music_dta = pd.DataFrame(file_names)

# Create label by parsing file name
music_dta['label'] = ""

ind = 0
for i in file_names:
    music_dta['label'][ind] = i.split(sep = ".")[0]
    ind += 1

# Clean up column names
music_dta.columns = ['Audio', 'Label']
music_dta = music_dta.loc[1:, :]

music_dta = music_dta[['Label', 'Audio']]

music_dta.to_csv('music_with_labels.csv', index=False)

# Create dataframe to only include subset of genres
music_dta_3 = music_dta[music_dta['Label'].isin(['rock', 'pop','hiphop'])]

music_dta_3.to_csv('music_with_labels_3_genres.csv', index=False)
