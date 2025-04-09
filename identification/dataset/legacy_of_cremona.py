import os
import glob
import pandas as pd
import re

data = []
folder = '/home/hugo/Thèse/Data/The Legacy of Cremona/'
pattern = re.compile(r"^Beethoven.*Violin by ([^\[\.]+).*\.mp3$")

for filename in os.listdir(folder):
    if filename.endswith(".mp3") and pattern.match(filename):
        violinmaker = pattern.match(filename).group(1)
        data.append({
            'file': folder + filename,
            'violin': violinmaker
        })

df = pd.DataFrame(data)
print(df)
df.to_pickle('data/processed/dataset_cremona.pkl')