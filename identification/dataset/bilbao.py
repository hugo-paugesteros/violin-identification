import os
import glob
import pandas as pd

data = []

# Bilbao Dataset (2019)
for player in range(1,27):
    for violin in range(1,14):
        
        # Scale
        file = f'/home/hugo/Thèse/Data/Bilbao/2019/Violin_scale_segementation/PLAYER{player}/Violin{violin}.wav'
        if(os.path.exists(file)):
            data.append({
                'file': file,
                'player': player,
                'violin': violin,
                'type': 'scale'
            })

    # Free
    files = glob.glob(f'/home/hugo/Thèse/Data/Bilbao/2019/Recordings_FreeCat/PLAYER{player}/cut_player{player}*violin*.wav')
    if files:
        data.append({
            'file': files[0],
            'player': player,
            'violin': int(files[0][-6:-4]) if files[0][-6:-4].isdigit() else int(files[0][-5]),
            'type': 'free'
        })

# Villefavard 2024 Dataset
for violin in [1, 4, 5, 9, 11, 13]:
    files = glob.glob(f'/home/hugo/Thèse/Data/Bilbao/2024 - Villefavard/Violin {violin}/*.mp3')
    for file in files:
        filename = os.path.basename(file)[:-4]
        if filename == 'menno':
            player = 30
        elif filename == 'casper':
            player = 31
        elif filename == 'ellin':
            player = 32
        else:
            player = 99
        data.append({
            'file': file,
            'player': player,
            'violin': violin,
            'type': 'villefavard'
        })

# D'alembert studio july 2024
for violin in [4, 5]:
    file = f'/home/hugo/Thèse/Data/Bilbao/2024 - D\'Alembert/{violin}.mp3'
    data.append({
        'file': file,
        'player': 99,
        'violin': violin,
        'type': 'dalembert'
    })

df = pd.DataFrame(data)
df.to_pickle('data/processed/dataset_bilbao.pkl')