import os
import glob
import pandas as pd

data = []
for player in range(1,27):
    for violin in range(1,14):
        
        # Scale
        file = f'/home/hugo/Thèse/Data/SegmentationPerViolin/PLAYER{player}/Violin{violin}.wav'
        if(os.path.exists(file)):
            data.append({
                'file': file,
                'player': player,
                'violin': violin,
                'type': 'scale'
            })

    # Free
    files = glob.glob(f'/home/hugo/Thèse/Data/BilbaoViolins/Recordings_FreeCat/PLAYER{player}/cut_player{player}*violin*.wav')
    if files:
        data.append({
            'file': files[0],
            'player': player,
            'violin': int(files[0][-6:-4]) if files[0][-6:-4].isdigit() else int(files[0][-5]),
            'type': 'free'
        })

df = pd.DataFrame(data)
df.to_pickle('recordings.pkl')