from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('sasv_results/SASASV/1986/all_2d.csv')
spk = pd.read_csv('sasv_results/SASASV/1986/spk_2d.csv')
spoof_proto_file = '../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

with open(spoof_proto_file, 'r') as f:
    spoof_proto = f.readlines()
spoof_id_dict = {}
for line in spoof_proto:
    line = line.split()
    spoof_id_dict[line[1]] = line[3]


data = []

for index, row in df.iterrows():
    cur = []
    cur.append(row['x'])
    cur.append(row['y'])
    if row['label'] != 'spoof':
        cur.append('bonafide')
    else:
        spoof_id = spoof_id_dict[row['utt_id']]
        if spoof_id in ['A05','A06','A17','A18','A19']:
            cur.append('vc')
        else:
            cur.append('tts')
    data.append(cur)
data = pd.DataFrame(data, columns=['x','y','label'])
ax =sns.scatterplot(x='x', y='y', data=data, hue='label',s=3)
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.show()
plt.close()
