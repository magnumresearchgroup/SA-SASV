import os
import random
import json
import glob
from pathlib import Path
from speechbrain.dataio.dataio import read_audio
from datasets.preprocess_enroll_data import get_enrol_protocols, get_eval_protocols


CM_PROTO_DIR = '../data/LA/ASVspoof2019_LA_cm_protocols'
CM_TRAIN_FILE = 'ASVspoof2019.LA.cm.train.trn.txt'
CM_DEV_FILE = 'ASVspoof2019.LA.cm.dev.trl.txt'

CM_FLAC_DIR = '../data/LA/ASVspoof2019_LA_%s/flac'

CM_EVAL_FILE = 'ASVspoof2019.LA.cm.eval.trl.txt'

PROCESSED_DATA_DIR = 'processed_sasv_data'
CM_SB_TRAIN_FILE = 'cm_sb_train.json'
CM_SB_DEV_FILE = 'cm_sb_dev.json'
CM_SB_EVAL_FILE = 'cm_sb_eval.json'


BONAFIDE = 'bonafide'
SPOOF = 'spoof'

# Statistics
SPOOF_TRAIN_PERCENT = 95
SPOOF_DEV_PERCENT = 5
RANDOM_SEED = 97271
SAMPLERATE = 16000
SPLIT = ['train', 'dev', 'eval']

random.seed(RANDOM_SEED)
Path(PROCESSED_DATA_DIR).mkdir(exist_ok=True)

#Check if we already have SpeechBrain format CM protocol
if os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_TRAIN_FILE)) and \
    os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_DEV_FILE)) and \
        os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_EVAL_FILE)):
    print('SpeechBrain format CM protocols exist...')
else:
    print('Start to convert original CM protocols to SpeechBrain format...')
    # Read CM protocols in train/dev/eval set
    save_files = [CM_SB_TRAIN_FILE, CM_SB_DEV_FILE, CM_SB_EVAL_FILE]
    for i, file in enumerate([CM_TRAIN_FILE, CM_DEV_FILE, CM_EVAL_FILE]):
        cm_features = {}
        with open(os.path.join(CM_PROTO_DIR, file), 'r') as f:
            cm_pros = f.readlines()
            print('% has %d data!'%(file, len(cm_pros)))
        for pro in cm_pros:
            pro = pro.strip('\n').split(' ')
            speaker_id = pro[0]
            auto_file_name = pro[1]
            spoof_id = pro[3]
            bonafide = pro[4]
            cm_features[auto_file_name] = {
                'speaker_id': speaker_id,
                'spoof_id': spoof_id,
                'bonafide': bonafide,
                'source': 'asv_spoof',
                'start': '-',
                'stop': '-',
            }
        # Read flac files and durations
        cur_flac_files = glob.glob(os.path.join( CM_FLAC_DIR%(SPLIT[i]),'*.flac'),
                                   recursive=True)
        n_miss = 0
        # Read each utt file and get its duration. Update cm features
        for file in cur_flac_files:
            signal = read_audio(file)
            duration = signal.shape[0] / SAMPLERATE
            utt_id = Path(file).stem
            if utt_id in cm_features:
                cm_features[utt_id]['file_path'] = file
                cm_features[utt_id]['duration'] = duration
            else: n_miss += 1
        print('%d files missed description in protocol file in %s set'%(n_miss, SPLIT[i]))
        # Save updated cm features into json
        with open(os.path.join(PROCESSED_DATA_DIR, save_files[i]), 'w') as f:
            json.dump(cm_features, f)

#Read SB format CM protocols
with open(os.path.join(PROCESSED_DATA_DIR, CM_SB_TRAIN_FILE), 'r') as f:
    cm_train = json.load(f)
    print('CM train protocols has %d data'%(len(cm_train)))
with open(os.path.join(PROCESSED_DATA_DIR, CM_SB_DEV_FILE), 'r') as f:
    cm_dev = json.load(f)
    print('CM dev protocols has %d data'%(len(cm_dev)))
with open(os.path.join(PROCESSED_DATA_DIR, CM_SB_EVAL_FILE), 'r') as f:
    cm_eval = json.load(f)
    print('CM eval protocols has %d data'%(len(cm_eval)))

all_train = {**cm_train, **cm_dev}
all_utt_ids = list(all_train.keys())
merge_train = {}
merge_dev = {}
random.shuffle(all_utt_ids)
break_point = int(len(all_utt_ids)*SPOOF_TRAIN_PERCENT*0.01)
for i in all_utt_ids[ : break_point]:
    merge_train[i] = all_train[i]
for i in all_utt_ids[break_point:]:
    merge_dev[i] = all_train[i]

with open(os.path.join(PROCESSED_DATA_DIR, 'merge_train.json'),'w') as f:
    json.dump(merge_train, f)
with open(os.path.join(PROCESSED_DATA_DIR, 'merge_dev.json'),'w') as f:
    json.dump(merge_dev, f)


with open(os.path.join(PROCESSED_DATA_DIR, 'tts_id_list.json'), 'w') as f:
    json.dump(["A01", "A02", "A03", "A04"], f)
with open(os.path.join(PROCESSED_DATA_DIR, 'vc_id_list.json'), 'w') as f:
    json.dump(["A05", "A06"], f)
speaker_id = set()
for utt_id in all_train:
    speaker_id.add(all_train[utt_id]['speaker_id'])
speaker_id = sorted(list(speaker_id))
with open(os.path.join(PROCESSED_DATA_DIR, 'speaker_id_list.json'), 'w') as f:
    json.dump(speaker_id, f)
print("Speaker ID list created")
triplet_id = ['TTS', 'VC']+speaker_id
with open(os.path.join(PROCESSED_DATA_DIR, 'triplet_id_list.json'), 'w') as f:
    json.dump(triplet_id, f)

get_enrol_protocols()
get_eval_protocols()



