import os
import json
from speechbrain.dataio.dataio import read_audio

SAMPLERATE = 16000

def get_enrol_protocols(pro_dir = '../data/LA/ASVspoof2019_LA_asv_protocols',
                     enrol_files = ('ASVspoof2019.LA.asv.eval.female.trn.txt',
                                  'ASVspoof2019.LA.asv.eval.male.trn.txt')
                     ):
    enroll_list = {}
    enroll = {}
    data_dir = '../data/LA/ASVspoof2019_LA_eval/flac'
    save_dir = 'processed_sasv_data'
    enroll_list_save_file = 'eval_enroll_speaker_list.json'
    enroll_save_file = 'eval_enroll.json'
    for file in enrol_files:
        with open(os.path.join(pro_dir, file), 'r') as f:
            enrol_pros = f.readlines()
        for pro in enrol_pros:
            pro = pro.split()
            speaker_id = pro[0]
            utt_ids = pro[1].split(',')
            enroll_list[speaker_id] = utt_ids
            for utt_id in utt_ids:
                enroll[utt_id] = {}
                enroll[utt_id]['file_path'] = os.path.join(data_dir, '%s.flac'%(utt_id))
                enroll[utt_id]['speaker_id'] = speaker_id
                signal = read_audio(enroll[utt_id]['file_path'])
                enroll[utt_id]['duration'] = signal.shape[0] / SAMPLERATE
    with open(os.path.join(save_dir, enroll_save_file), 'w') as f:
        json.dump(enroll, f)
    with open(os.path.join(save_dir, enroll_list_save_file), 'w') as f:
        json.dump(enroll_list,f)

def get_eval_protocols(pro_dir = '../data/LA/ASVspoof2019_LA_asv_protocols',
                       eval_file = 'ASVspoof2019.LA.asv.eval.gi.trl.txt'
                       ):
    data_dir = '../data/LA/ASVspoof2019_LA_eval/flac'
    save_dir = 'processed_sasv_data'
    save_file = 'eval_trl.json'
    eval_features = {}
    with open(os.path.join(pro_dir, eval_file), 'r') as f:
        test_pros = f.readlines()

    for pro in test_pros:
        pro = pro.split()
        claimed_speaker = pro[0]
        utt_id = pro[1]
        cm_id = pro[2]
        target_id = pro[3]

        utt_file_path = os.path.join(data_dir, '%s.flac'%(utt_id))
        signal = read_audio(utt_file_path)
        duration = signal.shape[0] / SAMPLERATE

        eval_features[utt_id] = {
            'speaker_id' : claimed_speaker,
            'cm_id': cm_id,
            'target_id': target_id,
            'file_path': utt_file_path,
            'duration': duration,
        }
    with open(os.path.join(save_dir, save_file), 'w') as f:
        json.dump(eval_features, f)


