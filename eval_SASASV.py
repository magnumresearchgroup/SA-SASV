import collections
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import speechbrain as sb
from sklearn.manifold import TSNE
from hyperpyyaml import load_hyperpyyaml
from models.SA_SASV import SA_SASV
from datasets.SASASV_eval_dataset import get_asv_eval_dataset
from tqdm.contrib import tqdm
import logging
from pathlib import Path

DATA_DIR = 'processed_sasv_data'

TRL_FILE = 'eval_trl.json'
ENROLL_FILE = 'eval_enroll_speaker_list.json'
DEV_TRL_FILE = 'dev_trl.json'
DEV_ENROLL_FILE = 'dev_enroll_speaker_list.json'


OUTPUT_DIR = 'predictions'
PREDS_FILE = 'sasv_preds.json'
KEYS_FILE =  'sasv_keys.json'
CM_PREDS_FILE = 'cm_preds.json'

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def compute_embedding_loop(encoder, data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}
    cm_scores_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            _, cm_output, _, _, embeddings = encoder.compute_forward(batch, sb.Stage.TEST)
            embeddings = embeddings.unsqueeze(1)
            cm_output = cm_output.squeeze(1)
            cm_scores = [cm_output[i].item()
                         for i in range(cm_output.shape[0])]

            seg_ids = batch.id
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = embeddings[i].detach().clone()
                cm_scores_dict[seg_id] = cm_scores[i]

    return embedding_dict, cm_scores_dict

def get_verification_scores(test_data, enroll_data):
    """ Computes positive and negative scores given the verification split.
    """
    asv_preds = []
    cm_preds = []
    submissions = []
    keys = []

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    for utt in test_data:
        speaker_id, utt_id, cm_id, target = utt.split()
        enrol_utt_list = enroll_data[speaker_id]

        test_emb = test_dict[utt_id]
        enrol_emb_list = [enrol_dict[enrol_id] for enrol_id in  enrol_utt_list]

        scores = [similarity(test_emb, enrol_emb)[0] for enrol_emb in enrol_emb_list]
        cur_score = sum(scores)/len(scores)

        asv_preds.append(cur_score.cpu().data.numpy()[0].item())
        keys.append(target)
        if cm_scores_dict[utt_id]<0.5:
            cur_score = 0
        submissions.append('%s %s %s %s %f'%(speaker_id,
                                             utt_id,
                                             cm_id,
                                             target,
                                             cur_score,
                                             ))

        cm_preds.append(cm_scores_dict[utt_id])

    with open(os.path.join(OUTPUT_DIR, PREDS_FILE),'w') as f:
        json.dump(asv_preds, f)
    with open(os.path.join(OUTPUT_DIR, KEYS_FILE), 'w') as f:
        json.dump(keys, f)
    with open(os.path.join(OUTPUT_DIR, CM_PREDS_FILE),'w') as f:
        json.dump(cm_preds, f)
    with open(os.path.join(OUTPUT_DIR,SUBMISSION_FILE), 'w') as f:
        for v in submissions:
            f.write(v+'\n')
    print('PREDICTION FINISHED!!!')




if __name__ == "__main__":
    VISUALIZE = True
    SCORE = True
    DEV = False
    if DEV:
        SUBMISSION_FILE = 'dev_submission.txt'
    else:
        SUBMISSION_FILE = 'eval_submission.txt'

    logger = logging.getLogger(__name__)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams_file = os.path.join(hparams['output_folder'], 'hyperparams.yaml')
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams['batch_size'] = 16
    hparams['dataloader_options']['batch_size'] = 16

    datasets = get_asv_eval_dataset()

    encoder = SA_SASV(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )



    test_dic_file= os.path.join(hparams['output_folder'],'test_dict.pt')
    cm_scores_dict_file  =  os.path.join(hparams['output_folder'],'cm_scores_dict.pt')
    enrol_dic_file = os.path.join(hparams['output_folder'],'enrol_dict.pt')

    if os.path.exists(test_dic_file):
        cm_scores_dict = torch.load(cm_scores_dict_file)
        test_dict = torch.load(test_dic_file)
        enrol_dict = torch.load(enrol_dic_file)
    else:
        if DEV:
            _, enrol_dict = encoder.evaluate(
                test_set=datasets["dev_enrol"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["dataloader_options"],
            )

            cm_scores_dict, test_dict = encoder.evaluate(
                test_set=datasets["dev_trl"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["dataloader_options"],
            )
        else:


            cm_scores_dict, test_dict = encoder.evaluate(
                test_set=datasets["trl"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["dataloader_options"],
            )

            _, enrol_dict = encoder.evaluate(
                test_set=datasets["enrol"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["dataloader_options"],
            )

        torch.save(cm_scores_dict, cm_scores_dict_file)
        torch.save(test_dict, test_dic_file)
        torch.save(enrol_dict, enrol_dic_file)
    if DEV:
        with open('../data/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt','r') as f:
            test_data = f.readlines()
        with open(os.path.join(DATA_DIR, DEV_ENROLL_FILE)) as f:
            enroll_data = json.load(f)
    else:
        with open('../data/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt','r') as f:
            test_data = f.readlines()
        with open(os.path.join(DATA_DIR, ENROLL_FILE)) as f:
            enroll_data = json.load(f)
    if SCORE:
        get_verification_scores(test_data, enroll_data)
    if VISUALIZE:
        if os.path.exists(os.path.join(hparams['output_folder'], '2d_all.csv')):
            df = pd.read_csv(os.path.join(hparams['output_folder'], '2d_all.csv'))
        else:

            for utt_id in test_dict:
                test_dict[utt_id] = torch.squeeze(test_dict[utt_id],0)
                test_dict[utt_id] = torch.squeeze(test_dict[utt_id],0)
                test_dict[utt_id] = test_dict[utt_id].cpu().numpy()

            # process all data
            def reduce_dimension(target_dict, output_name):
                X = list(target_dict.values())
                X = TSNE(n_components=2,perplexity=40, init='random').fit_transform(
                    X
                )
                utts = list(target_dict.keys())
                arr = []
                for i in range(len(utts)):
                    utt_id = utts[i]
                    if gt_data[utt_id]['cm_id'] == 'bonafide':
                        arr.append(np.append(X[i], [gt_data[utt_id]['speaker_id'], utt_id ]  ))
                    else:
                        arr.append(np.append(X[i], ['spoof', utt_id] ))
                df = pd.DataFrame(arr, columns=['x','y','label','utt_id'])
                df.to_csv(os.path.join(hparams['output_folder'], output_name))

            gt_data = collections.defaultdict(dict)
            for utt in test_data:
                speaker_id, utt_id, cm_id, target = utt.split()
                gt_data[utt_id]['speaker_id'] = speaker_id
                gt_data[utt_id]['cm_id'] = cm_id

            bonafide_dict = {}
            for utt_id in test_dict:
                if gt_data[utt_id]['cm_id'] == 'bonafide':
                    bonafide_dict[utt_id] = test_dict[utt_id]

            reduce_dimension(test_dict, 'all_2d.csv')
            reduce_dimension(bonafide_dict, 'spk_2d.csv')


    del enrol_dict, test_dict

