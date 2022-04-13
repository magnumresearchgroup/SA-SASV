import speechbrain as sb
import os
import json

LABEL_DIR = 'processed_sasv_data'
SPEAKER_ID_LIST = 'speaker_id_list.json'
TTS_ID_LIST = 'tts_id_list.json'
VC_ID_LIST = 'vc_id_list.json'
TRIPLET_ID_LIST = 'triplet_id_list.json'
def get_dataset(hparams):
    """
    Code here is basically same with code in SpoofSpeechDataset.py
    However, audio will not be load directly.
    A random compression will be made before load by torchaudio
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder_cm = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_asv = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_tts = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_vc = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_triplet = sb.dataio.encoder.CategoricalEncoder()


    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = sb.dataio.dataio.read_audio(file_path)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("speaker_id")
    @sb.utils.data_pipeline.provides("speaker_id", "speaker_encoded")
    def speaker_label_pipeline(speaker_id):
        yield speaker_id
        speaker_encoded = label_encoder_asv.encode_label_torch(speaker_id, True)
        yield speaker_encoded

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("bonafide")
    @sb.utils.data_pipeline.provides("bonafide", "bonafide_encoded")
    def bonafide_label_pipeline(bonafide):
        yield bonafide
        bonafide_encoded = label_encoder_cm.encode_label_torch(bonafide, True)
        yield bonafide_encoded

    @sb.utils.data_pipeline.takes("spoof_id")
    @sb.utils.data_pipeline.provides("spoof_id", "tts_encoded", "vc_encoded")
    def spoof_label_pipeline(spoof_id):
        yield spoof_id
        # spoof_encoded = label_encoder_spoof.encode_label_torch(spoof_id, True)
        tts_encoded = label_encoder_tts.encode_label_torch(spoof_id, True)
        vc_encoded = label_encoder_vc.encode_label_torch(spoof_id, True)
        # yield spoof_encoded
        yield tts_encoded
        yield vc_encoded

    @sb.utils.data_pipeline.takes("spoof_id","speaker_id")
    @sb.utils.data_pipeline.provides("triplet_encoded")
    def triplet_label_pipeline(spoof_id, speaker_id):
        if spoof_id in ["A01", "A02", "A03", "A04"]:
            triplet_id = "TTS"
        elif spoof_id in ["A05", "A06"]:
            triplet_id = "VC"
        else:
            triplet_id = speaker_id
        triplet_encoded = label_encoder_triplet.encode_label_torch(triplet_id, True)
        yield triplet_encoded



    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}

    for dataset in ["train", "dev", "eval"]:
        # print(hparams[f"{dataset}_annotation"])
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                           audio_pipeline,
                           speaker_label_pipeline,
                           bonafide_label_pipeline,
                           spoof_label_pipeline,
                           triplet_label_pipeline
                           ],
            output_keys=["id", "sig",
                         "spoof_id",
                         "bonafide", "bonafide_encoded",
                         "speaker_id", "speaker_encoded",
                         "tts_encoded", "vc_encoded",
                         "triplet_encoded"
                         ],
        )

    def load_create_label_encoder(label_enc_file,
                                  label_encoder,
                                  label_list_file = None,
                                  ):
        label_list = None
        if label_list_file == None:
            label_list = [("spoof", "bonafide")]
        else:
            with open(os.path.join(LABEL_DIR, label_list_file), 'r') as f:
                label_list = [tuple(json.load(f))]
        lab_enc_file = os.path.join(hparams["save_folder"], label_enc_file)
        label_encoder.load_or_create(
            path=lab_enc_file,
            sequence_input=False,
            from_iterables=label_list,
        )



    load_create_label_encoder(label_enc_file="label_encoder_speaker.txt",
                              label_encoder=label_encoder_asv,
                              label_list_file=SPEAKER_ID_LIST
                              )
    label_encoder_asv.add_unk()

    load_create_label_encoder(label_enc_file="label_encoder_tts.txt",
                              label_encoder=label_encoder_tts,
                              label_list_file=TTS_ID_LIST,
                              )
    label_encoder_tts.add_unk()

    load_create_label_encoder(label_enc_file="label_encoder_vc.txt",
                              label_encoder=label_encoder_vc,
                              label_list_file=VC_ID_LIST,
                              )
    label_encoder_vc.add_unk()

    load_create_label_encoder(label_enc_file="label_encoder_triplet.txt",
                              label_encoder=label_encoder_triplet,
                              label_list_file=TRIPLET_ID_LIST,
                              )

    load_create_label_encoder(label_enc_file="label_encoder_cm.txt",
                              label_encoder=label_encoder_cm,

                              )


    return datasets