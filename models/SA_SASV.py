import torch
import speechbrain as sb
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain import Stage
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.RNN import GRU
from speechbrain.lobes.models.ECAPA_TDNN import Res2NetBlock
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.linear import Linear
from tqdm.contrib import tqdm
from torch.nn import MaxPool1d
from models.BinaryMetricStats import BinaryMetricStats
import torch.nn.functional as F
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d


class SA_SASV(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, mag, fbanks, mfccs, lens = self.prepare_features(batch.sig,stage)
        wavs = torch.unsqueeze(wavs, 2)
        stack_features = []
        stack_features.append(torch.unsqueeze(self.modules.raw_encoder(wavs, lens),2))
        stack_features.append(self.modules.fbanks_encoder(fbanks, lens))
        enc_output = torch.cat(tuple(stack_features), dim=1)
        enc_output = self.modules.batch_norm(enc_output)
        enc_output = self.modules.conv_1d(enc_output)
        enc_output = enc_output.transpose(1, 2)
        asv_output = self.modules.classifier_asv(enc_output)
        cm_output = self.modules.classifier_cm(enc_output)
        tts_output = self.modules.discriminator_tts(enc_output)
        vc_output = self.modules.discriminator_vc(enc_output)

        return (asv_output, cm_output, tts_output, vc_output, enc_output)

    def prepare_features(self, wavs, stage):
        wavs, lens = wavs
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        mag, fbanks, mfccs = self.modules.mfcc(wavs)

        if fbanks!=None: fbanks = self.modules.mean_var_norm(fbanks, lens)
        if mag!=None:  mag = self.modules.mean_var_norm(mag, lens)
        if mfccs!=None:  mfccs = self.modules.mean_var_norm(mfccs, lens)

        return wavs, mag, fbanks, mfccs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        _, lens = batch.sig
        bonafide_encoded, _ = batch.bonafide_encoded
        speaker_encoded, _ = batch.speaker_encoded
        tts_encoded, _ = batch.tts_encoded
        vc_encoded, _ = batch.vc_encoded
        triplet_encoded, _ = batch.triplet_encoded


        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            speaker_encoded = torch.cat([speaker_encoded, speaker_encoded], dim=0)
            bonafide_encoded = torch.cat([bonafide_encoded, bonafide_encoded], dim = 0)
            tts_encoded = torch.cat([tts_encoded, tts_encoded], dim=0)
            vc_encoded = torch.cat([vc_encoded, vc_encoded], dim=0)
            triplet_encoded = torch.cat([triplet_encoded, triplet_encoded], dim=0)
            lens = torch.cat([lens, lens])

        asv_output, cm_output, tts_output, vc_output, encode_output = predictions

        loss = self.hparams.loss_metric(self.hparams.asv_loss_metric,
                                        asv_output,
                                        cm_output,
                                        tts_output,
                                        vc_output,
                                        speaker_encoded,
                                        bonafide_encoded,

                                        tts_encoded,
                                        vc_encoded,
                                        )
        triplet_loss = self.modules.triplet_loss_metric(torch.squeeze(encode_output,1),
                                                        torch.squeeze(triplet_encoded,1)
                                                        )
        loss +=  triplet_loss


        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:

            self.error_metrics.append(batch.id, cm_output, bonafide_encoded)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = BinaryMetricStats(
                positive_label=1,
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            print('before anchor: %f' % (self.hparams.lr_scheduler.anchor))
            old_lr, new_lr = self.hparams.lr_scheduler([self.optimizer],
                                                       current_epoch = epoch,
                                                       current_loss = stage_loss)
            print('patient counter: %d'%(self.hparams.lr_scheduler.patience_counter))
            print('after anchor: %f'%(self.hparams.lr_scheduler.anchor))

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats,
                                                 num_to_keep=5,
                                                 keep_recent=True)

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def evaluate_batch(self, batch, stage):
        """
        Overwrite evaluate_batch.
        Keep same for stage in (TRAIN, VALID)
        Output probability in TEST stage (from classify_batch)
        """

        if stage != sb.Stage.TEST:

            # Same as before
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
            return loss.detach().cpu()
        else:
            asv_output, cm_output, tts_output, vc_output, enc_output = self.compute_forward(batch, stage=stage)
            cm_output = cm_output.squeeze(1)
            return enc_output, cm_output

    def evaluate(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        """
        Overwrite evaluate() function so that it can output score file
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0

        """
        added here
        """
        cm_dict = {}
        embedding_dict = {}

        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                """
                Rewrite here
                """
                enc_output, cm_output  = self.evaluate_batch(batch, stage=Stage.TEST)
                enc_output = enc_output.unsqueeze(1)
                cm_scores = [cm_output[i].item()
                             for i in range(cm_output.shape[0])]
                for i, seg_id in enumerate(batch.id):
                    cm_dict[seg_id] = cm_scores[i]
                    embedding_dict[seg_id] = enc_output[i].detach().clone()



                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

        self.step = 0
        return cm_dict, embedding_dict

class RawEncoder(torch.nn.Module):
    def __init__(self,
                 device="cpu",
                 activation=torch.nn.LeakyReLU,
                 ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                Conv1d(
                    in_channels=1,
                    out_channels=128,
                    kernel_size=3,
                    # dilation=3,
                    stride = 3
                ),
                BatchNorm1d(input_size=128),
                activation(),
            ]
        )
        for _ in range(3):
            self.blocks.extend(
                [ Res2NetBlock(128, 128,  scale=1,
                               # dilation=1,
                               # kernel_size=3
                               ),
                  BatchNorm1d(input_size=128),
                  activation(),
                  Res2NetBlock(128, 128, scale=1,
                               # dilation=1,
                               # kernel_size=3
                               ),
                  BatchNorm1d(input_size=128),
                  activation(),
                  MaxPool1d(3)
                  ]
            )


        self.gru = GRU(
            512,
            input_size= 128,
            dropout= 0.3,
            bias=True,
            # num_layers=2,
        )
        self.linear = Linear(
            input_size=256,
            n_neurons=128,
            bias=True,
            combine_dims=False,
        )



    def forward(self, x, lens=None):
        """Returns the x-vectors.
        Arguments
        ---------
        x : torch.Tensor
        """
        # x = x.transpose(1, 2)
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                if isinstance(layer,MaxPool1d):
                    x = layer(x.permute(0,2,1)).permute(0,2,1)
                else:
                    x = layer(x)
        # x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        # x = self.linear(x)
        return x

class ASV_Decoder(torch.nn.Module):

    def __init__(
            self,
            input_size,
            device="cpu",
            lin_blocks=0,
            lin_neurons=192,
            out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)

class TTS_Discriminator(torch.nn.Module):
    def __init__(
            self,
            input_size,
            activation=torch.nn.LeakyReLU,
            lin_blocks=1,
            lin_neurons=512,
            out_neurons=1211,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                    activation(),
                    _BatchNorm1d(input_size=lin_neurons),
                ]
            )
            input_size = lin_neurons

        self.blocks.extend(
            [
                Linear(input_size=input_size, n_neurons=out_neurons),
                sb.nnet.activations.Softmax(apply_log=True)
            ]
        )

    def forward(self, x):
        # x = self.grl_layer(x)
        lam = torch.tensor(1.0)
        x = GRL.apply(x,lam)

        for layer in self.blocks:
            x = layer(x)

        return x
        # Need to be normalized
        # x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        # return x.unsqueeze(1)

class VC_Discriminator(torch.nn.Module):
    def __init__(
            self,
            input_size,
            activation=torch.nn.LeakyReLU,
            lin_blocks=1,
            lin_neurons=512,
            out_neurons=1211,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                    activation(),
                    _BatchNorm1d(input_size=lin_neurons),
                ]
            )
            input_size = lin_neurons

        self.blocks.extend(
            [
                Linear(input_size=input_size, n_neurons=out_neurons),
                sb.nnet.activations.Softmax(apply_log=True)
            ]
        )

    def forward(self, x):
        lam = torch.tensor(1.0)
        x = GRL.apply(x,lam)
        for layer in self.blocks:
            x = layer(x)
        return x

class CM_Decoder(sb.nnet.containers.Sequential):
    def __init__(
            self,
            input_shape,
            activation=torch.nn.LeakyReLU,
            lin_blocks=1,
            lin_neurons=512,
            out_neurons=1,
    ):
        super().__init__(input_shape=input_shape)

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            # Added here
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )
            self.DNN[block_name].append(activation(), layer_name="act")
        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )

class GRL(torch.autograd.Function):
    def __init__(self):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 1000  # be same to the max_iter of config.py

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)


    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

