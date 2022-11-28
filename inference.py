import warnings, librosa
from librosa.core import yin
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import utils
from data import remove_accent
from model import train_audio_transforms, AcousticModel, BoundaryDetection
import json
import argparse
from lib import nets
from lib import spec_utils
from lib import dataset
import json

np.random.seed(7)

def json2txt(json_path, save_path):
  gt = open(json_path)
  gt_json = json.load(gt)
  words = []
  with open(save_path, 'w', encoding="utf-8") as f:
    for line in gt_json:
      words.extend([word['d'] for word in line['l']])
      f.write(' '.join([word['d'] for word in line['l']])+' ')
  return save_path

def preprocess_from_file(audio_file, model_vocal, lyrics_file, word_file=None):
    y, sr = preprocess_audio(audio_file, model_vocal)

    words, lyrics_p, idx_word_p, idx_line_p = preprocess_lyrics(lyrics_file, word_file)

    return y, words, lyrics_p, idx_word_p, idx_line_p

def main(args):
    ls_path_audio = "./data/songs"
    # ls_path_lyrics = args.ls_path_lyrics
    # ls_json_lyrics = args.ls_json_lyrics

    # convert json to txt
    ls_json_lyrics="./data/new_labels_json"
    ls_path_lyrics="./data/new_labels_txt"
    resolution = 256 / 22050 * 3

    if not os.path.isdir(ls_path_lyrics):
      os.mkdir(ls_path_lyrics)
    ls_path = os.listdir(ls_json_lyrics)
    for i, path in enumerate(ls_path):
      _ = json2txt(os.path.join(ls_json_lyrics, path), \
      os.path.join(ls_path_lyrics, path.replace(".json", ".txt")))

    ckp_path = args.ckp_path
    save_folder = args.save_folder
    cuda = True
    method="Baseline"
    n_fft = 2048

    # constants
    resolution = 256 / 22050 * 3
    alpha = 0.8

    # decode method
    if "BDR" in method:
        model_type = method[:-4]
        bdr_flag = True
    else:
        model_type = method
        bdr_flag = False
    bdr_flag = True
    print("Model: {} BDR?: {}".format(model_type, bdr_flag))

    # prepare acoustic model params
    if model_type == "Baseline":
        n_class = 41
    elif model_type == "MTL":
        n_class = (41, 47)
    else:
        ValueError("Invalid model type.")

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 3,
        "rnn_dim": 256,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    device = 'cuda' if (cuda and torch.cuda.is_available()) else 'cpu'
    ac_model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'], \
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    print("Loading remove vocals models...")
    pretrained_model = "./checkpoints/baseline.pth"
    model_vocal = nets.CascadedNet(n_fft, 32, 128)
    model_vocal.load_state_dict(torch.load(pretrained_model, map_location=device))
    model_vocal.to(device)

    print("Loading acoustic model from checkpoint...")
    # state = utils.load_model(ac_model, "./checkpoints/checkpoint_{}".format(model_type), cuda=(device=="gpu"))
    state = utils.load_model(ac_model, ckp_path, cuda=(device=="gpu"))
    ac_model.eval()
    for i, path in enumerate(ls_path):
      # start timer
      t_s = time()
      print(f"Computing phoneme posteriorgram {path, i}...")
      if os.path.isfile(save_folder + path.replace(".txt", ".json")):
        print(f"skip this file, {path} is exist")
        continue
      path = path.replace("json", "txt")
      if path == "txt_lyrics":
        continue
      audio_file =  os.path.join(ls_path_audio, path.replace(".txt", ".wav"))
      lyrics_file = os.path.join(ls_path_lyrics, path)
      
      audio, words, lyrics_p, idx_word_p, idx_line_p = preprocess_from_file(audio_file, model_vocal, lyrics_file, word_file=None)
      # reshape input, prepare mel
      x = audio.reshape(1, 1, -1)
      x = utils.move_data_to_device(x, device)
      x = x.squeeze(0)
      x = x.squeeze(1)
      x = train_audio_transforms.to(device)(x)
      x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1)
      # predict
      all_outputs = ac_model(x)
      if model_type == "MTL":
          all_outputs = torch.sum(all_outputs, dim=3)

      all_outputs = F.log_softmax(all_outputs, dim=2)

      batch_num, output_length, num_classes = all_outputs.shape
      song_pred = all_outputs.data.cpu().numpy().reshape(-1, num_classes)  # total_length, num_classes
      total_length = int(audio.shape[1] / 22050 // resolution)
      song_pred = song_pred[:total_length, :]

      # smoothing
      P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
      song_pred = np.log(np.exp(song_pred) + P_noise)

      if bdr_flag:
          # boundary model: fixed
          bdr_hparams = {
              "n_cnn_layers": 1,
              "rnn_dim": 32,  # a smaller rnn dim than acoustic model
              "n_class": 1,  # binary classification
              "n_feats": 32,
              "stride": 1,
              "dropout": 0.1,
          }

          bdr_model = BoundaryDetection(
              bdr_hparams['n_cnn_layers'], bdr_hparams['rnn_dim'], bdr_hparams['n_class'],
              bdr_hparams['n_feats'], bdr_hparams['stride'], bdr_hparams['dropout']
          ).to(device)
          print("Loading BDR model from checkpoint...")
          state = utils.load_model(bdr_model, "./checkpoints/checkpoint_BDR", cuda=(device == "gpu"))
          bdr_model.eval()

          print("Computing boundary probability curve...")
          # get boundary prob curve
          bdr_outputs = bdr_model(x).data.cpu().numpy().reshape(-1)
          # apply log
          bdr_outputs = np.log(bdr_outputs) * alpha

          line_start = [d[0] for d in idx_line_p]

          # start alignment
          print("Aligning...It might take a few minutes...")
          word_align, score = utils.alignment_bdr(song_pred, lyrics_p, idx_word_p, bdr_outputs, line_start)
      else:
          # start alignment
          word_align, score = utils.alignment(song_pred, lyrics_p, idx_word_p)

      t_end = time() - t_s
      print("Alignment Score:\t{}\tTime:\t{}".format(score, t_end))
      # Write json for submission
      json_lyrics = json.load(open(os.path.join(ls_json_lyrics, path.replace(".txt", ".json"))))
      id = 0
      lines_arr = []
      for line in json_lyrics:
        for word in line['l']:
          # If words having more than one word
          # if len(word['d'].strip().split(' ')) > 1:
          #   print(word['d'], "=====================")
          #   word['s'] = int(word_align[id][0]*1000*resolution)
          #   word['e'] = int(word_align[id+len(word['d'].strip().split(' '))-1][1]*1000*resolution)
          #   id += len(word['d'].strip().split(' '))
          # else:
          #   # if word in csv == word in json format
          #   try:
          #     if remove_accent(word['d'].lower().strip()) == str(words[id]):
          #       word['s'] = int(word_align[id][0]*1000*resolution)
          #       word['e'] = int(word_align[id][1]*1000*resolution)
          #       id += 1
          #   except:
          #     print(remove_accent(word['d'].lower().strip().replace(",", '').replace('"', "")))
          #     continue
          
          # if word in csv == word in json format
          try:
            if remove_accent(word['d'].lower().strip()) == str(words[id]):
              word['s'] = int(word_align[id][0]*1000*resolution)
              word['e'] = int(word_align[id][1]*1000*resolution)
              id += 1
          except:
            print(remove_accent(word['d'].lower().strip().replace(",", '').replace('"', "")))
            continue
        line['s'] = line["l"][0]["s"]
        line['e'] = line["l"][-1]["e"]

        lines_arr.append(line)
      
      # Saving...
      json_object = json.dumps(lines_arr, indent=4, ensure_ascii=False)
      if not os.path.exists(save_folder):
        os.mkdir(save_folder)
      with open(os.path.join(save_folder, path.replace(".txt", ".json")), "w", encoding="utf-8") as outfile:
          outfile.write(json_object)

def preprocess_audio(audio_file, model_vocal, sr=22050):
    n_fft = 2048
    hop_length = 1024
    batchsize = 4
    cropsize = 256
    postprocess=True
    device = "cuda"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, curr_sr = librosa.load(audio_file, sr, False, dtype=np.float32, res_type='kaiser_fast')

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    X_spec = spec_utils.wave_to_spectrogram(X, hop_length, n_fft)
    sp = Separator(model_vocal, device, batchsize, cropsize, postprocess)

    _, v_spec = sp.separate_tta(X_spec)
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=hop_length)
    y = librosa.to_mono(wave)

    if len(y.shape) == 1:
        y = y[np.newaxis, :] # (channel, sample)

    return y, curr_sr

def preprocess_lyrics(lyrics_file, word_file=None):
    from string import ascii_lowercase
    # d = {ascii_lowercase[i]: i for i in range(26)}
    # d["'"] = 26
    # d[" "] = 27
    # d["~"] = 28

    # process raw
    with open(lyrics_file, 'r') as f:
        raw_lines = f.read().splitlines()
 
    raw_lines = ["".join([remove_accent(c) for c in line.lower()]).strip() for line in raw_lines]
    raw_lines = [" ".join(line.split()) for line in raw_lines if len(line) > 0]
    
    # concat
    full_lyrics = " ".join(raw_lines)

    if word_file:
        with open(word_file) as f:
            words_lines = f.read().splitlines()
    else:
        words_lines = full_lyrics.split()

    lyrics_p, words_p, idx_word_p, idx_line_p = utils.gen_phone_gt(words_lines, raw_lines)

    return words_lines, lyrics_p, idx_word_p, idx_line_p


class Separator(object):

    def __init__(self, model, device, batchsize, cropsize, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _separate(self, X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in range(0, patches, self.batchsize):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                pred = self.model.predict_mask(X_batch)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec

    def separate(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask_tta = self._separate(X_mag_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ls_path_audio', type=str,
                        help='List audio vocals')
    parser.add_argument('--ls_path_lyrics', type=str,
                        help='Where all the lyrics of the vocals.')
    parser.add_argument('--ls_json_lyrics', type=str,
                        help='Where all the json lyrics format for submit')
    parser.add_argument('--save_folder', type=str, required=True,
                        help='Saving all json for submit')
    parser.add_argument('--ckp_path', type=str, required=True,
                        help='Checkpoint path for inference')
    args = parser.parse_args()
    print(args)

    main(args)