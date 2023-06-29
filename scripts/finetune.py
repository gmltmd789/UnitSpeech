import argparse
import json
import librosa
import os
import random
import torch
import torchaudio
from tqdm import tqdm

from unitspeech.unitspeech import UnitSpeech
from unitspeech.encoder import Encoder
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import HParams, fix_len_compatibility, process_unit, generate_path, sequence_mask
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.meldataset import mel_spectrogram
from unitspeech.vocoder.models import BigVGAN


def fine_tune(cond_x, y, y_mask, y_lengths, y_max_length, attn, spk_emb, segment_size, n_feats, decoder):
    if y_max_length < segment_size:
        pad_size = segment_size - y_max_length
        y = torch.cat([y, torch.zeros_like(y)[:, :, :pad_size]], dim=-1)
        y_mask = torch.cat([y_mask, torch.zeros_like(y_mask)[:, :, :pad_size]], dim=-1)

    max_offset = (y_lengths - segment_size).clamp(0)
    offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
    out_offset = torch.LongTensor([
        torch.tensor(random.choice(range(start, end)) if end > start else 0)
        for start, end in offset_ranges
    ]).to(y_lengths)

    attn_cut = torch.zeros(attn.shape[0], attn.shape[1], segment_size, dtype=attn.dtype, device=attn.device)
    y_cut = torch.zeros(y.shape[0], n_feats, segment_size, dtype=y.dtype, device=y.device)
    y_cut_lengths = []
    for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
        y_cut_length = segment_size + (y_lengths[i] - segment_size).clamp(None, 0)
        y_cut_lengths.append(y_cut_length)
        cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
        y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
        attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
    y_cut_lengths = torch.LongTensor(y_cut_lengths)
    y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

    if y_cut_mask.shape[-1] < segment_size:
        y_cut_mask = torch.nn.functional.pad(y_cut_mask, (0, segment_size - y_cut_mask.shape[-1]))

    attn = attn_cut
    y = y_cut
    y_mask = y_cut_mask

    # Align encoded text with mel-spectrogram and get cond_y segment
    cond_y = torch.matmul(attn.squeeze(1).transpose(1, 2).contiguous(), cond_x.transpose(1, 2).contiguous())
    cond_y = cond_y.transpose(1, 2).contiguous()
    cond_y = cond_y * y_mask

    # Compute loss of score-based decoder
    diff_loss, xt = decoder.compute_loss(y, y_mask, cond_y, spk_emb=spk_emb)

    return diff_loss


def main(args, hps):
    segment_size = fix_len_compatibility(
        hps.train.out_size_second * hps.data.sampling_rate // hps.data.hop_length,
        len(hps.decoder.dim_mults) - 1
    )
    num_units = hps.data.n_units

    print('Initializing Vocoder...')
    with open(hps.train.vocoder_config_path) as f:
        h = AttrDict(json.load(f))
    vocoder = BigVGAN(h)
    vocoder.load_state_dict(torch.load(hps.train.vocoder_ckpt_path, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    print('Initializing Speaker Encoder...')
    spk_embedder = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(args.speaker_encoder_path, map_location=lambda storage, loc: storage)
    spk_embedder.load_state_dict(state_dict['model'], strict=False)
    _ = spk_embedder.cuda().eval()

    print('Initializing Unit Extracter...')
    dense_model_name = "mhubert-base-vp_en_es_fr"
    quantizer_name, vocab_size = "kmeans", 1000

    unit_extractor = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        deduplicate=True,
        need_f0=False
    )
    _ = unit_extractor.cuda().eval()

    # Load the normalization parameters for mel-spectrogram normalization.
    mel_min = torch.load("unitspeech/parameters/mel_min.pt").unsqueeze(0).unsqueeze(-1)
    mel_max = torch.load("unitspeech/parameters/mel_max.pt").unsqueeze(0).unsqueeze(-1)

    # Load the reference audio and extract mel-spectrogram and speaker embeddings.
    wav, sr = librosa.load(args.reference_path)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    mel = mel_spectrogram(wav, hps.data.n_fft, hps.data.n_feats, hps.data.sampling_rate, hps.data.hop_length,
                          hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax, center=False)
    mel = (mel - mel_min) / (mel_max - mel_min) * 2 - 1
    mel = mel.cuda()
    resample_fn = torchaudio.transforms.Resample(sr, 16000).cuda()
    wav = resample_fn(wav.cuda())
    spk_emb = spk_embedder(wav)
    spk_emb = spk_emb / spk_emb.norm()

    # Extract the units and unit durations to be used for fine-tuning.
    encoded = unit_extractor(wav.to("cuda"))

    unit, duration = process_unit(encoded, hps.data.sampling_rate, hps.data.hop_length)

    # Initialize model and optimizer
    unit_encoder = Encoder(
        n_vocab=num_units,
        n_feats=hps.data.n_feats,
        **hps.encoder
    )

    unit_encoder_dict = torch.load(args.encoder_path, map_location=lambda loc, storage: loc)
    unit_encoder.load_state_dict(unit_encoder_dict['model'])
    _ = unit_encoder.cuda().eval()

    unitspeech = UnitSpeech(
        n_feats=hps.data.n_feats,
        **hps.decoder
    )

    decoder_dict = torch.load(args.decoder_path, map_location=lambda loc, storage: loc)
    unitspeech.load_state_dict(decoder_dict['model'])
    _ = unitspeech.cuda().train()

    optimizer = torch.optim.Adam(params=unitspeech.parameters(), lr=args.learning_rate)

    if args.fp16_run:
        scaler = torch.cuda.amp.GradScaler()

    # Reshape the input to match the dimensions and convert it to a PyTorch tensor.
    unit = unit.unsqueeze(0).cuda()
    duration = duration.unsqueeze(0).cuda()
    mel = mel.cuda()

    unit_lengths = torch.LongTensor([unit.shape[-1]]).cuda()
    mel_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
    spk_emb = spk_emb.cuda().unsqueeze(1)

    with torch.no_grad():
        cond_x, x, x_mask = unit_encoder(unit, unit_lengths)

    mel_max_length = mel.shape[-1]
    mel_mask = sequence_mask(mel_lengths, mel_max_length).unsqueeze(1).to(x_mask)
    attn_mask = x_mask.unsqueeze(-1) * mel_mask.unsqueeze(2)

    attn = generate_path(duration, attn_mask.squeeze(1))

    # Fine-tuning.
    for _ in tqdm(range(args.n_iters)):
        cond_x = cond_x.detach()
        mel = mel.detach()
        mel_mask = mel_mask.detach()
        mel_lengths = mel_lengths.detach()
        spk_emb = spk_emb.detach()
        attn = attn.detach()

        unitspeech.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16_run):
            diff_loss = fine_tune(cond_x, mel, mel_mask, mel_lengths, mel_max_length, attn, spk_emb, segment_size, hps.data.n_feats, unitspeech)

        loss = sum([diff_loss])

        if args.fp16_run:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _ = torch.nn.utils.clip_grad_norm_(unitspeech.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(unitspeech.parameters(), max_norm=1)
            optimizer.step()

    if "/" in args.output_decoder_path:
        os.makedirs(os.path.dirname(args.output_decoder_path), exist_ok=True)

    _ = unitspeech.eval()
    torch.save({
        'model': unitspeech.state_dict(),
        'spk_emb': spk_emb.cpu()},
        f=args.output_decoder_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_path', type=str, required=True,
                        help='The reference audio you want to adapt to.')
    parser.add_argument('--encoder_path', type=str, default="unitspeech/checkpoints/unit_encoder.pt",
                        help='Path of the unit encoder checkpoint.')
    parser.add_argument('--decoder_path', type=str, default="unitspeech/checkpoints/pretrained_decoder.pt",
                        help='Path of the decoder checkpoint.')
    parser.add_argument('--speaker_encoder_path', type=str, default="unitspeech/speaker_encoder/checkpts/speaker_encoder.pt",
                        help='Path to the speaker encoder checkpoint.')
    parser.add_argument('--config_path', type=str, default="unitspeech/checkpoints/finetune.json",
                        help='Path to the configuration file for fine-tuning.')
    parser.add_argument('--output_decoder_path', type=str, default="unitspeech/outputs/finetuned_decoder.pt",
                        help='Path to save the finetuned decoder checkpoint.')
    parser.add_argument('--n_iters', type=int, default=500,
                        help='Number of fine-tuning iterations.')
    parser.add_argument('--learning_rate', type=int, default=2e-5,
                        help='Learning rate of the optimizer during fine-tuning.')
    parser.add_argument('--fp16_run', action='store_true',
                        help='Whether to perform fine-tuning with half precision.')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hps = HParams(**config)

    main(args, hps)