import argparse
import json
import os
import phonemizer
from scipy.io.wavfile import write
import torch

from unitspeech.unitspeech import UnitSpeech
from unitspeech.duration_predictor import DurationPredictor
from unitspeech.encoder import Encoder
from unitspeech.text import cleaned_text_to_sequence, phonemize, symbols
from unitspeech.util import HParams, intersperse, fix_len_compatibility, sequence_mask, generate_path
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.models import BigVGAN


@torch.no_grad()
def text_to_speech(
        args, text_encoder, duration_predictor, decoder, phoneme, phoneme_lengths, spk_emb, num_downsamplings_in_unet
):
    cond_x, x, x_mask = text_encoder(phoneme, phoneme_lengths)
    logw = duration_predictor(x, x_mask, w=None, g=spk_emb, reverse=True)
    w = torch.exp(logw) * x_mask
    w_ceil = torch.ceil(w) * args.length_scale

    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max_length = int(y_lengths.max())
    y_max_length_ = fix_len_compatibility(y_max_length, num_downsamplings_in_unet)

    # Using obtained durations `w` construct alignment map `attn`
    y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

    # Align encoded text and get mu_y
    cond_y = torch.matmul(attn.squeeze(1).transpose(1, 2).contiguous(), cond_x.transpose(1, 2).contiguous())
    cond_y = cond_y.transpose(1, 2).contiguous()

    z = torch.randn_like(cond_y, device=cond_y.device)

    # Generate sample by performing reverse dynamics
    decoder_outputs = decoder(
        z, y_mask, cond_y, spk_emb, args.diffusion_step,
        text_gradient_scale=args.text_gradient_scale, spk_gradient_scale=args.spk_gradient_scale
    )
    decoder_outputs = decoder_outputs[:, :, :y_max_length]
    return decoder_outputs


def main(args, hps):
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
    )

    # Load the normalization parameters for mel-spectrogram normalization.
    mel_min = torch.load("unitspeech/parameters/mel_min.pt").unsqueeze(0).unsqueeze(-1)
    mel_max = torch.load("unitspeech/parameters/mel_max.pt").unsqueeze(0).unsqueeze(-1)

    # Initialize & load model
    text_encoder = Encoder(
        n_vocab=len(symbols) + 1,
        n_feats=hps.data.n_feats,
        **hps.encoder
    )

    text_encoder_dict = torch.load(args.encoder_path, map_location=lambda loc, storage: loc)
    text_encoder.load_state_dict(text_encoder_dict['model'])
    _ = text_encoder.cuda().eval()

    duration_predictor = DurationPredictor(
        **hps.duration_predictor
    )

    duration_predictor_dict = torch.load(args.duration_predictor_path, map_location=lambda loc, storage: loc)
    duration_predictor.load_state_dict(duration_predictor_dict['model'])
    _ = duration_predictor.cuda().eval()

    unitspeech = UnitSpeech(
        n_feats=hps.data.n_feats,
        **hps.decoder
    )

    decoder_dict = torch.load(args.decoder_path, map_location=lambda loc, storage: loc)
    unitspeech.load_state_dict(decoder_dict['model'])
    _ = unitspeech.cuda().train()

    # Initialize & load vocoder.
    with open(hps.train.vocoder_config_path) as f:
        h = AttrDict(json.load(f))
    vocoder = BigVGAN(h)
    vocoder.load_state_dict(torch.load(hps.train.vocoder_ckpt_path, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # Prepare input
    phoneme = phonemize(args.text, global_phonemizer)
    phoneme = cleaned_text_to_sequence(phoneme)
    phoneme = intersperse(phoneme, len(symbols))  # add a blank token, whose id number is len(symbols)
    phoneme = torch.LongTensor(phoneme).cuda().unsqueeze(0)
    phoneme_lengths = torch.LongTensor([phoneme.shape[-1]]).cuda()

    spk_emb = decoder_dict['spk_emb'].cuda()

    with torch.no_grad():
        mel_generated = text_to_speech(
            args, text_encoder, duration_predictor, unitspeech,
            phoneme, phoneme_lengths, spk_emb, len(hps.decoder.dim_mults) - 1
        )

        mel_generated = ((mel_generated + 1) / 2 * (mel_max.to(mel_generated.device) - mel_min.to(mel_generated.device))
                         + mel_min.to(mel_generated.device))
        audio_generated = vocoder.forward(mel_generated).cpu().squeeze().clamp(-1, 1).numpy()

    if "/" in args.generated_sample_path:
        os.makedirs(os.path.dirname(args.generated_sample_path), exist_ok=True)
    write(args.generated_sample_path, hps.data.sampling_rate, audio_generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default="unitspeech/checkpoints/text_encoder.pt",
                        help='Path of the text encoder checkpoint.')
    parser.add_argument('--decoder_path', type=str, default="unitspeech/outputs/finetuned_decoder.pt",
                        help='Path of the finetuned decoder checkpoint.')
    parser.add_argument('--duration_predictor_path', type=str, default="unitspeech/checkpoints/duration_predictor.pt",
                        help='Path of the duration predictor checkpoint.')
    parser.add_argument('--config_path', type=str, default="unitspeech/checkpoints/text-to-speech.json",
                        help='Path to the configuration file for text-to-speech.')
    parser.add_argument('--generated_sample_path', type=str, default="unitspeech/outputs/output_tts.wav",
                        help='The path to save the generated audio.')

    parser.add_argument('--text', type=str, required=True,
                        help='The desired transcript to be generated.')
    parser.add_argument('--text_gradient_scale', type=float, default=0.0,
                        help='Gradient scale of classifier-free guidance (cfg) for text condition. (0.0: wo cfg)')
    parser.add_argument('--spk_gradient_scale', type=float, default=1.0,
                        help='Gradient scale of classifier-free guidance (cfg) for speaker condition. (0.0: wo cfg)')
    parser.add_argument('--length_scale', type=float, default=1.0,
                        help='The parameter for adjusting speech speed. The smaller it is compared to 1, the faster the speech becomes.')
    parser.add_argument('--diffusion_step', type=int, default=50,
                        help='The number of iterations for sampling in the diffusion model.')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hps = HParams(**config)

    main(args, hps)