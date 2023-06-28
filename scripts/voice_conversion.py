import argparse
import json
import librosa
import os
from scipy.io.wavfile import write
import torch
import torchaudio
from transformers import HubertModel

from unitspeech.unitspeech import UnitSpeech
from unitspeech.encoder import Encoder
from unitspeech.text import symbols
from unitspeech.util import HParams, fix_len_compatibility, sequence_mask
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.models import BigVGAN


@torch.no_grad()
def voice_conversion(
        args, contentvec_encoder, decoder, contentvec, contentvec_length, mel_length, spk_emb, num_downsamplings_in_unet
):
    cond_x, x, x_mask = contentvec_encoder(contentvec, contentvec_length)
    cond_y = cond_x
    y_lengths = torch.LongTensor([contentvec_length]).to(contentvec.device)

    encoder_outputs = torch.nn.functional.interpolate(
        cond_y, size=mel_length, mode='linear'
    )
    y_max_length = mel_length
    y_max_length_ = fix_len_compatibility(mel_length, num_downsamplings_in_unet)
    cond_y = torch.cat([encoder_outputs, torch.zeros_like(encoder_outputs)[:, :, :y_max_length_ - mel_length]], dim=-1)
    y_mask = sequence_mask(torch.LongTensor([mel_length]).to(y_lengths.device), y_max_length_)\
        .unsqueeze(1).to(x_mask.dtype)

    z = torch.randn_like(cond_y, device=cond_y.device)

    # Generate sample by performing reverse dynamics
    decoder_outputs = decoder(
        z, y_mask, cond_y, spk_emb, args.diffusion_step,
        text_gradient_scale=args.text_gradient_scale, spk_gradient_scale=args.spk_gradient_scale
    )
    decoder_outputs = decoder_outputs[:, :, :y_max_length]
    return decoder_outputs


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)


def main(args, hps):
    # Load the source audio and extract the contentvec.
    contentvec_extractor = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
    _ = contentvec_extractor.cuda().eval()

    wav, sr = librosa.load(args.source_path)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    resample_fn = torchaudio.transforms.Resample(sr, 16000).to("cuda")
    wav = wav.cuda()
    mel_length = wav.shape[-1] // hps.data.hop_length

    wav = resample_fn(wav)
    contentvec = contentvec_extractor(wav)["last_hidden_state"]

    # Initialize & load model
    contentvec_encoder = Encoder(
        n_vocab=len(symbols) + 1,
        n_feats=hps.data.n_feats,
        **hps.encoder
    )

    contentvec_encoder_dict = torch.load(args.encoder_path, map_location=lambda loc, storage: loc)
    contentvec_encoder.load_state_dict(contentvec_encoder_dict['model'])
    _ = contentvec_encoder.cuda().eval()

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
    contentvec = contentvec.cuda()
    contentvec_length = torch.LongTensor([contentvec.shape[1]]).cuda()

    spk_emb = decoder_dict['spk_emb'].cuda()

    with torch.no_grad():
        mel_generated = voice_conversion(
            args, contentvec_encoder, unitspeech,
            contentvec, contentvec_length, mel_length, spk_emb, len(hps.decoder.dim_mults) - 1
        )

        audio_generated = vocoder.forward(mel_generated).cpu().squeeze().clamp(-1, 1).numpy()

    if "/" in args.generated_sample_path:
        os.makedirs(os.path.dirname(args.generated_sample_path), exist_ok=True)
    write(args.generated_sample_path, hps.data.sampling_rate, audio_generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default="unitspeech/checkpoints/contentvec_encoder.pt",
                        help='Path of the text encoder checkpoint.')
    parser.add_argument('--decoder_path', type=str, default="unitspeech/outputs/finetuned_decoder.pt",
                        help='Path of the finetuned decoder checkpoint.')
    parser.add_argument('--config_path', type=str, default="unitspeech/checkpoints/voice-conversion.json",
                        help='Path to the configuration file for voice conversion.')
    parser.add_argument('--generated_sample_path', type=str, default="unitspeech/outputs/output_vc.wav",
                        help='The path to save the generated audio.')

    parser.add_argument('--source_path', type=str, required=True,
                        help='The source audio file path for voice conversion.')
    parser.add_argument('--text_gradient_scale', type=float, default=0.0,
                        help='Gradient scale of classifier-free guidance (cfg) for text condition. (0.0: wo cfg)')
    parser.add_argument('--spk_gradient_scale', type=float, default=1.0,
                        help='Gradient scale of classifier-free guidance (cfg) for speaker condition. (0.0: wo cfg)')
    parser.add_argument('--diffusion_step', type=int, default=50,
                        help='The number of iterations for sampling in the diffusion model.')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hps = HParams(**config)

    main(args, hps)