import torch

from unitspeech.vocoder.meldataset import mel_spectrogram


segment_length = 32640


@torch.no_grad()
def extract_speaker_embedding(audio, speaker_embedder, n_fft, n_feats, sampling_rate, hop_length, win_length, mel_fmin,
                              mel_fmax, **kwargs):
    audio = audio / torch.abs(audio).max() * 1.
    audio = torch.clamp(audio, -1, 1)

    # Take segment
    if audio.size(0) < segment_length:
        audio = torch.nn.functional.pad(audio, (0, segment_length - audio.size(0)), 'constant').data

    audio = audio[:audio.size(0) // segment_length * segment_length]
    audio_batch = audio.view(-1, segment_length)

    if audio.size(0) > segment_length + segment_length // 2:
        audio_1 = audio[segment_length // 2: segment_length // 2 + (
                audio.size(0) - segment_length // 2) // segment_length * segment_length]
        audio_1_batch = audio_1.view(-1, segment_length)
        audio_batch = torch.cat([audio_batch, audio_1_batch], dim=0)

    mel = mel_spectrogram(
        audio_batch.clone().detach(),
        n_fft,
        n_feats,
        sampling_rate,
        hop_length,
        win_length,
        mel_fmin,
        mel_fmax,
        center=False
    ).transpose(1, 2)

    embed = speaker_embedder(mel.cuda()).mean(0).cpu().detach()
    embed = embed / embed.norm()
    return embed