""" from https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS """

import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0],
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def fix_len_compatibility(length, num_downsamplings_in_unet=3):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return int(length)
        length += 1


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def process_unit(encoded, sampling_rate, hop_length):
    # A method that aligns units and durations (50Hz) extracted from 16kHz audio with
    # mel-spectrograms extracted from 22,050Hz audio.

    unit = encoded["units"].cpu().tolist()
    duration = encoded["durations"].cpu().tolist()

    duration = [int(i) * (sampling_rate // 50) for i in duration]

    expand_unit = []

    for u, d in zip(unit, duration):
        for _ in range(d):
            expand_unit.append(u)

    new_length = len(expand_unit) // hop_length * hop_length

    unit = torch.LongTensor(expand_unit)[:new_length].reshape(-1, hop_length).mode(1)[0].tolist()

    squeezed_unit = [unit[0]]
    squeezed_duration = [1]

    for u in unit[1:]:
        if u == squeezed_unit[-1]:
            squeezed_duration[-1] += 1
        else:
            squeezed_unit.append(u)
            squeezed_duration.append(1)

    unit = torch.LongTensor(squeezed_unit)
    duration = torch.LongTensor(squeezed_duration)

    return unit, duration


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()