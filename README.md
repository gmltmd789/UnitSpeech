## UnitSpeech: Speaker-adaptive Speech Synthesis with Untranscribed Data (INTERSPEECH 2023, Oral)
#### Heeseung Kim, Sungwon Kim, Jiheum Yeom, Sungroh Yoon
![model-1](https://github.com/gmltmd789/UnitSpeech/assets/49265950/44cb4991-abb0-44b2-81fd-fce92cc1f3f1)
<br><br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jAglTVrBNeEQbOAJ3T_YRotoKqCBPNn9?usp=sharing)
### [Paper](https://arxiv.org/abs/2306.16083)
### [Audio demo](https://unitspeech.github.io/)

## Updates (Updated components compared to the version of INTERSPEECH.)
- **Change in vocoder (from HiFi-GAN to BigVGAN).**
- **Support for speaker classifier-free guidance (advantageous for adapting to more unique voices.)**
- **Change "training-free text classifier-free guidance" to "text classifier-free guidance" (learning text unconditional embedding).**
- **Ensure compatibility with various recent works on unit-based speech synthesis (number of clusters of unit (K): 200 &rightarrow; 1000)**
- **Substantial improvement in pronunciation accuracy**
  - **To improve TTS (Text-to-Speech) pronunciation, an IPA-based phonemizer is used.** 
  - **To improve VC (Voice Conversion) pronunciation, a contentvec encoder is introduced.**

# Warning: Ethical & Legal Considerations
1. **UnitSpeech was created with the primary objective of facilitating research endeavors.**
2. **When utilizing samples generated using this model, it is crucial to clearly disclose that the samples were generated using AI technology. Additionally, it is necessary to provide the sources of the audio used in the generation process.**
3. **We notify that users take full responsibility for any possible negative outcomes and legal & ethical issues that may arise due to their misuse of the model.**
4. **As a precautionary measure against possible misapplication, we intend to introduce a classification model capable of discerning samples generated through the utilization of this model.**

## TO DO
- [ ] Release a classification model to distinguish samples from UnitSpeech

## Installation
**Tested on Ubuntu 20.04.5 LTS, Python 3.8, Anaconda (2023.03-1) environment**  
First, install the necessary package for the IPA phonemizer.
```shell
sudo apt-get install espeak=1.48.04+dfsg-8build1 espeak-ng=1.50+dfsg-6
```
If you are unable to install the specific versions of espeak and espeak-ng on Ubuntu 18.04 or earlier, please install the available versions of each package.<br>
Note: If you have a different version of espeak-ng, the output of phonemizing text may vary, which can affect pronunciation accuracy.

After that, create a conda environment and install the unitspeech package and the package required for extracting speaker embeddings.
```shell
conda create -n unitspeech python=3.8
conda activate unitspeech
git clone https://github.com/gmltmd789/UnitSpeech.git
cd UnitSpeech
pip install -e .
```

## Pretrained Models
**We provide the [pretrained models](https://drive.google.com/drive/folders/1yFkb2TAYB_zMmoTuUOXu-zXb3UI9pVJ9?usp=sharing).**
|File Name|Usage|
|------|---|
|contentvec_encoder.pt|Used for any-to-any voice conversion tasks.|
|unit_encoder.pt|Used for fine-tuning and unit-based speech synthesis tasks.<br>(e.g., Adaptive Speech Synthesis for Speech-to-Unit Translation)|
|text_encoder.pt|Used for adaptive text-to-speech tasks.|
|duration_predictor.pt|Used for adaptive text-to-speech tasks.|
|pretrained_decoder.pt|Used for all adaptive speech synthesis tasks.|
|speaker_encoder.pt|Used for extracting speaker embeddings.|
|bigvgan.pt|[Vocoder](https://github.com/NVIDIA/BigVGAN) checkpoint.|
|bigvgan-config.json|Configuration for the vocoder.|

**After downloading the files, please arrange them in the following structure.**
```buildoutcfg
UnitSpeech/...
    unitspeech/...
        checkpoints/...
            contentvec_encoder.pt
            duration_predictor.pt
            pretrained_decoder.pt
            text_encoder.pt
            unit_encoder.pt
            ...
        speaker_encoder/...
            checkpts/...
                speaker_encoder.pt
            ...
        vocoder/...
            checkpts/...
                bigvgan.pt
                bigvgan-config.json
            ...
        ...
    ...
```

## Fine-tuning
The decoder is fine-tuned using the target speaker's voice, employing the unit encoder. **It is recommended to use a reference English speech with a duration of at least 5~10 seconds.**

```shell
python scripts/finetune.py \
--reference_path REFERENCE_SPEECH_PATH \
--output_decoder_path FILEPATH1/FINETUNED_DECODER.pt
```

By executing the code, your personalized decoder will be saved as "FILEPATH1/FINETUNED_DECODER.pt".<br>
With the fine-tuned decoder, you can perform adaptive text-to-speech and any-to-any voice conversion, as described below. <br> <br>
By default, fine-tuning is conducted in fp32 using the Adam optimizer with a learning rate of 2e-5 for 500 iterations.<br>
You can adjust the above elements through arguments provided. (--fp16_run, --learning_rate, --n_iters)<br>
**For speakers with unique voices, increasing the number of fine-tuning iterations can help achieve better results.** <br>

## Inference
```shell
# script for adaptive text-to-speech
python scripts/text_to_speech.py \
--text "TEXT_TO_GENERATE" \
--decoder_path FILEPATH1/FINETUNED_DECODER.pt \
--generated_sample_path FILEPATH2/PATH_TO_SAVE_SYNTHESIZED_SPEECH.wav


# script for any-to-any voice conversion
python scripts/voice_conversion.py \
--source_path SOURCE_SPEECH_PATH_TO_CONVERT.wav \
--decoder_path FILEPATH1/FINETUNED_DECODER.pt \
--generated_sample_path FILEPATH2/PATH_TO_SAVE_SYNTHESIZED_SPEECH.wav
```
You can adjust the number of diffusion steps, text gradient scale, and speaker gradient scale as arguments.<br>
- text_gradient_scale : responsible for pronunciation accuracy and audio quality. Increasing its value makes the pronunciation of the samples more accurate.<br>
- spk_gradient_scale : responsible for speaker similarity. Increasing its value generates voices that are closer to the reference speech.<br>

By default, text gradient scale is set to 0.0, and speaker gradient scale is set to 1.0.<br>
**If you want better pronunciation and audio quality, please increase the value of "text_gradient_scale." This will slightly reduce speaker similarity.**<br>
**If you want better speaker similarity, please increase the value of "spk_gradient_scale." This will slightly degrade pronunciation accuracy and audio quality.**<br>

You can adjust the speed of speaking as arguments. (default: 1.0) <br>
- length_scale : Increasing its value (> 1.0) makes the speech slow, while decreasing its value (< 1.0) makes the speech fast <br>

**Note: Using excessively large gradient scales can degrade the audio quality.**

## License

The code and model weights of UnitSpeech are released under the CC BY-NC-SA 4.0 license.

## References
* [BigVGAN](https://github.com/NVIDIA/BigVGAN) (for vocoder)
* [textlesslib](https://github.com/facebookresearch/textlesslib) (for unit extraction)
* [ContentVec](https://github.com/auspicious3000/contentvec) (for contentvec extraction)
* [VITS](https://github.com/jaywalnut310/vits) (for text & IPA phoneme sequence processing)
* [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) (for overall architecture and code)
* [denoising-diffusion-pytorch](https://github.com/rosinality/denoising-diffusion-pytorch) (for diffusion-based sampler)
* [Pytorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification) (for speaker embedding extraction)

## Citation
```
@misc{kim2023unitspeech,
      title={UnitSpeech: Speaker-adaptive Speech Synthesis with Untranscribed Data}, 
      author={Heeseung Kim and Sungwon Kim and Jiheum Yeom and Sungroh Yoon},
      year={2023},
      eprint={2306.16083},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
