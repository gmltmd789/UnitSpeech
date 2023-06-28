from setuptools import setup

setup(
    name="unitspeech",
    py_modules=["unitspeech"],
    install_requires=[
        "amfm_decompy==1.0.11",
        "einops==0.6.1",
        "fairseq==0.12.2",
        "inflect==6.0.4",
        "joblib==1.2.0",
        "librosa==0.10.0.post2",
        "matplotlib==3.7.1",
        "packaging==23.1",
        "phonemizer==3.2.1",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        "transformers==4.30.2",
        "unidecode==1.3.6",
    ],
)