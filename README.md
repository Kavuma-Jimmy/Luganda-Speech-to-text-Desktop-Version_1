---
language: lg
datasets:
- common_voice
metrics:
- wer
tags:
- audio
- automatic-speech-recognition
- speech
license: apache-2.0
model-index:
- name: Wav2Vec2 Luganda by Indonesian-NLP
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice lg
      type: common_voice
      args: lg
    metrics:
       - name: Test WER
         type: wer
         value: 7.53
---

# Automatic Speech Recognition for Luganda

This is the model built for the 
[Mozilla Luganda Automatic Speech Recognition competition](https://zindi.africa/competitions/mozilla-luganda-automatic-speech-recognition).
It is a fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
model on the [Luganda Common Voice dataset](https://huggingface.co/datasets/common_voice) version 7.0.

We also provide a [live demo](https://huggingface.co/spaces/indonesian-nlp/luganda-asr) to test the model.

When using this model, make sure that your speech input is sampled at 16kHz.

## Usage
The model can be used directly (without a language model) as follows:
```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("common_voice", "lg", split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-luganda")
model = Wav2Vec2ForCTC.from_pretrained("indonesian-nlp/wav2vec2-luganda")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    if "audio" in batch:
        speech_array = torch.tensor(batch["audio"]["array"])
    else:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset[:2]["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset[:2]["sentence"])
```


## Evaluation

The model can be evaluated as follows on the Indonesian test data of Common Voice.

```python
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

test_dataset = load_dataset("common_voice", "lg", split="test")
wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-luganda")
model = Wav2Vec2ForCTC.from_pretrained("indonesian-nlp/wav2vec2-luganda") 
model.to("cuda")

chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", "‘", "’", "’"]
chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    if "audio" in batch:
        speech_array = torch.tensor(batch["audio"]["array"])
    else:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
```

WER without KenLM: 15.38 %

WER With KenLM:

**Test Result**: 7.53 %

## Training

The Common Voice `train`, `validation`, and ... datasets were used for training as well as ... and ...  # TODO

The script used for training can be found [here](https://github.com/indonesian-nlp/luganda-asr)
