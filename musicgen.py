import json

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor

from repeng import ControlVector, ControlModel, DatasetEntry

model_name = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

model = MusicgenForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
model = ControlModel(model, list(range(0, 24)))

suffixes = ['music', 'song', 'track', 'tune', 'melody', 'harmony', 'rhythm', 'beat', 'composition', 'piece']

def make_dataset(
  template: str,
  positive_personas: list[str],
  negative_personas: list[str],
  suffix_list: list[str]
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template} {suffix}",
                    negative=f"{negative_template} {suffix}",
                )
            )
    return dataset

dataset = make_dataset(
  '{persona}',
  ['scary', 'intense terrifying', 'frightening', 'creepy', 'spooky', 'haunting', 'eerie', 'horror', 'macabre', 'chilling'],
  ['happy', 'joyful', 'cheerful', 'content', 'delighted', 'pleased', 'glad', 'satisfied', 'joyous', 'merry'],
  suffixes
)
scary_vector = ControlVector.train(model, processor, dataset)
print(scary_vector)