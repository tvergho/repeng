# repeng

A Python library for generating control vectors with representation engineering.
Train a vector in less than sixty seconds!

_For a full example, see the notebooks folder._

```python
...

from repeng import ControlVector, ControlModel, DatasetEntry

# load and wrap Mistral-7B
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = ControlModel(model, list(range(-5, -18, -1)))

...

# generate a dataset with closely-opposite paired statements
trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes,
)

# train the vector—takes less than a minute!
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

# set the control strength and let inference rip!
for strength in (-2.2, 1, 2.2):
    print(f"strength={strength}")
    model.set_control(trippy_vector, strength)
    out = model.generate(
        **tokenizer(
            f"[INST] Give me a one-sentence pitch for a TV show. [/INST]",
            return_tensors="pt"
        ),
        do_sample=False,
        ...
    )
    print(tokenizer.decode(out.squeeze()).strip())
    print()
```

> strength=-2.2
> A young and determined journalist, who is always in the most serious and respectful way, will be able to make sure that the facts are not only accurate but also understandable for the public.
>
> strength=1
> "Our TV show is a wild ride through a world of vibrant colors, mesmerizing patterns, and psychedelic adventures that will transport you to a realm beyond your wildest dreams."
>
> strength=2.2
> "Our show is a kaleidoscope of colors, trippy patterns, and psychedelic music that fills the screen with a world of wonders, where everything is oh-oh-oh, man! ��psy����������oodle����psy��oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

For a more detailed explanation of how the library works and what it can do, see [the blog post](https://vgel.me/posts/representation-engineering).