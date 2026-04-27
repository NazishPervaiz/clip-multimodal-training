from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")

sample = dataset["test"][0]

image = sample["image"]
caption = sample["caption"][0]

plt.imshow(image)
plt.title(caption)
plt.axis("off")
plt.show()