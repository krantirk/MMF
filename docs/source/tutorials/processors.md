# Tutorial: Adding a processor

Processors can be thought of as torchvision transforms which transform a sample into a form usable by the model. Each processor takes in a dictionary and returns back a dictionary. Processors are initialized as member variables of the dataset and can be used while generating samples.


For this tutorial, we will create a processor to get sentence embeddings.

## Create the processor class

A [`fasttext`](https://github.com/facebookresearch/mmf/blob/f11adf0e4a5a28e85239176c44342f6471550e84/mmf/datasets/processors/processors.py#L361) processor is available in MMF that returns word embeddings instead of sentence embedding. We will create a `fasttext` *sentence* processor here by extending the `fasttext` word processor.

```python

import torch

# registry is needed to register processor and model to be MMF discoverable
from mmf.common.registry import registry
# We will inherit the FastText Processor already present in MMF
from mmf.datasets.processors import FastTextProcessor


# Register the processor so that MMF can discover it
@registry.register_processor("fasttext_sentence_vector")
class FastTextSentenceVectorProcessor(FastTextProcessor):
   # Override the call method
   def __call__(self, item):
       # This function is present in FastTextProcessor class and loads
       # fasttext bin
       self._load_fasttext_model(self.model_file)
       if "text" in item:
           text = item["text"]
       elif "tokens" in item:
           text = " ".join(item["tokens"])

       # Get a sentence vector for sentence and convert it to torch tensor
       sentence_vector = torch.tensor(
           self.model.get_sentence_vector(text),
           dtype=torch.float
       )
       # Return back a dict
       return {
           "text": sentence_vector
       }

   # Make dataset builder happy, return a random number
   def get_vocab_size(self):
       return None
```

## Add the processor configuration to dataset config

Let's assume we are building this processor for VQA dataset. We will add the processor's configuration to the VQA dataset's config:

```yaml
dataset_config:
  vqa2:
    processors:
      text_processor:
        type: fasttext_sentence_vector
        params:
          max_length: null
          model_file: wiki.en.bin
```

The `fasttext_sentence_vector` proccessor will be then available to the VQA dataset class as `text_processor`, to be used to process questions and get their sentence vectors. Here `model_file` specifies the `fasttext` model used in our processor class.

## Next Steps

Learn more about processors in the
[processors documentation](../lib/datasets/processors).
