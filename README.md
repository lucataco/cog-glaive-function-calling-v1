# glaiveai/glaive-function-calling-v1 Cog model

This is an implementation of the [glaiveai/glaive-function-calling-v1](https://huggingface.co/glaiveai/glaive-function-calling-v1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="I am thinking of having a 10 day long vacation in Greece, can you help me plan it?"
