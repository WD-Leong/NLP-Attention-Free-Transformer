# NLP-Attention-Free-Transformer
This repository contains the implementation of [An Attention Free Transformer](https://openreview.net/forum?id=pW--cu2FCHY) in PyTorch. It is trained on the [movie dialog dataset](https://github.com/Abonia1/TF-Chatbot/tree/master/data) using the [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) architecture.

## Training the model
The data is first processed into sub-word tokens prior to training by running the script
```
python process_movie_dialog_subword.py
```
followed by running
```
pythoon train_movie_dialog_sw_aft.py
```
to train the Attention Free Transformer (AFT) model. The model's causality is enforced using [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) as the authors of [Performer](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html) suggested. In the code, this is implemented with `torch.cumsum`.

Once the model is trained, run the script
```
python infer_movie_dialog_sw_aft.py
```
to perform inference.

## Model inference examples 
Below are some examples of the model inference in the wild:
```
Input Phrase:
how are you?
Generated Reply:
SOS fine . how are you ? EOS
--------------------------------------------------
Input Phrase:
what time is it?
Generated Reply:
SOS almost two a . m . EOS
--------------------------------------------------
Input Phrase:
how much is it?
Generated Reply:
SOS a dollar seventy - five . EOS
--------------------------------------------------
Input Phrase:
where are we going?
Generated Reply:
SOS someplace safe . EOS
--------------------------------------------------
Input Phrase:
when are we leaving?
Generated Reply:
SOS tomorrow . EOS
--------------------------------------------------
Input Phrase:
how long is it?
Generated Reply:
SOS six weeks . EOS
--------------------------------------------------
Input Phrase:
what are you doing?
Generated Reply:
SOS i ' m trying to help ... EOS
--------------------------------------------------
Input Phrase:
what did he say?
Generated Reply:
SOS i think he said he ' s right here . EOS
--------------------------------------------------
Input Phrase:
bye.
Generated Reply:
SOS bye , troy ! EOS
--------------------------------------------------
```
Please note that the PAD tokens following the EOS token are omitted. Please also note that the response is generated at randomly using `top-k` sampling.
