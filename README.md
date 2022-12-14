# NLP-Attention-Free-Transformer
This repository contains the implementation of [An Attention Free Transformer](https://openreview.net/forum?id=pW--cu2FCHY) in PyTorch. It is trained on the [movie dialog dataset](https://github.com/Abonia1/TF-Chatbot/tree/master/data) using the architecture mentioned in the paper.

## Training the model
The data is first processed into sub-word tokens prior to training by running the script
```
python process_movie_dialog_subword.py
```
followed by running
```
python train_movie_dialog_sw_aft.py
```
to train the Attention Free Transformer (AFT) model. The model's causality is enforced using [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) as the authors of [Performer](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html) suggested. In the code, this is implemented with `torch.cumsum`. The training of the model can support much longer sequences by computing the Cross-Entropy Loss in segments, although it should be noted that gradient checkpointing was not implemented in this code.

Once the model is trained, run the script
```
python infer_movie_dialog_sw_aft.py
```
to perform inference.

### GPT Model
For convenience, the [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) model is also included and running the command
```
python train_movie_dialog_sw_gpt.py
```
trains the model. Inference can be done via
```
python infer_movie_dialog_sw_gpt.py
```
after the model has been trained.

## Model inference examples 
Below are some examples of the model inference in the wild:
```
Input Phrase:
how are you?
Generated Reply:
SOS fine . EOS
--------------------------------------------------
Input Phrase:
what time is it?
Generated Reply:
SOS eight o ' clock . EOS
--------------------------------------------------
Input Phrase:
how much does it cost?
Generated Reply:
SOS thirty five dollars . EOS
--------------------------------------------------
Input Phrase:
where are we going?
Generated Reply:
SOS out of here -- ? EOS
--------------------------------------------------
Input Phrase:
when are we leaving?
Generated Reply:
SOS tomorrow ? EOS
--------------------------------------------------
Input Phrase:
how long is it?
Generated Reply:
SOS three hours . EOS
--------------------------------------------------
Input Phrase:
what are you doing?
Generated Reply:
SOS i ' m going to take a shower . EOS
--------------------------------------------------
Input Phrase:
what did he say?
Generated Reply:
SOS nothing . EOS
--------------------------------------------------
Input Phrase:
bye.
Generated Reply:
SOS bye .! EOS
--------------------------------------------------
```
Please note that the PAD tokens following the EOS token are omitted. Please also note that the response is generated randomly using `top-k` sampling.
