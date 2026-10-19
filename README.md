<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
    A Language-Informed Loss for<br/>Visual Generation</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://www.suchiragarwal.com/" target="_blank" style="text-decoration: none;">Suchir&nbsp;Agarwal</a>,&nbsp;
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan&nbsp;Chandrasegaran</a>,&nbsp;
    <a href="https://ziyc.github.io/" target="_blank" style="text-decoration: none;">Ziyu&nbsp;Chen</a>,&nbsp;
    <a href="https://kylesargent.github.io/" target="_blank" style="text-decoration: none;">Kyle&nbsp;Sargent</a>,&nbsp;<br/>
    <a href="https://www.niebles.net/" target="_blank" style="text-decoration: none;">Juan&nbsp;Carlos&nbsp;Niebles</a>,&nbsp;
    <a href="https://profiles.stanford.edu/fei-fei-li" target="_blank" style="text-decoration: none;">Li&nbsp;Fei-Fei</a>,&nbsp;
    <a href="https://jiajunwu.com" target="_blank" style="text-decoration: none;">Jiajun&nbsp;Wu</a><br/>
<span class="author-block">Stanford University</span><br/>
<a href="https://arxiv.org/abs/2610.23653" title="arXiv" target="_blank" rel="nofollow" style="text-decoration: none;">📄 arXiv</a> |
<a href="https://www.suchiragarwal.com/lilie/" title="Website" target="_blank" rel="nofollow" style="text-decoration: none;">🌎 Website</a> |
<a href="https://github.com/SuchirAgarwal/lilie" title="Code" target="_blank" rel="nofollow" style="text-decoration: none;">💻 Code</a> |
<a href="https://huggingface.co/suchiragarwal/lilie" title="Models" target="_blank" rel="nofollow" style="text-decoration: none;">🤗 Models</a>
</p>

JAX code for training **LILIE** (Language-Informed and Lossless Image Encoder) and for
training and sampling JiT-T2I with LILIE
loss on [GPIC](https://huggingface.co/datasets/stanford-vision-lab/gpic).

## Pretrained models

Our checkpoints are on Hugging Face at
[suchiragarwal/lilie](https://huggingface.co/suchiragarwal/lilie):


```bash
hf download suchiragarwal/lilie --local-dir checkpoints
```



## Installation

Python 3.12. Install JAX for your accelerator, then the pinned dependencies:

```bash
pip install "jax[tpu]==0.10.2"
pip install -r requirements.txt
hf auth login
```



## Training LILIE

```bash
python train_lilie.py --config configs/lilie/small.yaml distributed=false
```

`configs/lilie/` lists the Small, Tiny and Micro encoder configurations.

## Training a generator

```bash
python train_jit.py --config configs/jit/g_pixel+lilie.yaml \
  lilie_checkpoint=checkpoints/lilie/small distributed=false
```



## Sampling

For an interactive demo, see [demo/sampling.ipynb](demo/sampling.ipynb).

```bash
python sample_jit.py --config configs/sample/g_pixel+lilie.yaml \
  captions=captions.jsonl distributed=false
```

`configs/sample/` has one config per released generator. Each samples at the CFG scale
selected on GPIC Validation-50K, the scale used for that model's results in the paper.

`captions` is a JSONL file with one `{"key": ..., "caption": ...}` per line, for example
the captions of the GPIC Validation or Test split.

## Evaluation

Score the NPZ files with the [GPIC evaluation toolkit](https://github.com/keshik6/gpic/tree/main/gpic_eval)
against the official validation or test reference statistics.

## Acknowledgments

This code builds on [TARFlow](https://github.com/apple-aiml-research/ml-tarflow) and
[BiFlow](https://github.com/Lyy-iiis/BiFlow) (encoder), [PixelGen](https://github.com/Zehong-Ma/PixelGen)
and the [GPIC](https://github.com/keshik6/gpic) baseline (generator, training and sampling),
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (text encoder) and
[SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
(captioning model), with the model ports adapted from [Transformers](https://github.com/huggingface/transformers).
