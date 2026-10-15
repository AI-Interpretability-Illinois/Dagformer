# DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections

[![arXiv](https://img.shields.io/badge/arXiv-2610.19625-b31b1b.svg)](https://arxiv.org/abs/2610.19625)

**Xiaocong Yang\*, Yuren Hao\*, ChengXiang Zhai**
<br>AI Interpretability @ Illinois, University of Illinois Urbana-Champaign
<br><sub>\* Equal contribution</sub>

<p align="center">
  <img src="docs/assets/fig1_architecture.png" alt="DAGFormer architecture: the topology predictor reads the input tokens and outputs the planned cross-layer graph; each layer of the backbone reads its query, key, value and residual inputs from the embedding and all earlier layers with the planned weights" width="100%">
</p>

<sub>**Figure 1.** For token *x<sub>t</sub>*, the topology predictor outputs the planned graph as four triangular matrices, for the query, key and value inputs of every attention head and the residual input of every layer: one column per earlier output, one row per head (one per layer for the residual). Only the edges into layer 5 are drawn, with width proportional to the weight. All weights shown are real values from a trained DAGFormer.</sub>

## Overview

We propose DAGFormer, a Transformer language model whose cross-layer connections are planned by a jointly trained topology predictor. Beyond the residual connections in a standard Transformer, DAGFormer adds direct connections from earlier layers to later ones, and the predictor controls the strength of every connection based on input tokens. Unlike existing cross-layer routing architectures, which set each layer's connections separately, DAGFormer plans the routing of the whole model at once. On matched OLMo-2 pairs from 75M to 1B parameters, DAGFormer lowers WikiText-2 perplexity by 17% to 25% over the vanilla Transformer, and a 1B DAGFormer trained on 5B tokens outperforms the vanilla model trained on 10B. Against four routing baselines with the same data and budget, it reaches the lowest perplexity at 75M, 300M and 1B. The gain carries over to 13 commonsense, science, reading and math evaluation tasks (2.0 to 2.5 points higher mean accuracy at 300M and 1B) and to instruction finetuning (63% to 70% AlpacaEval win rate against the vanilla Transformer, leading every routing baseline), and under structured pruning DAGFormer degrades the least in five of six settings.

In a standard Transformer, a layer sees earlier layers only through the residual stream, the running sum of all their outputs. DAGFormer instead lets every attention head read its query, key and value inputs, and every layer its residual input, as weighted mixtures of the embedding and all earlier layer outputs. A small causal Transformer, the **topology predictor**, reads the input tokens and predicts the weights of all these connections at once; each layer then refines its own weights with a small local module. The predictor is trained jointly with the backbone.

## Results

All comparisons use matched OLMo-2 backbones: the same tokenizer, data order, token budget, optimizer and seed for every model of a size. The 75M to 300M models are trained for about 21 tokens per parameter on Dolma v1.7; the 1B models for 5B tokens, continued to 10B.

**Language modeling.** Held-out WikiText-2 NLL (nats per token; lower is better). Gap to vanilla: vanilla minus DAGFormer. Bold: best per row. –: not run (the baselines were trained at 5B tokens only).

| Size | vanilla | DenseFormer | Hyper-conn. | mHC | MUDDFormer | DAGFormer | Gap to vanilla |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 75M | 4.997 | 5.027 | 4.964 | 4.913 | 4.974 | **4.711** | +0.29 |
| 150M | 4.268 | 4.246 | 4.220 | 4.209 | **4.041** | 4.085 | +0.18 |
| 300M | 3.808 | 3.745 | 3.692 | 3.652 | 3.649 | **3.544** | +0.26 |
| 1B, 5B tokens | 3.569 | 3.513 | 3.492 | 3.438 | 3.423 | **3.307** | +0.26 |
| 1B, 10B tokens | 3.446 | – | – | – | – | **3.219** | +0.23 |

Prefill time per token relative to the vanilla Transformer (forward pass over 8 sequences of 1024 tokens, compiled, on an NVIDIA A100 40GB; lower is better):

| Size | vanilla | DenseFormer | Hyper-conn. | mHC | MUDDFormer | DAGFormer |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 75M | 1.00 | 1.01 | 1.99 | 1.75 | 1.40 | 1.49 |
| 150M | 1.00 | 1.02 | 2.11 | 1.78 | 1.48 | 1.61 |
| 300M | 1.00 | 1.02 | 2.14 | 1.74 | 1.54 | 1.89 |
| 1B | 1.00 | 1.01 | 1.90 | 1.54 | 1.36 | 1.81 |

**Evaluation tasks.** Base models, mean accuracy (%) per kind of task: commonsense (HellaSwag, PIQA, WinoGrande, CommonsenseQA, Social IQa), science (ARC-Easy/Challenge, OpenBookQA, SciQ), reading (BoolQ, LAMBADA) and math (MathQA, GSM8K, both close to chance at these sizes), and over all 13 tasks. Mean rank: rank within the block averaged over the 13 tasks (1 = best). Bold: best per block.

| Size | Model | Commonsense (5) | Science (4) | Reading (2) | Math (2) | Mean (13) | Mean rank |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 300M | vanilla | 39.8 | 36.5 | 38.1 | 11.7 | 34.2 | 4.15 |
|  | DenseFormer | 38.9 | 36.5 | 40.8 | 12.7 | 34.4 | 4.23 |
|  | Hyper-conn. | 38.5 | 38.5 | 38.3 | 12.3 | 34.4 | 3.69 |
|  | mHC | 39.2 | 38.8 | 41.8 | **12.9** | 35.4 | 2.54 |
|  | MUDDFormer | 39.7 | 38.6 | 40.9 | 11.5 | 35.2 | 3.08 |
|  | DAGFormer | **39.9** | **39.8** | **43.5** | 12.8 | **36.2** | **2.38** |
| 1B, 5B tokens | vanilla | 40.0 | 38.9 | 42.1 | 12.1 | 35.7 | 4.54 |
|  | DenseFormer | 40.2 | 39.5 | 43.0 | 11.4 | 36.0 | 4.38 |
|  | Hyper-conn. | 40.7 | 40.7 | 41.6 | 12.0 | 36.4 | 3.77 |
|  | mHC | 41.1 | 42.2 | 43.8 | **12.5** | 37.4 | 2.62 |
|  | MUDDFormer | 40.7 | 42.4 | 38.3 | 12.2 | 36.5 | 3.15 |
|  | DAGFormer | **41.8** | **43.0** | **44.7** | 12.4 | **38.1** | **2.08** |
| 1B, 10B tokens | vanilla | 40.5 | 41.0 | 43.0 | 11.5 | 36.6 | – |
|  | DAGFormer | **42.4** | **44.1** | **47.0** | **12.8** | **39.1** | – |

**Instruction finetuning.** Each model finetuned on Alpaca (cleaned) plus Dolly, or on smol-smoltalk. NLL: held-out negative log-likelihood on the instruction data. Win rate: AlpacaEval win rate against the vanilla Transformer of the same size and recipe (both answer orders judged by gpt-5.6-sol, ties count half). Bold: best per block.

| Size | Model | Alpaca NLL | Alpaca win rate | SmolTalk NLL | SmolTalk win rate |
|:---|:---|:---:|:---:|:---:|:---:|
| 300M | vanilla | 2.392 | – | 1.760 | – |
|  | DenseFormer | 2.342 | 0.55 | 1.722 | 0.57 |
|  | Hyper-conn. | 2.298 | 0.59 | 1.684 | 0.58 |
|  | mHC | 2.277 | 0.58 | 1.664 | 0.62 |
|  | MUDDFormer | 2.256 | 0.63 | 1.639 | 0.64 |
|  | DAGFormer | **2.203** | **0.65** | **1.595** | **0.70** |
| 1B, 5B tokens | vanilla | 2.208 | – | 1.728 | – |
|  | DenseFormer | 2.186 | 0.46 | 1.706 | 0.50 |
|  | Hyper-conn. | 2.150 | 0.51 | 1.673 | 0.56 |
|  | mHC | 2.110 | 0.55 | 1.633 | 0.61 |
|  | MUDDFormer | 2.104 | 0.57 | 1.617 | 0.56 |
|  | DAGFormer | **2.042** | **0.63** | **1.558** | **0.66** |
| 1B, 10B tokens | vanilla | 2.120 | – | 1.669 | – |
|  | DAGFormer | **1.974** | **0.66** | **1.516** | **0.70** |

**Structured pruning.** Pruning during domain finetuning on MathInstruct: held-out NLL of the unpruned finetune (0%) and the damage (loss increase relative to it, %) at 50% and 70% backbone sparsity. Bold: best per column.

| Model | 75M NLL (0%) | 75M damage 50% | 75M damage 70% | 150M NLL (0%) | 150M damage 50% | 150M damage 70% | 300M NLL (0%) | 300M damage 50% | 300M damage 70% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| vanilla | 2.387 | 9.0 | **17.9** | 1.854 | 9.9 | 21.2 | 1.676 | 7.4 | 17.8 |
| DenseFormer | 2.383 | 9.5 | 18.7 | 1.830 | 9.5 | 20.7 | 1.641 | 7.9 | 18.5 |
| Hyper-conn. | 2.306 | 9.5 | 19.5 | 1.804 | 9.7 | 21.4 | 1.604 | 7.7 | 17.8 |
| mHC | 2.328 | 9.0 | 20.1 | 1.813 | 9.5 | 21.0 | 1.585 | 8.4 | 18.4 |
| MUDDFormer | 2.362 | 10.6 | 22.3 | **1.692** | 12.8 | 28.6 | 1.558 | 9.3 | 22.0 |
| DAGFormer | **2.185** | **7.6** | 22.2 | 1.724 | **9.2** | **20.2** | **1.507** | **6.9** | **16.5** |

More results, ablations and analyses are in the [paper](https://arxiv.org/abs/2610.19625).

## Code

| Path | Contents |
|---|---|
| [`src/model/olmo_graph.py`](src/model/olmo_graph.py) | DAGFormer backbone: OLMo-2 whose per-head query/key/value inputs and residual input mix the embedding and all earlier layer outputs, with the local refinement |
| [`src/model/predictor.py`](src/model/predictor.py) | Topology predictor: a small causal Transformer over the input tokens |
| [`src/model/modular_routing.py`](src/model/modular_routing.py) | Modular variant (paper appendix) |
| [`src/model/denseformer/`](src/model/denseformer), [`hyperconnection/`](src/model/hyperconnection), [`muddformer/`](src/model/muddformer) | Baselines |
| [`src/training/`](src/training), [`src/data/`](src/data), [`src/pruning/`](src/pruning) | Trainer, memory-mapped token data, pruning masks and schedule |
| [`scripts/pretokenize.py`](scripts/pretokenize.py) | Tokenizes Dolma v1.7 into memory-mapped shards |
| [`scripts/pretrain_dagformer.py`](scripts/pretrain_dagformer.py), [`scripts/pretrain_baseline.py`](scripts/pretrain_baseline.py) | Pretraining of DAGFormer and of the vanilla Transformer |
| [`scripts/pretrain_denseformer.py`](scripts/pretrain_denseformer.py), [`scripts/pretrain_muddformer.py`](scripts/pretrain_muddformer.py) | Baseline pretraining |
| [`scripts/prune_finetune.py`](scripts/prune_finetune.py) | Structured pruning during domain finetuning |
| [`scripts/eval_lm_harness.py`](scripts/eval_lm_harness.py) | Evaluation with the lm-evaluation-harness |
| [`configs/`](configs) | YAML configs, e.g. [`pretrain_300m/`](configs/pretrain_300m), [`pretrain_1b/`](configs/pretrain_1b), [`prune/`](configs/prune), [`locality/`](configs/locality) |
| [`tests/`](tests) | Unit tests |

The pretraining scripts for hyper-connections and mHC (`scripts/pretrain_hc_paper.py`, `scripts/pretrain_mhc.py`), instruction finetuning (`scripts/sft/`), chat evaluation (`scripts/chat_eval_*.py`) and the cost benchmark (`scripts/inference_bench.py`) are on the [`pruning/gradual-finetune`](https://github.com/AI-Interpretability-Illinois/dagformer/tree/pruning/gradual-finetune) branch.

## Getting started

```bash
pip install -r requirements-train.txt
python -m pytest tests -q
```

Tokenize the pretraining data once (a batch job; it streams Dolma v1.7 from the Hugging Face Hub):

```bash
python scripts/pretokenize.py --dataset allenai/dolma --dataset-version v1_7 \
    --out-dir <data_dir> --token-budget 21000000000 --seq-len 1024 --seed 0
```

Set `mmap_index_path` (the tokenized data) and `save_dir` (checkpoints) in a config, then train a matched pair, here at 300M on 4 GPUs:

```bash
torchrun --nproc_per_node=4 scripts/pretrain_dagformer.py --config configs/pretrain_300m/300m_fourway_corrected_mb4.yaml
torchrun --nproc_per_node=4 scripts/pretrain_baseline.py --config configs/pretrain_300m/300m_dense_mb4.yaml
```

Evaluate a checkpoint:

```bash
python scripts/eval_lm_harness.py --config configs/pretrain_300m/300m_fourway_corrected_mb4.yaml \
    --ckpt <save_dir>/checkpoint_step<N>.pt --tasks default --output_path results.json
```

The configs keep the cluster paths used for the paper; replace them with your own.

## Citation

```bibtex
@article{yang2026dagformer,
  title   = {{DAGFormer}: Global Topology Prediction for Transformer Cross-Layer Connections},
  author  = {Yang, Xiaocong and Hao, Yuren and Zhai, ChengXiang},
  journal = {arXiv preprint arXiv:2610.19625},
  year    = {2026}
}
```

## Acknowledgments

We thank the INVITE Institute at the University of Illinois Urbana-Champaign and the ACCESS program, through the National Center for Supercomputing Applications (NCSA), for their support with computing credits.

Development notes and the experiment history are in [`docs/project_notes.md`](docs/project_notes.md) and [`experiments/results.md`](experiments/results.md).
