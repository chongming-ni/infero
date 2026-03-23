# Inferō: The Decode
## A Blog Series on LLM Inference Engineering
### by Chongming

> *From algorithms to silicon — everything that makes language models fast, smart, and deployable at scale*

**Series tagline:** LLM inference, fully decoded.

---

Training a language model is the headline. **Inference is the story.**

Every time a user gets a response — every token, every millisecond, every dollar spent on GPU time — that's the inference stack at work. And yet inference is where most of the hard engineering lives: the memory walls, the attention bottlenecks, the hardware trade-offs, the reasoning loops that run inside the model before a single word appears on screen.

**Inferō: The Decode** is a 53-post deep-dive for ML engineers and researchers who want to understand the full stack — from KV caches and speculative decoding, to RLVR and test-time compute, to TPUs and wafer-scale silicon, to the attention variants reshaping frontier models, to multimodal and diffusion inference, and finally to the long-horizon agentic workloads that break all single-turn assumptions.

The name says what this series does: *inferre* — to carry the conclusion forward, to bring the hidden into the open. One topic at a time.

---

## Series Overview

| Chapter | Title | Subtitle | Posts |
|---------|-------|----------|-------|
| Ch. 1 | **The Machine Room** | Production Inference: building systems that serve at scale | 12 |
| Ch. 2 | **The Thinking Layer** | RL Inference: making models reason harder, not just faster | 10 |
| Ch. 3 | **The Foundry** | AI Accelerators: the silicon that shapes everything above | 8 |
| Ch. 4 | **The Attention Wars** | Attention Variants: the mechanism is no longer settled | 4 |
| Ch. 5 | **Beyond the Word** | Multimodal & Generative Inference | 11 |
| Ch. 6 | **The Long Game** | Agentic Inference: when a single query becomes a multi-hour task | 8 |

**Total: 53 posts**

---

## Naming Conventions

| Element | Format | Example |
|---------|--------|---------|
| Series name | Inferō: The Decode | Full title on covers and headers |
| Short form | Inferō | Navigation, bylines, social handles |
| Post slug | `infero/1.2-kv-cache-optimization` | Chapter.Post kebab-case |
| Post byline | Inferō: The Decode — Ch. 1 · Post 1.2 | Under each post title |
| Social tagline | LLM inference, fully decoded. | Twitter/X, LinkedIn footers |
| Chapter callout | Part of *The Machine Room* — Ch. 1 of Inferō: The Decode | Top of each post |

---

# Chapter 1 — The Machine Room
## Production Inference: building systems that serve at scale

*Theme: Speed, cost, and throughput at serving time. Every technique fights the same enemy — the memory bandwidth wall — from a different angle.*

---

### Post 1.1 — Anatomy of Inference

The foundation post everyone needs before diving into techniques.

**What's covered:**
- Autoregressive decoding: why each token requires a full forward pass
- Prefill (parallel, compute-bound) vs. decode (sequential, memory-bandwidth-bound)
- The memory wall: why decode is bottlenecked by HBM bandwidth, not compute FLOPs
- The KV cache: what it is, why it grows linearly, why it dominates GPU memory
- The roofline model and arithmetic intensity
- Key metrics: TTFT, TBT, throughput vs. latency, goodput

**Key Papers:**
- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- Kaplan et al., "Scaling Laws for Neural Language Models" (OpenAI, 2020)

---

### Post 1.2 — KV Cache Optimization

The single most impactful area in production inference today.

**What's covered:**
- Naive KV cache: memory fragmentation and the original vLLM problem
- **PagedAttention**: virtual memory paging for KV cache
- **Prefix caching**: reusing KV states for shared system prompts
- **KV cache quantization**: storing cached states at INT8/FP8
- **KV cache eviction**: H2O, StreamingLLM attention sinks, importance-based eviction
- Distributed KV cache: LMCache for multi-turn agents
- **Prompt compression**: LLMLingua — reduce input tokens 4–20× before the KV cache

**Key Papers:**
- ⭐ Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference" (EMNLP 2023)

---

### Post 1.3 — Attention Head Variants

The KV sharing ladder — now the industry default.
*(Moved from Ch. 4: GQA and MLA are mainstream production techniques.)*

**What's covered:**
- **MHA → MQA → GQA**: the KV sharing progression — GQA is now standard in Llama 3, Mistral, Qwen, GPT-4, Claude
- **Multi-Head Latent Attention (MLA)**: low-rank KV compression — DeepSeek V3/R1, spreading to Kimi K2.5, GLM-5
- GQA retains 99% of MHA quality; MLA offers superior quality at engineering complexity cost
- How head variant choice directly affects KV memory, decode latency, and batch size

**Key Papers:**
- ⭐ Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) — invented MQA
- ⭐ Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (EMNLP 2023)
- DeepSeek-AI, "DeepSeek-V2 Technical Report" (2024) — introduced MLA

---

### Post 1.4 — Sliding Window Attention & Attention Sinks

Bounded KV cache for long contexts — and the fix for its failure mode.
*(Moved from Ch. 4: deployed in Mistral 7B, TensorRT-LLM, OpenAI models.)*

**What's covered:**
- **SWA**: each token attends to `w` nearest neighbors — bounded KV cache
- Mistral 7B: GQA + SWA rolling buffer = 8× smaller KV cache vs. Llama 2
- The **attention sink problem**: naive SWA fails when early tokens are evicted — perplexity spikes
- **StreamingLLM**: permanently keep KV of first 4 tokens alongside rolling window — stable at 4M+ tokens
- Production: Mistral 7B, TensorRT-LLM, HuggingFace Transformers, OpenAI open-weight models

**Key Papers:**
- ⭐ Jiang et al., "Mistral 7B" (2023)
- ⭐ Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)

---

### Post 1.5 — Long Context: RoPE Scaling & Position Interpolation

Extending context windows to 128K–1M tokens without full retraining.

**What's covered:**
- Why standard RoPE breaks at context lengths beyond training
- **Position Interpolation**: compress positions by a scaling factor
- **NTK-aware RoPE**: Neural Tangent Kernel-based interpolation
- **YaRN**: the practical recipe used by Mistral, Llama, Qwen for context extension
- How Llama 3.1 (128K) and Llama 4 Scout (10M tokens) achieve their context lengths
- The "lost in the middle" problem: extended context ≠ ability to use it

**Key Papers:**
- Chen et al., "Extending Context Window via Positional Interpolation" (2023)
- Peng et al., "YaRN: Efficient Context Window Extension" (ICLR 2024)

---

### Post 1.6 — Batching Strategies

How smarter request scheduling transforms GPU utilization.

**What's covered:**
- Static, dynamic, and **continuous batching** (the Orca/vLLM breakthrough)
- Chunked prefill: interleaving prefill and decode to reduce TTFT spikes
- Request scheduling: priority queues, SLA-aware routing, decode-length prediction
- 10–50× higher throughput vs. naive single-request serving

**Key Papers:**
- ⭐ Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)
- Agrawal et al., "Sarathi-Serve: Taming Throughput-Latency Tradeoff with Chunked Prefill" (OSDI 2024)

---

### Post 1.7 — Speculative Decoding & Multi-Token Prediction

Generating multiple tokens at once — two different approaches.

**What's covered:**
- **Speculative decoding**: draft model proposes, large model verifies — lossless via rejection sampling
- Variants: classic · Medusa (prediction heads) · EAGLE (draft in target feature space) · self-speculative
- Typical speedup: 1.5–3× latency reduction
- **Multi-Token Prediction (MTP)**: train the model to predict N tokens simultaneously — DeepSeek-V3, Meta FAIR
- MTP vs. speculative decoding: training-time vs. inference-time technique

**Key Papers:**
- ⭐ Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework" (2024)
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
- ⭐ Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction" (Meta FAIR, 2024)

---

### Post 1.8 — Quantization

Shrinking models without losing their mind.

**What's covered:**
- Precision ladder: FP16 → INT8 → INT4 → FP8
- PTQ: **GPTQ**, **AWQ** — quantize after training with minimal accuracy loss
- The outlier problem: **LLM.int8()**, **SmoothQuant**
- FP8 in production: NVIDIA Hopper/H100, DeepSeek-V3's FP8 training recipe
- 4-bit PTQ maintains near full-precision quality on most tasks

**Key Papers:**
- ⭐ Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (NeurIPS 2022)
- Frantar et al., "GPTQ" (ICLR 2023)
- Lin et al., "AWQ: Activation-aware Weight Quantization" (MLSys 2024)
- Xiao et al., "SmoothQuant" (ICML 2023)

---

### Post 1.9 — Model Parallelism & Disaggregated Inference

Splitting the model across GPUs — and physically separating prefill from decode.

**What's covered:**
- **Tensor Parallelism**, **Pipeline Parallelism**, **Sequence Parallelism**, **Expert Parallelism**
- Communication overhead: NVLink vs. InfiniBand tradeoffs
- **Disaggregated Prefill/Decode (PD)**: physically separate phases across different hardware
- Why PD matters: prefill is compute-bound (H100s), decode is memory-bound (different chips)
- Production: DistServe, llm-d, AWS Trainium + Cerebras CS-3

**Key Papers:**
- ⭐ Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models" (2019/2021)
- ⭐ Zhong et al., "DistServe: Disaggregating Prefill and Decoding" (OSDI 2024)

---

### Post 1.10 — FlashAttention & Kernel Optimizations

Making the attention mechanism fast at the hardware level.

**What's covered:**
- Why vanilla attention is slow: full n×n matrix forces HBM round-trips
- **FlashAttention**: IO-aware tiling — compute in SRAM, never materialize full matrix
- **FlashAttention-2**: 2× speedup, better parallelism and work partitioning
- **FlashAttention-3**: H100-optimized, async pipeline, FP8 support
- **FlashInfer**: flexible kernels for heterogeneous serving (variable-length, paged KV)
- Kernel fusion: fusing adjacent operations to reduce memory bandwidth

**Key Papers:**
- ⭐ Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- Dao, "FlashAttention-2" (ICLR 2024)
- Shah et al., "FlashAttention-3" (NeurIPS 2024)

---

### Post 1.11 — Pruning, Distillation, MoE & Model Routing

Making the model smaller and smarter about when to use its full capacity.

**What's covered:**
- **Structured pruning**: NVIDIA Minitron — pruning + distillation outperforms training from scratch
- **Knowledge distillation**: DeepSeek-R1 distilled variants
- **Mixture of Experts (MoE)**: activate a fraction of parameters per token — Mixtral, DeepSeek-V3, Llama 4
- Early exit / adaptive depth
- **Model routing / cascades**: RouteLLM, FrugalGPT — easy queries to 3B, hard to 70B

**Key Papers:**
- ⭐ Jiang et al., "Mixtral of Experts" (2024)
- Muralidharan et al., "Compact Language Models via Pruning and Knowledge Distillation (Minitron)" (NeurIPS 2024)

---

### Post 1.12 — Serving Frameworks

Putting all the techniques together in real production systems.

**What's covered:**
- **vLLM**: PagedAttention + continuous batching — the open-source standard
- **TensorRT-LLM**: CUDA kernel fusion, FP8, graph compilation — fastest on H100
- **SGLang**: RadixAttention, best for structured generation
- **TGI**: Rust-based HuggingFace production server
- Comparison: latency, throughput, ease of use, hardware compatibility

**Key Systems:**
- ⭐ vLLM / Kwon et al. (SOSP 2023)
- SGLang / Zheng et al. (MLSys 2024)

---

# Chapter 2 — The Thinking Layer
## RL Inference: making models reason harder, not just faster

*Theme: Test-time compute as a new axis of model capability — from simple sampling to pure RL-trained reasoning.*

---

### Post 2.1 — Test-Time Compute Scaling

The paradigm shift from "train bigger" to "think longer."

**What's covered:**
- Pre-training scaling laws and their diminishing returns
- The core question: how much can a model improve with more inference-time compute?
- Two mechanisms: searching over outputs vs. extending reasoning chain length
- Inference-time compute as a tradeoff against pre-training compute

**Key Papers:**
- ⭐ Snell et al., "Scaling LLM Test-Time Compute Optimally" (ICLR 2025 Oral)
- Wu et al., "Inference Scaling Laws" (ICLR 2025)

---

### Post 2.2 — Chain-of-Thought and Its Variants

Teaching models to show their work.

**Key Papers:**
- ⭐ Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (NeurIPS 2022)
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (ICLR 2023)
- Yao et al., "Tree of Thoughts" (NeurIPS 2023)

---

### Post 2.3 — Reward Models & Verifiers

Teaching models what "good" looks like — and using verifiers at inference time.

**Key Papers:**
- ⭐ Lightman et al., "Let's Verify Step by Step" (OpenAI, ICLR 2024) — PRMs at scale
- Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021) — foundational ORM
- Wang et al., "Math-Shepherd" (ACL 2024)

---

### Post 2.4 — Best-of-N, Voting & Adaptive Compute

Simple but surprisingly powerful test-time scaling.

**Key Papers:**
- ⭐ Wu et al., "Inference Scaling Laws" (ICLR 2025)
- Fu et al., "Dynasor: Certaindex for Adaptive Reasoning Compute" (2024)

---

### Post 2.5 — Tree Search at Inference Time

Bringing AlphaGo-style search to language generation.

**Key Papers:**
- Silver et al., "Mastering the game of Go" (Nature 2016) — the AlphaGo inspiration
- ⭐ Wu et al., "Inference Scaling Laws — REBASE" (ICLR 2025)

---

### Post 2.6 — RLHF & PPO

The origins of reasoning via reinforcement learning.

**Key Papers:**
- ⭐ Ouyang et al., "InstructGPT: Training Language Models to Follow Instructions" (NeurIPS 2022)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)

---

### Post 2.7 — GRPO & RLVR: How DeepSeek-R1 Changed Everything

The paper of 2024–25 for RL inference.

**Key Papers:**
- ⭐ DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning" (Nature 2025)
- Shao et al., "DeepSeekMath" (2024) — introduced GRPO

---

### Post 2.8 — Token Efficiency & Overthinking

Not all thinking tokens are equal.

**Key Papers:**
- Pfau et al., "Let's Think Dot by Dot" (COLM 2024)

---

### Post 2.9 — Distilling Reasoning Into Smaller Models

Making reasoning accessible without the full compute cost.

**Key Papers:**
- ⭐ DeepSeek-AI, "DeepSeek-R1" distillation section (2025)
- Ho et al., "Large Language Models as Reasoning Teachers" (ACL 2023)

---

### Post 2.10 — The Unified View

Where production inference and RL inference converge — and what comes next.

**What's covered:**
- Serving challenges for reasoning models with long outputs
- KV cache management for 100K+ token reasoning chains
- Test-Time Reinforcement Learning (TTRL)
- Future: self-improving agents, online RL inference

---

# Chapter 3 — The Foundry
## AI Accelerators: the silicon that shapes everything above

*Theme: The hardware substrate that determines which inference optimizations are even possible.*

---

### Post 3.1 — Why GPUs Dominate (and Their Limits)

**What's covered:** CUDA ecosystem lock-in, memory bandwidth bottleneck, NVIDIA roadmap (H100 → Blackwell), AMD MI300X, Intel Gaudi 3, NVLink / InfiniBand networking.

---

### Post 3.2 — Google TPUs

**What's covered:** Systolic array architecture, v7 Ironwood (4,614 TFLOPS/chip), optical circuit switches, GCP-only constraints, JAX/XLA tradeoffs.

**Key Paper:** Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017)

---

### Post 3.3 — AWS Trainium & Inferentia

**What's covered:** Trainium2 (2.52 PFLOPS FP8/chip), 500K+ chip Anthropic deployment, AWS + Cerebras disaggregated architecture, Neuron SDK, cloud lock-in tradeoff.

---

### Post 3.4 — Groq LPU

**What's covered:** Compiled SRAM architecture, 80 TB/s on-die bandwidth, 1,665 tok/s with speculative decoding, deterministic execution, inference-only limitation.

---

### Post 3.5 — Cerebras WSE

**What's covered:** 4 trillion transistors, 900K AI cores, wafer-scale eliminates inter-chip comms, 125 PFLOPS peak, AWS + Cerebras disaggregated decode role.

---

### Post 3.6 — Tenstorrent & Open-Source Silicon

**What's covered:** RISC-V + Tensix cores, Blackhole (745 TFLOPS FP8, $999), IP licensing to LG/Hyundai/Samsung, tt-Metalium open toolchain.

---

### Post 3.7 — Edge AI Accelerators

**What's covered:** Apple ANE (18 TOPS), Qualcomm Hexagon NPU, Hailo-8/10H, on-device LLM requirements, fragmented software stack.

---

### Post 3.8 — Photonics, Neuromorphic & Next-Gen Architectures

**What's covered:** Lightmatter (photonic, $850M raised), D-Matrix (in-memory compute), Intel Loihi (neuromorphic), optical interconnects already in TPU pods, the von Neumann bottleneck.

---

# Chapter 4 — The Attention Wars
## Attention Variants: the mechanism is no longer settled

*Theme: As of 2026, every major lab has made a different bet. This chapter maps the bets and gives an honest verdict.*

---

### Post 4.1 — The Quadratic Problem & the Design Space

**What's covered:** Why softmax attention can't scale infinitely. The dual-mode ideal: parallel training + O(1) inference. The design spectrum: Softmax → Sparse → Linear → SSM → Hybrid. Where each frontier model sits.

---

### Post 4.2 — Linear Attention & Gated DeltaNet

**What's covered:** Katharopoulos (2020) → RetNet → GLA → DeltaNet → Gated DeltaNet (ICLR 2025). The 3:1 hybrid pattern. Qwen3.5, Kimi Linear in production. MiniMax's rollback — the honest counterpoint.

**Key Papers:**
- ⭐ Katharopoulos et al., "Transformers are RNNs" (ICML 2020)
- ⭐ Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025)

---

### Post 4.3 — Mamba & SSM Hybrids

**What's covered:** Selective state spaces, Mamba-2, the 1:7 hybrid ratio, Jamba 1.5, Nemotron-H (3× throughput), Bamba-9B, FalconMamba.

**Key Papers:**
- ⭐ Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (2023)
- Dao & Gu, "Transformers are SSMs" / Mamba-2 (ICML 2024)
- ⭐ Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model" (ICLR 2025)

---

### Post 4.4 — Sparse Attention & The Honest Verdict

**What's covered:** DeepSeek Sparse Attention (DSA), GLM-5 (MLA + DSA). The verdict matrix: what's production-ready vs. still a bet. The meta-lesson on workload-dependent architecture choice.

---

# Chapter 5 — Beyond the Word
## Multimodal & Generative Inference

*Theme: What changes when models process images, video, and generated media — and the unique inference challenges each modality brings.*

---

### Post 5.1 — How Multimodal LLMs Work

Late fusion (LLaVA, BLIP-2) vs. early fusion (GPT-4o, Llama 4). Token count problem: 336×336 image = 576 tokens; 1080p tiled = 2,000–5,000 tokens.

---

### Post 5.2 — The Visual Token Bottleneck

Spatial and semantic redundancy. KV cache explosion from visual tokens. The inverse scaling result: fewer tokens + larger LLM wins.

---

### Post 5.3 — Visual Token Compression

Q-Former, Resampler, TokenPacker. FastV (prune after layer 2), SparseVLM (54% FLOP reduction, 97% accuracy), ATP-LLaVA, Matryoshka Multimodal Models.

---

### Post 5.4 — High-Resolution & Tiling

Dynamic tiling (LLaVA-HD). Qwen2.5-VL Window-Attention ViT (any resolution natively). VL-CACHE modality-aware KV eviction. MiniCPM-V on mobile.

---

### Post 5.5 — Video Inference

Temporal redundancy — consecutive frames 95%+ identical. LLaMA-VID (2 tokens/frame). PruneVid temporal+spatial merging. STORM (NVIDIA) long video inference.

---

### Post 5.6 — Vision Encoders

CLIP vs. SigLIP vs. DINOv2 vs. native tokenization (GPT-4o). Qwen2.5-VL Window-Attention ViT. Trade-offs in flexibility, OCR quality, and inference cost.

---

### Post 5.7 — Multimodal KV Cache

Modality asymmetry: visual KV entries become less important as generation progresses. VL-CACHE modality-aware eviction. Shared prefix caching for repeated-image workflows. Inf-MLLM streaming on one GPU.

---

### Post 5.8 — How Diffusion Models Work

Denoising loop vs. autoregressive generation. Latent diffusion (VAE + denoiser). CFG doubling cost. UNet → DiT backbone shift (SD3, FLUX, Sora).

**Key Papers:**
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- ⭐ Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)" (ICCV 2023)

---

### Post 5.9 — Accelerating Diffusion Inference

DDIM (10–50× step reduction). DPM-Solver (10–20 quality steps). Consistency models (1–4 steps). SDXL-Turbo, FLUX.1-schnell. Quality-speed tradeoff map.

**Key Papers:**
- ⭐ Song et al., "Denoising Diffusion Implicit Models (DDIM)" (ICLR 2021)
- Lu et al., "DPM-Solver" (NeurIPS 2022)
- ⭐ Song et al., "Consistency Models" (ICML 2023)

---

### Post 5.10 — Video Diffusion & the Temporal Challenge

Temporal attention for frame consistency. DiT at video scale (Sora). Efficient video generation: sliding window temporal attention, frame caching, distillation.

**Key Papers:**
- Ho et al., "Video Diffusion Models" (NeurIPS 2022)
- OpenAI, "Sora: Video generation models as world simulators" (2024)

---

### Post 5.11 — Frontier Multimodal Architectures

GPT-4o (unified tokenizer) · Gemini 2.5 Pro (frozen SigLIP-ViT) · Llama 4 (early fusion) · Qwen2.5-VL (Window-ViT + MoE) · InternVL2 (CLIP + DINOv2) · DeepSeek-VL2 (MoE routing). Two camps: unified tokenization vs. vision tower + projector.

---

# Chapter 6 — The Long Game
## Agentic Inference: when a single query becomes a multi-hour task

*Theme: Agentic inference breaks every assumption from chapters 1–5. It's about completing multi-step tasks taking hours, not serving individual responses in milliseconds.*

---

### Post 6.1 — The Anatomy of an Agentic Workload

The ReAct loop in production. A 2-hour SWE-bench run: 50–200 LLM calls, 200K+ tokens per trajectory. Why "tokens/sec" is the wrong metric — task completion rate per dollar is what matters.

---

### Post 6.2 — The Context Explosion Problem

How context grows across turns: history + tool outputs + reasoning traces. "Lost in the middle" at 100K+ tokens. Strategies: truncation, summarization, RAG for agent memory, LLMLingua compression.

---

### Post 6.3 — KV Cache for Agents: Persistence, Reuse & TTL

KV cache lives across many turns and idles during tool calls. **Continuum** KV TTL scheduling. **KVFlow** for multi-agent workflows. Plan-level caching. The isolation requirement: concurrent agents must not share state.

---

### Post 6.4 — Tool Call Latency: The Bottleneck Nobody Optimizes

Tool execution = 60–80% of wall-clock time in production traces. Sequential vs. **parallel tool calls** (up to 4× speedup). Async execution and synchronization. The critical path vs. total token cost distinction.

---

### Post 6.5 — Multi-Agent Orchestration

When to spawn sub-agents. Critical execution path — not total token count — determines latency. DAG-based orchestration. Agent routing to specialist models. 8 smaller-model runs often beats 1 large-model run.

---

### Post 6.6 — Agentic Memory

In-context (KV), external (vector DB), parametric (fine-tuned). **EM-LLM** episodic memory. **MemOS**: agent memory as an OS — working memory, long-term storage, cold archive.

---

### Post 6.7 — Cost Management at Scale

200-step trajectory × $0.01/call × 1,000 users/day = $2K/day on one workflow. Adaptive compute allocation. **Certaindex** (Dynasor) for measuring reasoning progress. Budget-constrained orchestration. Diverse ensemble agents outperform a few state-of-the-art ones.

---

### Post 6.8 — Serving Infrastructure for Agents

Session-aware routing (llm-d: 87% cache hit rate, 88% faster TTFT). State isolation for concurrent agents. Fault tolerance for 2-hour trajectories. **MCP (Model Context Protocol)** as the emerging tool connectivity standard. Kubernetes + Argo Workflows at 8K parallel runs.

---

## Complete Post Index

| Post | Title | Chapter |
|------|-------|---------|
| 1.1 | Anatomy of Inference | The Machine Room |
| 1.2 | KV Cache Optimization | The Machine Room |
| 1.3 | Attention Head Variants (MHA → GQA → MLA) | The Machine Room |
| 1.4 | Sliding Window Attention & Attention Sinks | The Machine Room |
| 1.5 | Long Context: RoPE Scaling & Position Interpolation | The Machine Room |
| 1.6 | Batching Strategies | The Machine Room |
| 1.7 | Speculative Decoding & Multi-Token Prediction | The Machine Room |
| 1.8 | Quantization | The Machine Room |
| 1.9 | Model Parallelism & Disaggregated Inference | The Machine Room |
| 1.10 | FlashAttention & Kernel Optimizations | The Machine Room |
| 1.11 | Pruning, Distillation, MoE & Model Routing | The Machine Room |
| 1.12 | Serving Frameworks | The Machine Room |
| 2.1 | Test-Time Compute Scaling | The Thinking Layer |
| 2.2 | Chain-of-Thought and Its Variants | The Thinking Layer |
| 2.3 | Reward Models & Verifiers | The Thinking Layer |
| 2.4 | Best-of-N, Voting & Adaptive Compute | The Thinking Layer |
| 2.5 | Tree Search at Inference Time | The Thinking Layer |
| 2.6 | RLHF & PPO | The Thinking Layer |
| 2.7 | GRPO & RLVR: How DeepSeek-R1 Changed Everything | The Thinking Layer |
| 2.8 | Token Efficiency & Overthinking | The Thinking Layer |
| 2.9 | Distilling Reasoning Into Smaller Models | The Thinking Layer |
| 2.10 | The Unified View | The Thinking Layer |
| 3.1 | Why GPUs Dominate (and Their Limits) | The Foundry |
| 3.2 | Google TPUs | The Foundry |
| 3.3 | AWS Trainium & Inferentia | The Foundry |
| 3.4 | Groq LPU | The Foundry |
| 3.5 | Cerebras WSE | The Foundry |
| 3.6 | Tenstorrent & Open-Source Silicon | The Foundry |
| 3.7 | Edge AI Accelerators | The Foundry |
| 3.8 | Photonics, Neuromorphic & Next-Gen Architectures | The Foundry |
| 4.1 | The Quadratic Problem & the Design Space | The Attention Wars |
| 4.2 | Linear Attention & Gated DeltaNet | The Attention Wars |
| 4.3 | Mamba & SSM Hybrids | The Attention Wars |
| 4.4 | Sparse Attention & The Honest Verdict | The Attention Wars |
| 5.1 | How Multimodal LLMs Work | Beyond the Word |
| 5.2 | The Visual Token Bottleneck | Beyond the Word |
| 5.3 | Visual Token Compression | Beyond the Word |
| 5.4 | High-Resolution & Tiling | Beyond the Word |
| 5.5 | Video Inference | Beyond the Word |
| 5.6 | Vision Encoders | Beyond the Word |
| 5.7 | Multimodal KV Cache | Beyond the Word |
| 5.8 | How Diffusion Models Work | Beyond the Word |
| 5.9 | Accelerating Diffusion Inference | Beyond the Word |
| 5.10 | Video Diffusion & the Temporal Challenge | Beyond the Word |
| 5.11 | Frontier Multimodal Architectures | Beyond the Word |
| 6.1 | The Anatomy of an Agentic Workload | The Long Game |
| 6.2 | The Context Explosion Problem | The Long Game |
| 6.3 | KV Cache for Agents | The Long Game |
| 6.4 | Tool Call Latency | The Long Game |
| 6.5 | Multi-Agent Orchestration | The Long Game |
| 6.6 | Agentic Memory | The Long Game |
| 6.7 | Cost Management at Scale | The Long Game |
| 6.8 | Serving Infrastructure for Agents | The Long Game |

---

*Inferō: The Decode — LLM inference, fully decoded.*
*Published on GitHub and Substack. Last updated: March 2026.*
