# One Architecture to Rule Them All
### Inferō: The Decode — Ch. 1 · Post 1.1
**by Chongming**

---

Before we can talk about LLM inference techniques, we need to answer a more basic question: why does the modern LLM look the way it does? Every optimization we'll cover in this chapter is fighting the same underlying physics. To understand why the solutions are shaped the way they are, you first need to understand the problem — and that means going back to the architectural decisions that brought us here.

By the end of this post, you'll understand why the transformer beat out its predecessors, why decoder-only became the dominant design, why next-token prediction turns out to be a surprisingly powerful training objective, and what the scaling laws tell us about where things are headed. These are the foundations that made modern LLM inference both necessary and genuinely hard.

---

## Part 1 — Why the Transformer Won

If you came into the LLM space after ChatGPT took off, you might have noticed something a little strange: all modern LLMs look basically the same. The parameter count varies, and Mixture of Experts (MoE) has grown more popular since DeepSeek published their models — but they're all transformer-based, and more specifically, decoder-only transformers. The transformer architecture itself dates back to 2017. So how did the field get here, and why did it converge so completely on one design?

![Figure 1: Transformer architecture (left) from Vaswani et al. (2017); RNN and LSTM unrolled across timesteps (right) from Olah (2015). The RNN's single tanh update causes gradients to vanish over 20–30 steps. The LSTM's gated cell state keeps the gradient path intact. The transformer processes all tokens in parallel via self-attention.](fig1-transformer-architecture.png)
*Figure 1: Transformer architecture (left) from Vaswani et al., "Attention Is All You Need" (2017). RNN and LSTM diagrams (right) from Olah, "Understanding LSTM Networks" (2015). The RNN's single tanh update causes gradients to vanish over 20–30 steps. The LSTM's additive cell state update (the horizontal line through the middle cell) keeps the gradient path intact — the same principle as residual connections in ResNet.*

The short answer is **scalability**.

### The world before 2017

In the early 2010s, the dominant approach to language modeling was the Recurrent Neural Network (RNN). RNNs felt like a natural fit: they process sequences one token at a time and maintain a hidden state $h_t$ that carries information forward as they go. Each token's representation depends on everything that came before it, all compressed into that running state.

The trouble showed up as sequences got longer. Training an RNN requires propagating gradients backward through every time step — a process called Backpropagation Through Time (BPTT). At each step, gradients get multiplied by a Jacobian matrix $\frac{\partial h_t}{\partial h_{t-1}}$. When those values are consistently below 1, the signal shrinks exponentially. Over just 20–30 timesteps, gradients can decay by a factor of $10^7$ or more, making early tokens essentially invisible to the learning process. This is the **vanishing gradient problem**, and it made RNNs practically useless for long sequences.

The Long Short-Term Memory network (LSTM), introduced in 1997 by Hochreiter and Schmidhuber, was the fix. It adds a separate cell state $c_t$ alongside the hidden state, controlled by three learnable gates — input $i_t$, forget $f_t$, and output $o_t$:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The key insight is that the cell state updates **additively** rather than multiplicatively — similar in spirit to residual connections in ResNet. Because $c_t = f_t \odot c_{t-1} + \ldots$ is a sum, the gradient through this path is just $f_t$, with no weight matrix or activation derivative compounding across timesteps. That's the gradient highway the RNN never had. With LSTMs, models could finally learn dependencies across hundreds of tokens, and the architecture became the standard tool in NLP throughout the 2010s.

But fixing the vanishing gradient problem didn't win the war. RNNs and LSTMs both shared a deeper flaw that would eventually cost them the race: **they're inherently sequential**.

For a sequence of $n$ tokens, the model has to process them one at a time — token $i+1$ can't start until token $i$ is done. On a GPU with thousands of parallel cores, almost all of them sit idle waiting for the previous step to finish. You're leaving most of your hardware unused. More critically, this puts a hard ceiling on how large and data-hungry your models can be — exactly the wrong constraint when the scaling era was just getting started.

### The transformer's answer

The transformer, introduced by Vaswani et al. in 2017, took a completely different approach. Rather than reading tokens one at a time, it processes the entire sequence **at once** using self-attention to learn how tokens relate to each other.

Here's how it works: for each token, self-attention computes a weighted sum over every other token in the sequence. The weights — attention scores — capture how relevant each other token is. They come from taking the dot product of queries $Q$ against keys $K$, scaling by $\sqrt{d_k}$ for numerical stability, applying a softmax to get a probability distribution, and using those weights to aggregate the values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

There's no sequential dependency anywhere in this. Everything runs as one large batched matrix multiplication — 2,048 tokens in a single forward pass, not 2,048 sequential steps. Every GPU core is doing useful work simultaneously.

This one architectural shift produces two compounding advantages:

**Parallelism means better hardware utilization.** GPUs and AI accelerators are built around massive parallel matrix multiplications. The transformer fits that perfectly. High hardware utilization combined with the ability to scale across more hardware makes it economically viable to train on dramatically more data.

**Every token has a direct connection to every other.** In an RNN or LSTM, two distant tokens can only communicate through the chain of hidden states between them — every hop is a chance for information to get lost. In a transformer, token 1 and token 1,000 have an equally direct connection through attention, regardless of how far apart they are. Long-range dependencies are no harder to learn than short ones.



The results were immediate. On the WMT 2014 English-to-German benchmark, the transformer hit 28.4 BLEU — more than 2 points ahead of the previous best, trained in a fraction of the time. Within a couple of years, it had largely displaced LSTMs across all of NLP.

---

## Part 2 — Why Decoder-Only Won

The original transformer had two halves: an encoder that reads the full input bidirectionally and builds a rich contextual representation, and a decoder that generates output one token at a time using cross-attention to query the encoder.

Researchers quickly found you could strip one half away. Keep just the encoder and you get BERT — excellent at understanding text, good for classification and similar tasks, but unable to generate. Keep just the decoder and you get GPT — purpose-built for text generation.

By 2023, the field had almost entirely settled on decoder-only. Every major frontier model — GPT-4, Claude, LLaMA, Qwen, DeepSeek — follows this design. Encoder-only and encoder-decoder architectures are still used in specialized contexts, but they've been pushed off the frontier. Here's why.

### Generation is the universal task

The clearest part of the story: encoder-only models can't generate text, and generation turns out to be the most generally useful thing a language model can do.

BERT-style models are trained with masked language modeling — mask some tokens, predict them from context. This produces great representations for understanding tasks. But the model always sees the full sequence and fills in blanks; there's no way to produce new text. You can't use it to write code, answer open-ended questions, or hold a conversation.

The idea of casting every NLP task as generation was formalized by Raffel et al. (2019) in the T5 paper: convert everything to text-to-text format, where both input and output are plain strings. Want a classifier? Generate the label. Want a translation? Generate the target sentence. Want a summary? Generate it. GPT-3 (Brown et al., 2020) then took this further, showing that a single large model could handle dozens of tasks through in-context learning — no task-specific fine-tuning required at all.

Once that framing takes hold, encoder-only models are simply out of the game. No matter how large you make BERT, it still can't generate. The paradigm shifted, and BERT couldn't follow.

Encoder-decoder models like T5 and BART can generate, and they were genuinely competitive for a while. But they lost on a different front: training efficiency.

### Training efficiency: why next-token prediction excels

![Figure 2: Wang et al. (ICML 2022) Table 3. After self-supervised pretraining, the causal decoder consistently outperforms encoder-decoder on zero-shot generalization across 30 tasks. Source: Wang et al. (2022).](fig2-decoder-vs-encoder-decoder.png)
*Figure 2: Table 3 from Wang et al. (ICML 2022). The causal decoder (FLM) leads on both EAI-Eval and T0-Eval benchmarks after self-supervised pretraining — even when the encoder-decoder is given twice the parameters to match compute. Source: Wang et al. (2022).*

This is the less obvious part of the story, but arguably the more important one.

**How encoder-decoder models train:** T5 uses span corruption — mask out random spans, replace with sentinel tokens, train the decoder to reconstruct them. The loss only applies to those reconstructed spans, which average around 15% of the sequence. The other 85% runs through the encoder and shapes the representations, but produces **no gradient signal**. Most of your training data contributes nothing to the weight updates.

**How decoder-only models train:** Next-token prediction on every token. Given any prefix, predict what comes next. For a sequence of $n$ tokens, you get $n$ gradient updates in a single forward pass — nothing is wasted. An encoder-decoder model processing 1,024 tokens might get gradients on ~150 of them; the decoder-only model uses all 1,024.

This might sound like an implementation detail. It isn't. With the same parameter count and compute budget, a decoder-only model sees far more training signal per step. The causal language modeling objective is remarkably data-efficient.

Wang et al. (ICML 2022) confirmed this rigorously. To keep the comparison fair, they gave the encoder-decoder model twice the parameters — enough to match the compute of a decoder-only model. Even then, after evaluating zero-shot generalization across 30 tasks, the decoder-only model came out ahead: causal models trained with a next-token prediction objective show the strongest zero-shot generalization after self-supervised pretraining.

That parameter gap carries directly into inference. To match a decoder-only model's performance, an encoder-decoder needs roughly $2\times$ the parameters. When two models perform similarly, the smaller one is always the better deployment choice — less memory, lower cost, faster serving.

One note: the encoder hasn't disappeared entirely. In multimodal models, vision encoders are very much alive — they convert images or audio into hidden states that a decoder attends to when generating text. The encoder lives on, just not in pure text models.

---

## Part 3 — Scaling Laws and the Bitter Lesson

A pattern runs through everything we've covered: **the architecture that scales wins**. The transformer replaced LSTMs because it could exploit hardware parallelism and grow with data. Decoder-only replaced encoder-decoder because it extracted more training signal from the same data. In both cases, the winner wasn't the most sophisticated design — it was the one that scaled better.

This isn't a coincidence. The scaling laws make it quantitative.

### The original scaling laws (Kaplan et al., 2020)

Kaplan et al. at OpenAI ran a systematic empirical study of how transformer performance relates to scale. They trained hundreds of models with varied parameter counts $N$, dataset sizes $D$, and compute budgets $C$ — spanning over seven orders of magnitude — and measured the resulting loss.

![Figure 3: Loss vs. compute (left), dataset size (center), and parameter count (right) — each a clean power law spanning over six orders of magnitude. Source: Kaplan et al., "Scaling Laws for Neural Language Models," OpenAI 2020.](fig3-scaling-laws.png)
*Figure 3: Figure 1 from Kaplan et al. (2020). Three smooth power laws, no saturation in sight. This was the empirical green light for GPT-3. Source: Kaplan et al. (2020).*

The result was clean: loss follows a **power law** with each variable independently:

$$\begin{aligned}
L(N) &\propto N^{-\alpha} \\
L(D) &\propto D^{-\beta} \\
L(C) &\propto C^{-\gamma}
\end{aligned}$$

where $\alpha$, $\beta$, and $\gamma$ are fitted constants. These relationships hold across many orders of magnitude with no signs of leveling off. The practical conclusions:

- More parameters → lower loss, predictably
- More data → lower loss, predictably
- More compute → lower loss, predictably
- Architectural details like width vs. depth matter far less than total scale

Most importantly, the curves showed **no saturation** within the ranges studied — no wall, no plateau, just smooth improvement as you scaled up. This was the empirical green light for GPT-3.

### The Chinchilla correction (Hoffmann et al., 2022)

Kaplan's results had a blind spot. Because parameters and data were varied somewhat independently, the apparent takeaway was: given a fixed compute budget, spend it mostly on parameters. This led to a generation of very large models trained on relatively little data. GPT-3 had 175B parameters but was trained on only about 300B tokens.

Hoffmann et al. at DeepMind ran a tighter experiment in 2022, training over 400 models from 70M to 16B parameters on 5B to 500B tokens, and optimizing the joint allocation of a fixed compute budget across both.

![Figure 4: IsoFLOP curves from Chinchilla. Left: for each FLOP budget, loss has a clear minimum at a specific model size. Center/right: optimal parameter count and token count both scale as C^0.5 — equal proportions. Source: Hoffmann et al., "Training Compute-Optimal Large Language Models," DeepMind 2022.](fig4-chinchilla-isoflop.png)
*Figure 4: Figure 3 from Hoffmann et al. (2022). Each IsoFLOP curve has a distinct valley — the optimal model size for that compute budget. Both optimal parameters and optimal tokens scale at the same rate (~C^0.5), meaning you should always grow data and model size equally. Source: Hoffmann et al. (2022).*

The finding: for compute-optimal training, **scale parameters and tokens roughly equally** — about 20 tokens per parameter. A 7B model should train on ~140B tokens; a 70B model on ~1.4T tokens.

By that standard, GPT-3 was massively undertrained. The proof: Chinchilla, a 70B model trained on 1.4T tokens, outperformed both Gopher (280B parameters) and GPT-3 (175B parameters) across the board using only a quarter of the parameters.

The field adjusted quickly. Llama 1 trained 7B parameters on 1T tokens. Llama 2 pushed to 2T. Llama 3 went to 15T tokens for 8B parameters — a 200:1 ratio, ten times beyond Chinchilla's recommendation. The reasoning: Meta deliberately kept the model small to make it cheap to serve. At millions of requests per day, inference cost dominates everything else, and a heavily overtrained small model can match a much larger one at a fraction of the cost.

### The data wall

The Chinchilla logic raises an uncomfortable follow-up: what happens when you run out of training data?

At NeurIPS 2024, Ilya Sutskever, co-founder of OpenAI, put it plainly: *"Pre-training as we know it will unquestionably end. We have but one internet, and the data is not growing."* He compared internet data to fossil fuel — accumulated over decades, consumed at scale, and not being replenished.

The field has responded in two ways: generating synthetic training data with capable models, and shifting focus toward scaling compute **at inference time** instead of at training time.

### Inference-time scaling

The scaling laws framed model quality as a function of $N \times D \times C$ during training. But they opened up a different question: can you scale compute **after** training and still get better outputs?

A model that reasons through 1,000 tokens before answering does more computation than one that jumps straight to a response. A system that samples 64 candidate answers and picks the best one uses $64\times$ the inference compute of single-sample decoding. The question is whether that extra compute actually buys better results.

It does — and by 2025, inference-time compute scaling has become just as critical as training-time scaling. We'll spend all of Chapter 2 on this.

### The Bitter Lesson

In 2019, Richard Sutton — one of the founders of reinforcement learning — published a short essay called "The Bitter Lesson." Looking back across 70 years of AI research, he found the same pattern playing out again and again:

> *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation."*

Researchers keep building systems that encode human knowledge — handcrafted chess heuristics, linguistic rules, domain-specific features. Those systems work well for a while. Then compute gets cheap enough that a simpler, more general method trained on more data overtakes them. AlphaZero learned chess from scratch through self-play, no human knowledge required. Hand-designed vision features were replaced by learned convolutional networks. Expert NLP pipelines were replaced by large pretrained language models.

The lesson is "bitter" because it's uncomfortable: carefully accumulated human knowledge keeps losing to methods that just scale better. Sutton's prescription: build **general methods** that improve predictably as compute increases, rather than baking in human knowledge that compute will eventually make obsolete. His two candidates: **search** and **learning**.

The transformer is exactly that kind of method. Self-attention has no built-in assumptions about syntax, word order, or language structure — it learns what to attend to purely from data. And it scales predictably with compute. Whatever eventually replaces the transformer will share this property: a general method that gets better with scale.

The scaling laws and the Bitter Lesson are the same story told from different angles. Compute is the primary lever. General methods beat specialized ones. And the architecture that won is the one that lets you apply compute most efficiently.

---

## What's Next

We've traced the arc from RNNs to decoder-only transformers — the architecture underlying almost every modern LLM and every inference technique in this series. In the next post, we'll get into how inference actually works: the prefill and decode stages, the KV cache, and why serving these models at scale is such a hard engineering problem.

---

## References

1. Vaswani et al., "Attention Is All You Need" (NeurIPS 2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Olah, "Understanding LSTM Networks" (2015) — [colah.github.io/posts/2015-08-Understanding-LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. Hochreiter & Schmidhuber, "Long Short-Term Memory" (Neural Computation, 1997)
4. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (JMLR 2020) — [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
5. Brown et al., "Language Models are Few-Shot Learners" (NeurIPS 2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
6. Wang et al., "What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?" (ICML 2022) — [proceedings.mlr.press/v162/wang22u/wang22u.pdf](https://proceedings.mlr.press/v162/wang22u/wang22u.pdf)
7. Kaplan et al., "Scaling Laws for Neural Language Models" (OpenAI, 2020) — [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
8. Hoffmann et al., "Training Compute-Optimal Large Language Models" (DeepMind, 2022) — [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
9. Sutskever, NeurIPS 2024 talk — "Pre-training as we know it will unquestionably end"
10. Sutton, "The Bitter Lesson" (2019) — [incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
