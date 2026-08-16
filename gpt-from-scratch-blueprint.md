# Build a GPT From Scratch — Complete Blueprint

**Goal:** understand and build a working GPT-style transformer in PyTorch, from foundations you don't have yet, without watching all 7 videos in Karpathy's series first.

**Your pace assumption:** ~6 focused hours/day.

**End state:** a transformer you built and understand, trained on data you chose, with a benchmark you can defend — and a resume bullet where every word survives an interview.

---

## How to use this document

Go top to bottom. Don't skip ahead. Each phase unlocks the next.

The rule for **every** video session below: **code editor open, typing as he types, pausing constantly.** You are not watching a movie. A 2-hour video is really a week of 30–45 minute build-sessions where you pause, type, break things, and fix them. If you only watch, you'll *feel* like you learned it and then freeze in front of a blank file. Typing it yourself is the whole point.

### The struggle rule

- 🟢 **Push through (sit with it 30+ min before looking anything up).** Not understanding why a gradient flows the way it does. Confusion about what a tensor operation produced. Why the loss isn't going down. These *are* the learning — the struggle is the skill forming.
- 🔴 **Bail fast (10 min max).** Python/PyTorch install conflicts. CUDA/MPS device errors during setup. Exact deprecated function names. These teach nothing. When stuck here, wipe and restart clean — don't treat it as a puzzle.

### Tracking progress

Every `- [ ]` is a checkbox. Tick them as you go. The **checkpoints** are the real units — if you can't answer a checkpoint out loud in your own words, redo the block before moving on. Checkpoints are how you catch "I watched it" masquerading as "I learned it."

---

## The map: five phases

- [ ] **Phase 0** — Setup (30–60 min)
- [ ] **Phase 1** — Understand backpropagation (read below, ~30 min)
- [ ] **Phase 2** — Prerequisites: micrograd + makemore bigram (2.5 days)
- [ ] **Phase 3** — The main event: build GPT (2 days)
- [ ] **Phase 4** — Make it yours: custom data + systems optimization + benchmark (2–3 days)

**Which of Karpathy's 7 videos you actually need:**

| Lecture | Title                            | Watch it?                                |
| ------- | -------------------------------- | ---------------------------------------- |
| 1       | micrograd (autograd engine)      | ✅ **Required**                           |
| 2       | makemore part 1 (bigram)         | ✅ **Required** (at least the first ~2/3) |
| 3       | makemore part 2 (MLP)            | ⚪ Optional — helpful, skippable          |
| 4       | makemore part 3 (batchnorm)      | ⛔ Skip                                   |
| 5       | makemore part 4 (backprop ninja) | ⛔ Skip                                   |
| 6       | makemore part 5 (WaveNet)        | ⛔ Skip                                   |
| 7       | **Let's build GPT**              | ✅ **The main event**                     |

You need **1, 2, and 7.** That's it. All videos are linked from Karpathy's course page: https://karpathy.ai/zero-to-hero.html (repo: https://github.com/karpathy/nn-zero-to-hero).

---

## Phase 0 — Setup

You're on a Mac, which is fine for everything up through Phase 3 and for a small Phase 4 training run (Apple Silicon uses PyTorch's **MPS** backend instead of CUDA).

- [ ] Install Python 3.10+ and create a clean virtual environment
- [ ] `pip install torch numpy matplotlib jupyter` — verify with `python -c "import torch; print(torch.__version__)"`
- [ ] Confirm your accelerator: `python -c "import torch; print(torch.backends.mps.is_available())"` — should print `True` on Apple Silicon
- [ ] Get comfortable running a Jupyter notebook (Karpathy works in notebooks) OR a plain `.py` file — either is fine

🔴 **Any install pain is bail-fast territory.** If `pip` fights you for more than 10 minutes, start from a fresh venv. Do not debug dependency hell.

**Note for Phase 4:** a small model trains fine on MPS, but if you want faster training or a bigger run, a free/cheap cloud GPU (Colab, or a rented L4/A10G) will be much faster than your Mac. Decide when you get there.

---

## Phase 1 — Understand backpropagation *first*

Read this before you touch micrograd. It's the single idea everything rests on. Walking into the micrograd video already holding this concept is what makes the video click instead of wash over you.

### Backpropagation, the actual intuition

Forget the scary name. Backprop answers one question: **"each knob in my network is currently set to some value — if I nudge this knob a tiny bit, does my error get better or worse, and by how much?"** A network learns by turning every knob in the direction that reduces error, a little at a time. Backprop is just the efficient way to figure out, for *every* knob at once, which way to turn it.

**1. A neural network is a big nested function.** Input goes in, gets multiplied by weights (the knobs), gets added and squished, over and over, and a prediction comes out. Then you compare the prediction to the right answer and compute a single number: the **loss** (how wrong you were). So the whole thing is: knobs → math → loss. Training means finding knob values that make loss small.

**2. The gradient is just "slope."** For one knob, imagine plotting loss as you slowly turn that knob. You get a curve. At your current setting, that curve has a slope. If the slope is positive, turning the knob up *increases* loss (bad — turn it down). If negative, turning it up *decreases* loss (good). The steepness tells you how *much* that knob matters right now. The gradient is just the collection of these slopes, one per knob. "Compute the gradient" = "find the slope of the loss with respect to every knob."

**3. The chain rule is how the slope travels backward.** A knob deep in the network doesn't touch the loss directly — it affects some intermediate value, which affects another, which eventually affects the loss. So its influence is a *chain*. The chain rule says: to get the slope of the loss with respect to an early knob, you multiply the slopes along that chain. Concretely — if `a` affects `b`, and `b` affects loss, then (how much `a` affects loss) = (how much `a` affects `b`) × (how much `b` affects loss). You just multiply the local slopes along the path.

**4. Backprop = applying the chain rule from the loss backward, reusing work.** This is why it's called *back*prop. You start at the loss (slope with respect to itself is trivially 1), then walk *backward* through the network. At each step you already know the slope for the value ahead of you, so you just multiply by the one local slope at the current step to get the slope for the value behind you. You reuse the downstream result instead of recomputing the whole chain for every knob. That reuse is the entire trick — it's what makes training a million-knob network feasible instead of impossibly slow.

Then the learning step itself is dead simple: for every knob, `new_value = old_value − (small number) × slope`. The small number is the **learning rate**. Nudge every knob slightly downhill on the loss curve, repeat thousands of times, and the network gets less wrong. That loop — forward to get loss, backward to get slopes, nudge every knob downhill — *is* training. Every model, including the GPT you'll build, is that same loop.

**The one image to keep:** you're standing on a hilly landscape (the loss), in the dark, and the gradient is you feeling which way is downhill under each foot. You take a small step downhill. Repeat until you're in a valley. People literally call this gradient *descent*.

**Vocabulary you now own:** *forward pass* = run input through to get the loss. *Backward pass* = the backward walk computing all the slopes. That's the whole vocabulary for now.

- [ ] I can explain backprop in my own words using the "knobs and slopes" framing
- [ ] I understand that micrograd's job is to *automate* this backward walk

---

## Phase 2 — Prerequisites (≈ 2.5 days)

### Video 1: micrograd (Lecture 1) — 4 sessions

You build an autograd engine by hand. This is where backprop stops being a concept and becomes code you wrote.

- [ ] **Session 1A** (~45 min) — Intro through the `Value` object and basic operations (add, multiply).
  - **Checkpoint:** I can explain what the `Value` object stores and *why* each number needs to "remember" its parents. (That remembering is what makes the automatic backward walk possible.)

- [ ] **Session 1B** (~45 min) — Manual backprop through a small expression by hand, then the `backward()` method.
  - **Checkpoint:** I can trace one backward step myself — given the slope ahead, I compute the slope behind by multiplying the local slope. (This is the chain rule from Phase 1, now in code.)

- [ ] **Session 1C** (~45 min) — Building a neuron, then a layer, then an MLP out of `Value` objects.
  - **Checkpoint:** I can explain how a single neuron turns inputs into an output, and name what its knobs are.

- [ ] **Session 1D** (~40 min) — The training loop: forward, backward, nudge, repeat — and watching the loss go down.
  - **Checkpoint (the most important one so far):** I can point at each line of the loop and name it — "this is the forward pass, this is backward, this is the knob nudge." This loop is the skeleton of *every* model you'll ever train.

**🔑 Consolidation exercise (do NOT skip):**
- [ ] Close the video. Open a blank file. **Rebuild micrograd's core from memory** — the `Value` object, `backward()`, and a tiny training loop — without looking. Struggle here for real (🟢). When you can do this unaided, you *own* backprop. This is the difference between "I followed along" and "I can build it."

### Video 2: makemore part 1 / bigram (Lecture 2) — 3 sessions

Now you shift from micrograd's hand-built engine to **real PyTorch tensors**. Same concepts, production tool. This gives you the tensor fluency the GPT video assumes you already have.

- [ ] **Session 2A** (~45 min) — Loading data, building bigram counts, the sampling idea.
  - **Checkpoint:** I understand what the model predicts — the next character from the current one — and how text becomes numbers.

- [ ] **Session 2B** (~45 min) — `torch.Tensor` mechanics: indexing, broadcasting, the operations he leans on.
  - **Checkpoint:** I can create a tensor, index into it, and explain what broadcasting does. (This is the PyTorch fluency the GPT video takes for granted.)

- [ ] **Session 2C** (~40 min) — Recasting the bigram model as a tiny neural net trained with gradient descent.
  - **Checkpoint (your green light for the GPT video):** I recognize the training loop from Session 1D, now written in real PyTorch. When "it's the same loop" clicks, you're ready.

> You can stop the makemore video once the neural-net bigram is training. You don't need the rest of Lecture 2, and you're skipping Lectures 3–6 entirely. Lecture 3 (MLP) is *optional* if you want extra tensor practice — helpful, not required.

**Phase 2 done when:** you rebuilt micrograd unaided AND you can point at a PyTorch training loop and name every line.

---

## Phase 3 — The main event: build GPT (Lecture 7, ≈ 2 days)

This is "Let's build GPT: from scratch, in code, spelled out." You now have the foundation, so only the genuinely new part — **attention** — will be new. Everything else you'll recognize.

- [ ] **Session 3A** (~50 min) — Setup, data loading, the bigram baseline in this new codebase, the batch/block structure.
  - **Checkpoint:** I understand how the data is chunked into context blocks and batches, and what the model is being asked to predict at each position.

- [ ] **Session 3B** (~60 min) — **Self-attention** — the mathematical trick, then queries, keys, values. *This is the conceptual heart.* Go slow. Pause. Rewatch parts.
  - **Checkpoint:** I can explain, in my own words, what Q, K, and V do and why attention lets each token "look at" other tokens. If this is fuzzy, redo the session — everything downstream depends on it.

- [ ] **Session 3C** (~50 min) — Multi-head attention, the feed-forward layer, residual connections, layer norm.
  - **Checkpoint:** I can explain why residual connections help deep networks train, and what layer norm is doing. I can name the pieces of one transformer block.

- [ ] **Session 3D** (~50 min) — Stacking blocks into the full model, scaling up, the final training run, generating text.
  - **Checkpoint:** I can trace one forward pass through the whole model — tokens in, through embedding, through the blocks, to a probability distribution over the next token — and I understand the generation loop that samples one token at a time.

**🔑 The big consolidation exercise (this is what makes the resume claim TRUE):**
- [ ] Close the video. Open a blank file. **Rebuild the GPT from scratch, unaided.** You'll get stuck; that's 🟢, push through. Refer to your Phase-3 notes but NOT the video. When you can produce a working transformer from an empty editor, you have genuinely built a GPT from scratch — not followed a tutorial. **This is the single most important step in the entire document.**

**Milestone reached:** after this exercise you can honestly put *"implemented a GPT-style transformer from scratch in PyTorch"* on your resume. Phase 4 makes it stronger.

---

## Phase 4 — Make it yours (≈ 2–3 days)

A completed tutorial is not a portfolio piece. These steps turn it into one — and Steps 2–3 are where you show the **ML-platform/infra** flavor that matches your target roles.

### Step 1 — Train on data YOU chose (½–1 day)
- [ ] Swap Shakespeare for a dataset you actually care about (song lyrics, a codebase, your own writing, a book you like — anything that isn't the tutorial default). Someone trained theirs on classical Tamil text; pick something that's yours.
- [ ] Get it training and generating recognizable output.
  - **Checkpoint:** I can explain what changed when I swapped datasets (vocab size, tokenization) and I have sample generated output saved.

### Step 2 — Bring in the systems/infra layer (1 day) — *this is your differentiator*
Pick the ones you can do well on your hardware:
- [ ] Add **mixed-precision training** (`torch.autocast`) and measure the speedup. On Mac note the MPS vs. CUDA device-type quirks — understanding them is itself good interview fodder.
- [ ] Add **gradient accumulation** so you can simulate larger batch sizes.
- [ ] Profile training with `torch.profiler` to find where time actually goes.
- [ ] (If you use a cloud GPU) try `DistributedDataParallel` or just document the single-GPU throughput carefully.
  - **Checkpoint:** I can explain what mixed precision does and *why* it's faster, and I can point at my profiler output and say where the bottleneck was.

### Step 3 — Benchmark and prove it (½ day) — *the credibility multiplier*
- [ ] Build a small benchmark harness: measure training throughput (tokens/sec) **before and after** your Step 2 optimizations.
- [ ] Produce a results table and, ideally, a plot. These numbers must be **real measurements you can reproduce** — not invented. (Interviewers probe suspiciously precise numbers hardest.)
  - **Checkpoint:** I have a before/after table, I can reproduce it on command, and I can explain what caused the improvement.

### Step 4 — Polish (½ day)
- [ ] Clean README with a short architecture description and one diagram of the model / training flow.
- [ ] Commit generated-text samples and the benchmark table.
- [ ] Push to GitHub with clear commit history and a `requirements.txt` so it runs with one command.
  - **Checkpoint:** a stranger could clone my repo, read the README, and understand what I built and why in two minutes.

---

## Timeline at ~6 hours/day

| Day | Focus | Resume status |
|---|---|---|
| **1** | Phase 0 setup + micrograd 1A–1C | — |
| **2** | micrograd 1D + rebuild micrograd unaided + makemore 2A | — |
| **3** | makemore 2B–2C + consolidation | Prereqs done |
| **4** | GPT video 3A–3B (through self-attention) | — |
| **5** | GPT video 3C–3D (full model + training) | — |
| **6** | **Rebuild GPT from a blank file, unaided** | ✅ *"Built a GPT from scratch in PyTorch"* is now true |
| **7** | Phase 4 Step 1 — custom dataset, train, debug | Portfolio-worthy |
| **8** | Phase 4 Step 2 — mixed precision, profiling | Stronger |
| **9** | Phase 4 Step 3–4 — benchmark, README, push | ✅ Full bullet, defensible |

**Bottom line:** you cross the first resume threshold on **Day 6** (a from-scratch GPT you can defend). You have the **full, differentiated bullet by Day 9.**

Realistically, budget slippage — debugging, life, sessions that run long, the rebuild exercises taking two tries. **Plan for ~9 focused days of work spread across ~2 calendar weeks.** If you have less than 6 hrs some days, stretch accordingly; the *sequence* matters more than the exact days.

---

## Your resume bullet

**After Day 6 (honest, solid):**
> Implemented a GPT-style transformer language model from scratch in PyTorch (self-attention, multi-head attention, residual connections), trained on a custom text corpus.

**After Day 9 (differentiated, infra-flavored — your target):**
> Implemented a GPT-style transformer from scratch in PyTorch; optimized training throughput [N]× via mixed-precision and gradient accumulation, with profiler-guided tuning and a reproducible benchmark harness.

Fill `[N]` with your **real measured number**. Never a placeholder, never invented.

## The 6 things you must be able to explain cold

If you can answer these in your own words, the bullet is genuinely yours and every interview follow-up is one you can handle:

1. **Backprop** — the knobs/slopes/chain-rule story, and why it's called "back."
2. **The training loop** — forward → loss → backward → optimizer step, line by line.
3. **Self-attention** — what Q, K, V do, and why attention beats older approaches.
4. **Why residuals and layer norm** — what breaks in deep networks without them.
5. **The generation loop** — why text is produced one token at a time (autoregressive), and how that differs from training.
6. **Your optimization** — what mixed precision actually does and why your benchmark improved.

---

## Where to go next (optional, after this is done)

- Karpathy's **"Let's reproduce GPT-2" / build-nanogpt** (https://github.com/karpathy/build-nanogpt) — takes your toy GPT to a real GPT-2 reproduction, heavy on the systems/infra side you care about.
- The **inference-serving** path (vLLM) — the layer that sits directly on top of a trained model and is closest to the ML-platform role you're targeting. Your from-scratch build makes the KV-cache and prefill/decode concepts there click from the inside.

---

*Track your progress by ticking the boxes. The checkpoints are the real test — if you can't say it out loud, redo the block. The two "rebuild from a blank file" exercises are non-negotiable; they're what convert watching into knowing.*
