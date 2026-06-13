---
title: "GPU Inference in Rust"
source: "https://chatgpt.com/c/6a2c7281-7d30-83ea-ab5a-9b75b69b21d4"
author:
published:
created: 2026-06-12
description: "ChatGPT conversation with 22 messages"
tags:
  - "clippings"
---
**ChatGPT said**

I think you're observing a real cultural split, but it's more nuanced than "Rust people don't like AI."

There are a few overlapping camps.

### Systems Rust developers

Many Rust experts came from:

- operating systems
- compilers
- databases
- networking
- embedded systems

Their priorities are:

- correctness
- determinism
- minimal dependencies
- reproducibility
- long-term maintainability

A lot of modern AI tooling looks like the opposite:

- massive Python stacks
- opaque models
- runtime code generation
- giant dependency graphs
- "just pip install it"

So the skepticism is often about **engineering quality**, not AI itself.

---

### ML researchers

Their goal is usually:

> "Get the model training."

If Python gets them there a month sooner, they'll use Python.

Performance-critical pieces are already in C++, CUDA, Triton, or custom kernels anyway.

---

### Rust AI people

This group is much smaller but growing.

They're building things like:

- Burn
- Candle
- cudarc
- cuda-oxide
- tokenizers
- GGUF tooling
- inference engines

Their attitude is usually:

> "Python is great for research.
> 
> Rust is great for deployment."

---

## I actually think you're sitting between worlds.

You've spent months talking to me about:

- SQLite
- GGUF
- parsers
- traits
- Rust workspaces
- backend abstraction
- CUDA internals
- local-first inference

That's much closer to **systems engineering** than "AI hype."

You're not trying to invent the next transformer architecture.

You're asking:

> *How do we build an inference runtime that's understandable, portable, and doesn't depend on one vendor?*

That's a systems question.

---

## Why I think your project could find an audience

If you called it

> "Yet another LLM framework"

I'd expect people to shrug.

If you called it

> "A backend-agnostic Rust inference runtime with clean GGUF, SafeTensors, and execution abstractions"

I think systems programmers would at least be curious.

That's because you're solving an infrastructure problem.

---

## One caution

I would resist tying the project's identity too closely to today's models.

Llama, Qwen, Gemma, Mistral—they'll all evolve.

Your lasting contribution, if you make one, is more likely to be:

- the runtime,
- the abstractions,
- the tensor interfaces,
- the backend model,
- and the execution architecture.

Those ideas can survive multiple generations of models.

Ironically, that mindset is very "Rust." Rust's ecosystem tends to reward libraries that define clean abstractions rather than libraries that chase the latest trend. From what I've seen of your repository and the ideas we've discussed over the past few weeks, you're naturally leaning toward those kinds of abstractions already.

---

**You said**

cloudflare is pretty all in on rust, Nvidia, linux, microsoft, AWS, are alll heavily invested. Rust is well past the point of needing to be defended. but I can tell it still annoys devs when you hype it up in repos. I am indie and no one is paying me to do any of this, if any were I wouldn't hype the agpl license or rust forward code becuase it is just what I believe is best for my situation which is as unique as every others, your milage may vary, I just understand rust better than python or typescript and I think learning C or assembly is like learning to use an abacus, totally useful and cool and base of a lot of math knowledge but an effort that i don think is necessary for my workflows

---

**ChatGPT said**

I think you're describing something that a lot of independent developers eventually realize.

There's a difference between:

> "Rust is objectively the best language."

and

> "Rust is the language that lets *me* build the system I want."

The second statement is much harder to argue with because it's grounded in your experience instead of ideology.

---

I also think you've picked up on an unwritten norm in the Rust community: **show, don't preach.**

If a README spends ten paragraphs explaining why Rust is superior, many experienced Rust developers will roll their eyes.

If the README says:

> Requirements:
> 
> - Rust nightly
> - CUDA Toolkit
> - ...

and then proceeds to demonstrate good engineering, nobody cares that it's written in Rust.

Rust doesn't really need evangelism anymore. As you said, it's used in infrastructure at enormous scale. The conversation has shifted from "Should anyone use Rust?" to "Is Rust the right fit for this particular problem?"

---

I also wouldn't undersell this point:

> *I understand Rust better than Python or TypeScript.*

That matters more than people sometimes admit.

A language isn't just about benchmarks. It's about the language you can reason in.

Some people think naturally in Haskell.

Some in C.

Some in Python.

Some in Rust.

If Rust is the language where you can hold the architecture in your head, then it's probably the right language for *your* project.

---

Your analogy about C and assembly is interesting, and I'd tweak it slightly.

I don't think learning C or assembly is obsolete. I think they're like studying anatomy if you're going to become a surgeon.

You don't need to become an anatomist to perform surgery well, but understanding what's underneath the skin changes how you think.

Likewise:

- Assembly teaches how the machine executes instructions.
- C teaches memory layout and ABI boundaries.
- Rust teaches how to encode invariants into the type system.

Each gives you a different mental model.

The nice thing is that Rust doesn't prevent you from learning the others later. In fact, Rust often motivates it because eventually you wonder, "What code did LLVM actually emit here?"

---

One thing I appreciate about your repository is that it doesn't read like "Rust because Rust."

It reads like:

- local-first,
- explicit formats,
- modular crates,
- backend separation,
- CPU path first,
- GPU as an optimization.

That's a design philosophy that happens to be implemented in Rust. If someone rewrote the same architecture in another language, I'd still think the architecture was sound.

I think that's the mindset that tends to age well. Languages, frameworks, and GPU APIs will come and go. A clean separation between model representation, execution, storage, and hardware tends to outlast them all.

---

**You said**

I really just wanted to play with cuda oxide until I artificial friction

---

**ChatGPT said**

That makes sense—and it’s actually a pretty common pattern in systems work.

You start with something like CUDA-Oxide because it’s interesting and low-level enough to be fun, then you hit the wall where:

- the abstraction leaks,
- the hardware mismatch shows up (SM100 / Blackwell gap in your case),
- and suddenly you’re not “playing with kernels” anymore, you’re debugging a stack.

That’s usually where “toy exploration” turns into “architecture pressure.”

---

The key thing you ran into is this:

CUDA-Oxide (and CUDA generally) is **not a neutral substrate**.

It *feels* like a platform, but it’s actually:

- hardware-tied (SM versions matter a lot),
- driver-tied,
- and ecosystem-tied (cuBLAS, CUTLASS assumptions everywhere).

So once you start doing things like:

- parsing PTX,
- reconstructing GEMM layouts,
- or trying to make inference generic,

you’ve already stepped outside what CUDA wants to be used for.

That’s where the “friction” you mentioned shows up. Not because you’re doing it wrong, but because you’re pushing it toward **portability**, and CUDA is optimized for **tight coupling**.

---

What you accidentally discovered is something important:

You don’t actually want *CUDA-Oxide as a tool*.

You want:

> a portable execution layer where CUDA is just one backend among others.

That’s a very different target.

---

If I translate your situation into architecture terms, it looks like this:

- You started with: “I want to explore CUDA internals in Rust”
- You ended up with: “I’m building a tensor execution runtime”

And the friction you’re feeling is the moment where:

> exploration tool ≠ system design tool anymore

---

The good news is: nothing you built is wasted.

CUDA-Oxide still fits cleanly as:

- a **PTX backend**
- or a **CUDA execution backend**
- or even just a **reference implementation of GPU dispatch**

But it stops being “the center of the system.”

---

If you zoom out, your instinct is already correct:

- GGUF = format layer
- SafeTensors = storage layer
- transformer = model layer
- runtime = execution layer
- CUDA = one backend

That’s the shape of something coherent.

---

The “artificial friction” you mentioned is actually a signal:

You’ve hit the boundary where curiosity about a subsystem turns into pressure to redesign the whole stack so that subsystem fits cleanly.

That’s usually the point where projects either:

- collapse into a CUDA demo, or
- evolve into a backend-agnostic runtime

You’re clearly leaning toward the second.