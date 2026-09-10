<p align="center"><img src="imgs/logo.png" width="720" alt="PocketVLA"></p>
<h1 align="center">PocketVLA: Building a 3M VLA from Absolute Zero</h1>

<p align="center">
  <a href="../README.md">简体中文</a> | English
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/torch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white" alt="torch">
  <img src="https://img.shields.io/badge/transformers-4.57%2B-FFD21E?logo=huggingface&logoColor=white" alt="transformers">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="license">
</p>

## Introduction

🚀 **Embodied AI is entering a period of explosive growth**, and VLA (Vision-Language-Action models) are one of the core technical paradigms driving this wave. This project builds a **VLA with only 3M parameters entirely from scratch** ✨ — the setup is light enough to train on an ordinary GPU 💻 and fast enough that ⚡ the full pipeline finishes within 10 minutes. PocketVLA consists of a vision encoder 👁️, a language-model trunk 🧠, and an action head 🦾 — the same structure as mainstream VLAs; it also provides an end-to-end implementation of **data generation → SFT behavior cloning → GRPO reinforcement learning → inference evaluation → OOD generalization evaluation** 🔄, so you can genuinely understand every piece of the VLA algorithm at minimal cost, instead of staying at the conceptual level 💡.


**🎉 This project includes:**

- 🧪 **Ultra-lightweight simulation environment** (`pvla/sim/`): a pure-NumPy 2-link planar-arm "pick & place" environment (forward kinematics, grasp/release semantics, snap-on grasping) + a 64×64 observation renderer with GIF replay export — zero heavy dependencies, runs on CPU.
- 🤖 **VLA model implemented from scratch** (`pvla/model/vla.py`): a VLM early-fusion architecture of vision encoder + language-model trunk + action head, fully compatible with HuggingFace `AutoModel` / `AutoTokenizer`.
- 🎯 **Analytic expert & data generation** (`pvla/expert/` + `pvla/data/`): a closed-form IK + dual-rate proportional control + state-machine expert; one command generates SFT demos, the fixed test set, GRPO RL scenario sets, and OOD test sets.
- 📚 **SFT behavior-cloning training** (`pvla/train.py`): action MSE + class-balanced grip loss + auxiliary-head supervision.
- 🚀 **GRPO reinforcement learning** (`pvla/grpo.py`): grouped closed-loop rollouts, advantage-weighted regression, and a KL anchor against policy drift.
- 📊 **Closed-loop inference evaluation** (`pvla/evaluate.py`): a single sweep over the fixed test set — per-scenario replay GIFs, failure-attribution statistics, per-color success-rate summary figures, and expert-vs-VLA world-coordinate trajectory comparison.
- 🌗 **OOD generalization evaluation** (`pvla/eval_ood.py` + `pvla/scenarios.py`): examines the model's out-of-distribution generalization.
- 📦 **Open model weights & datasets**: pretrained SFT / GRPO checkpoints (HuggingFace format, [download](#download)) plus all 5 training/evaluation datasets (~65MB).

## Model Architecture


<p align="center"><img src="imgs/arch.png" width="1800" alt="model architecture"></p>

**Design highlights**:

1. **Standard VLM fusion**: the vision encoder outputs vision tokens,
   which are linearly projected into the LM embedding space and concatenated with text tokens, a proprioception token, and a learnable
   action-query token before jointly passing through the shared language-model trunk; vision and language interact thoroughly via token-level bidirectional attention.
2. **LM output wired directly into the action head**: the action head reads only the hidden state of the action-query token in the LM output and decodes it through an MLP + tanh into actions — the LM hidden layer is the sole input to the action pathway; vision / language / proprioception all influence the action only through the fused trunk.
3. **Auxiliary heads**: from the fused vision-token hidden states, two independent lightweight readout heads (ball / bin) each output the
   ball/bin world coordinates via Conv + attention-weighted 8×8 grid centers. They provide only deep-supervision auxiliary losses and interpretability (never on the action pathway), and can be switched off with `use_aux_heads: false` — **how much do they matter for success rate?**
   See the [ablation study](#ablation-study) below.
4. **Class-balanced losses**: the grip is weighted by inverse frequency over 4 (holding × label) groups; ball-offset regression is weighted over 4 normalized-distance buckets.
  
## Downstream Task: 2D Grab

<p align="center">
  <img src="imgs/ep000.gif" width="200" alt="task example 1">
  <img src="imgs/ep007.gif" width="200" alt="task example 2">
  <img src="imgs/ep022.gif" width="200" alt="task example 3">
</p>

The embodiment is a **2-link planar arm** performing language-guided closed-loop "pick & place" tasks in a top-down 2D scene. Each element is defined as follows:

| Element | Definition |
|:---:|---|
| 🔵 **Scene** | Red / green / blue 3 balls + 1 bin; positions randomized every episode |
| 💬 **Instruction** | Natural language, e.g. `put the blue ball into the bin` as shown in the GIFs |
| 👁️ **Observation** | 64×64×3 top-down view (bin included, target never marked) + instruction tokens + proprioception `(q1, q2, ee_x, ee_y, gripper, holding)` |
| 🦾 **Action** | `(Δq1, Δq2, grip)` joint deltas (clamped to `±0.15 rad`); `grip > 0` closes / `≤ 0` opens |
| 🤏 **Grasp** | Closing the gripper snaps on a ball when the end-effector is `< 0.09` from its center (any color may be grasped; wrong grasps must be recovered by the policy itself) |
| 🧺 **Place** | Ball released and **fully inside the bin** → success ✅; at most `90` steps per episode |

**Why it is hard:** the three balls differ only in color, so the model must truly **understand the color word in the instruction** and localize the matching ball in the image; once in the placing phase, it must then localize the bin in the image as well. From **understanding the language** to **seeing the target** to **coordinating the arm to act** — this is VLA in a nutshell.


## Environment Setup

```bash
# Create and activate a conda environment
conda create -n pocketvla python=3.10 -y
conda activate pocketvla

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full pipeline
bash scripts/run_all.sh
```

Or run it step by step:

```bash
# 0. Dataset generation: train/val/test/grpo/ood_test in one shot (built-in defaults)
python3 -m pvla collect

# 1. Stage 1: SFT behavior cloning -> outputs_sft_3m/train/
python3 -m pvla train --config configs/sft_3m.yaml

# 2. Closed-loop eval on the fixed test set (GIFs + failure attribution + figures) -> outputs_sft_3m/eval/
python3 -m pvla evaluate --config configs/sft_3m.yaml

# 3. Stage 2: GRPO RL, fine-tunes the SFT checkpoint (full test-set eval runs automatically afterwards)
python3 -m pvla grpo --config configs/grpo_3m.yaml

# 4. OOD generalization eval (unseen distractor)
python3 -m pvla.eval_ood --config configs/sft_3m.yaml
python3 -m pvla.eval_ood -c configs/sft_3m.yaml --ckpt outputs_grpo_3m/pocketvla-3m-grpo
```

## Download

### Pretrained Models

All models are in HuggingFace format. From the repo root, first `import pvla.model` to register the custom classes, then load them with
`AutoModel.from_pretrained(..., trust_remote_code=True)`:

<table>
  <tr><th>Model</th><th>Params</th><th>Stage</th><th>Closed-loop success rate</th><th>Download</th></tr>
  <tr><td>pocketvla-3m-sft-2d-grab</td><td>3.02M</td><td>SFT</td><td>74.2%</td><td><a href="https://huggingface.co/penpenzh/pocketvla-3m-sft-2d-grab">🤗 HuggingFace</a></td></tr>
  <tr><td>pocketvla-3m-grpo-2d-grab</td><td>3.02M</td><td>SFT + GRPO</td><td><strong>80.2%</strong></td><td><a href="https://huggingface.co/penpenzh/pocketvla-3m-grpo-2d-grab">🤗 HuggingFace</a></td></tr>
</table>

### Datasets

All data is generated online by the simulator (see [Dataset Generation](#dataset-generation)); it can also be downloaded directly:

<table>
  <tr><th>File</th><th>Scale</th><th>Download</th></tr>
  <tr><td><code>train.npz</code></td><td>7000 episodes / 184,364 samples</td><td><a href="https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab">🤗 HuggingFace</a></td></tr>
  <tr><td><code>val.npz</code></td><td>3000 episodes / 79,150 samples</td><td><a href="https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab">🤗 HuggingFace</a></td></tr>
  <tr><td><code>test.npz</code></td><td>500 scenarios</td><td><a href="https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab">🤗 HuggingFace</a></td></tr>
  <tr><td><code>grpo.npz</code></td><td>3000 scenarios</td><td><a href="https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab">🤗 HuggingFace</a></td></tr>
  <tr><td><code>ood_test.npz</code></td><td>300 scenarios</td><td><a href="https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab">🤗 HuggingFace</a></td></tr>
</table>

### Quick Inference Example

After downloading a model, load and run inference with the standard HuggingFace interface — feed in one 64×64 top-down image +
one natural-language instruction + 6-dim proprioception, and get a 3-dim action `(Δq1, Δq2, grip)` (joint deltas in radians + gripper open/close):

```python
import pvla.model  # register PocketVLA with AutoModel/AutoTokenizer (run from repo root)
import torch
from transformers import AutoModel, AutoTokenizer

ckpt = "penpenzh/pocketvla-3m-grpo-2d-grab"   # HF repo id, or a local dir like outputs_grpo_3m/pocketvla-3m-grpo
model = AutoModel.from_pretrained(ckpt, trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

# --- inputs: image / instruction / proprioception ---
image = torch.rand(1, 3, 64, 64)          # (1, 3, 64, 64) top-down view in [0, 1]
L = model.config.lang_len                 # instruction length (10)
enc = tok("put the green ball into the bin", padding="max_length", truncation=True, max_length=L)
tokens = torch.tensor([enc["input_ids"]]) # (1, L), zero-padded / truncated
pad = tokens != 0                         # padding masked out of attention
proprio = torch.tensor([[0.5, -0.3, 0.2, 0.4, 0.0, 0.0]])
# (q1/pi, q2/pi, ee_x/1.1, ee_y/1.1, gripper, holding)

# --- forward ---
with torch.no_grad():
    action, ball_off, bin_off = model(image, tokens, pad, proprio)

print("action (dq1, dq2, grip):", [round(x, 3) for x in action[0].tolist()])
# e.g. [0.082, -0.031, -1.0]: joint deltas in rad; grip > 0 closes, <= 0 opens
print("aux readout (ball offset, bin offset):",
      [round(x, 3) for x in ball_off[0].tolist()],
      [round(x, 3) for x in bin_off[0].tolist()])
```

> 💡 Local checkpoint directories (e.g. `outputs_grpo_3m_pat/pocketvla-3m-grpo`) work the same way;
> `ball_off / bin_off` are the ball/bin offsets relative to the end-effector output by the auxiliary readout heads (normalized coordinates), used for visualization
> ; once you have an action, hand it to the simulator for closed-loop execution: `obs, _, done, info = env.step(action[0].numpy())` — the episode succeeds if the ball lands in the bin within 90 steps.

## Main Experiments

### Dataset Generation

All data is generated online by the simulator; no external datasets are needed:
`python3 -m pvla collect` 

**How it works**

- **Demonstration data** (`train/val.npz`): the 2-link planar arm admits an analytic solution — closed-form inverse kinematics (IK, with a fixed
  elbow-down branch) converts a target world coordinate directly into joint angles; combined with dual-rate proportional control and a gripper state machine (not holding → grasp the commanded ball / holding the right ball → carry to the bin / holding the wrong ball → release first and recover), this yields a privileged expert with a 100% success rate. Each step records the `(observation, instruction tokens, proprioception, expert action)` tuple; joint actions are normalized to [-1,1], ball/bin offsets relative to the end-effector are attached as auxiliary supervision labels, and a tiny σ=0.002 rad action noise is added to increase demo diversity.
- **Scenario sets** (`test/grpo/ood_test.npz`): each entry stores only one initial scenario
  `(ball positions, bin position, initial joints, instruction)` — during evaluation/RL the environment replays that exact initial state,
  and the policy executes the rest of the episode closed-loop on its own.



<table>
  <tr><th>File</th><th>Scale</th></tr>
  <tr><td><code>train.npz</code></td><td>7000 episodes / 184,364 samples</td></tr>
  <tr><td><code>val.npz</code></td><td>3000 episodes / 79,150 samples</td></tr>
  <tr><td><code>test.npz</code></td><td>500 scenarios</td></tr>
  <tr><td><code>grpo.npz</code></td><td>3000 scenarios</td></tr>
  <tr><td><code>ood_test.npz</code></td><td>300 scenarios</td></tr>
</table>

### Stage 1 — SFT

**Behavior-cloning loss** :

```
L = MSE(Δq1, Δq2)                    # joint action regression
  + 0.3 · w_grip · MSE(grip)         # class-balanced grip loss (action_grip_weight)
  + 1.0 · w_ball · MSE(ball_off)     # aux: commanded-ball offset (aux_target_weight)
  + 1.0 · MSE(bin_off)               # aux: bin offset (aux_bin_weight)
```

- `w_grip`: grip labels are weighted by inverse frequency over **4 (holding × label) groups** (√frequency, mean-normalized) — close/release are the few critical samples;
  without weighting, training degenerates into an "always open / always closed" degenerate solution
- `w_ball`: ball offsets are weighted by inverse frequency over **4 normalized-distance buckets** (0.08 / 0.2 / 0.5 / 1.6) — otherwise small-offset samples dominate
  and long-range visual localization suffers; both class-balancing terms can be disabled with `balanced_losses: false` for ablation
- Checkpoint selection: `ctrl = val_dq_mse + val_tgt_err + val_bin_err` (the grip term has high variance and is excluded from selection);
  whenever ctrl hits a new low, an HF-format checkpoint is saved to `train.output_dir/pocketvla-<size>/`
- The offset errors of the two spatial readout heads (ball / bin) serve as **deep-supervision** auxiliary losses (never on the action pathway); training curves are
  auto-plotted as three panels ("total loss / ball readout / bin readout", log axis) with the best-val selection point annotated:

<p align="center"><img src="imgs/loss_curve.png" width="860" alt="training curve"></p>

### Stage 2 — GRPO Reinforcement Learning (decoupled stage) 

Each iteration samples G group members playing full closed-loop episodes on each of `steps_per_domain` (default 20) in-distribution
scenarios (Gaussian exploration noise σ=0.03 is added only on the 2 joint dims; grip stays discrete).
Rewards are computed from the outcome (success +1.0 dominant + faster +0.1 + grasped-right +0.2 + landing-distance penalty −0.1 + grasped-wrong −0.3)
and group-normalized into advantages; a group is skipped if its within-group advantage variance is zero, otherwise only members with positive advantages receive **advantage-weighted regression** (the regression targets are the de-noised policy means — noisy actions of bad members are never learned), on top of a **KL anchor** (squared-distance form ‖μ_θ − μ_SFT‖², keeping the policy from drifting away from SFT).
The top-level yaml `grpo:` section controls all hyperparameters; a checkpoint is saved whenever the mean group reward hits a new high, with early stopping after `early_stop`
consecutive iterations without improvement (set to 20 in the 3m yaml). After training, the GRPO checkpoint with the **best mean group reward** (rather than the last-step policy) is automatically and fully evaluated on the 500-scenario test set, and `gifs` (30) random scenarios are exported as replay GIFs to `outputs_grpo_<size>/eval/case/`.

### Results

Closed-loop evaluation on the fixed 500-scenario test set (strictly comparable — the same test set; pat denotes an independent repeat run with the same configuration, used to gauge the impact of training stochasticity):

<table>
  <tr><th>Model</th><th>Stage</th><th>Closed-loop success rate</th><th>Grasp-right rate</th><th>Mean landing distance</th><th>Mean steps if success</th></tr>
  <tr><td>pocketvla-3m-sft-2d-grab</td><td>SFT</td><td>74.2%</td><td>79.0%</td><td>0.251</td><td>30.8</td></tr>
  <tr><td>pocketvla-3m-grpo-2d-grab</td><td>SFT + GRPO</td><td><strong>80.2%</strong></td><td>83.4%</td><td>0.220</td><td>29.4</td></tr>
</table>



### Ablation Study


**Why do the auxiliary heads matter?**

1. **Deep supervision improves visual-localization learning**: the ball/bin readout losses directly supervise the fused vision-token hidden states
   so the trunk "sees exactly where the target is", effectively adding a localization-related gradient path to the trunk and easing the sparsity of the single action-MSE signal.
2. **Interpretability & failure attribution**: the readout heads output ball/bin positions, training curves are plotted per panel (ball-head/bin-head errors),
   and failures can be attributed to "couldn't localize the ball" vs "couldn't localize the bin".
3. **A stability anchor for the GRPO stage**: RL rewards act only on the final action (success/steps/grasp-right/distance); auxiliary-head outputs
   receive no reward signal; their consistency regularizer (anchored to the frozen SFT outputs, weight `aux_weight`) is what directly keeps the trunk's localization representations from degrading during RL.

Under exactly the same configuration and data:

<table>
  <tr><th>Variant</th><th>Auxiliary loss</th><th>Closed-loop success rate</th><th>Grasp-right rate</th><th>Mean landing distance</th><th>Mean steps if success</th></tr>
  <tr><td>SFT (with aux heads)</td><td>ball offset + bin offset</td><td><strong>74.2%</strong></td><td><strong>79.0%</strong></td><td><strong>0.251</strong></td><td><strong>30.8</strong></td></tr>
  <tr><td>SFT (no aux heads, noaux)</td><td>—</td><td>6.0%</td><td>19.0%</td><td>0.876</td><td>46.4</td></tr>
</table>

With the auxiliary heads off, the success rate collapses from **74.2% to 6.0%** (−68.2 points), the grasp-right rate drops 79.0% → 19.0%,
the mean landing distance grows 0.251 → 0.876 (the arm almost never approaches the target), and successful episodes take 30.8 → 46.4 steps.
  




## Out-of-Distribution Generalization Test (OOD)

Generalization is examined under visual conditions never seen during training: the scene contains an extra **distractor in a color that never appears in training**
(random from yellow / purple / orange / cyan / pink / brown / white / gray; the shape is randomly a ball or a
square); the instructions still refer only to the three colors seen in training.

**Experiment log** (closed-loop evaluation on the 300-scenario OOD test set):

<table>
  <tr><th>Model</th><th>Stage</th><th>OOD success rate</th><th>Grasp-right rate</th><th>Mean landing distance</th><th>Mean steps if success</th></tr>
  <tr><td>pocketvla-3m-sft-2d-grab</td><td>SFT</td><td>34.7%</td><td>66.7%</td><td>0.485</td><td>31.1</td></tr>
  <tr><td>pocketvla-3m-grpo-2d-grab</td><td>SFT + GRPO</td><td>36.0%</td><td>67.0%</td><td>0.472</td><td>30.9</td></tr>
  <tr><td>pocketvla-3m (no aux heads)</td><td>SFT</td><td>4.0%</td><td>17.0%</td><td>0.888</td><td>44.8</td></tr>
</table>


## Further Exploration

Once you have gotten started with this project, keep exploring to gain an even deeper understanding of VLA algorithms!
> 💡 1. **Improve OOD generalization**: data augmentation (random backgrounds/lighting/ball sizes), domain randomization, involving distractors in training, contrastive learning between color words and visual features, larger parameter scales... any improvement can be validated with one command on the fixed 300-scenario OOD set (`python3 -m pvla.eval_ood`).

> 💡 2. **Extend downstream tasks**: use the `pocketvla-3m` architecture as a base for structural innovation, parameter scaling, and training-paradigm exploration; define new VLA-style tasks and evaluation schemes in any simulation environment, and train your own `pocketvla-[x]m-[task]`.

## Citation

If this project helps your study or research, you are welcome to cite it:

```bibtex
@misc{pocketvla2026,
  title  = {PocketVLA: Building a 3M VLA from Absolute Zero},
  author = {Enming Zhang},
  year   = {2026},
  url    = {https://github.com/penpenzh/pocketvla}
}
```

## License

This project is open-sourced under the **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** (see [LICENSE](../LICENSE))
