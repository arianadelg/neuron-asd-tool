# Neuron ASD — Interactive Explorer · User Guide

An interactive Google Colab tool for the validated **Neuron ASD** methodology. For each autistic
(ASD) subject you provide, it shows the **predicted effect** of receptor and neuromodulator
modulations that move the subject's resting-EEG spectral profile toward a **typically-developing
(TD)** reference.

> **Model-based predictions for hypothesis generation — not clinical prescriptions.**
> The aperiodic (1/f) read-out places each subject on the excitation/inhibition (E/I) axis
> individually; it is not a group-level claim.

**Repository:** <https://github.com/arianadelg/neuron-asd-tool> · **DOI:** `10.5281/zenodo.21581231`

---

## Quick start

1. Open `Neuron_ASD_Colab_Explorer.ipynb` in Google Colab.
2. Run the cells **in order** (each control is a native Colab form — sliders and dropdowns appear
   to the right of the cell).
3. After Steps 1–3, use **Step 4** to explore modulations as many times as you like — instantly.

The workflow:

| Step | Purpose |
|---|---|
| **1. Setup** | Install the validated engine (+ MNE for EEG reading). |
| **2. Load EEG** | Upload ASD subjects (1, 3, or any number) and a TD pool. |
| **3. Engine + TD reference** | Choose the engine and build the (cached) TD reference. |
| **3b. Accelerator** *(optional)* | Train the CNN surrogate — only for large batches. |
| **4. Modulate manually** | Set knobs → predicted effect + clinical-style note. Repeatable. |
| **5. Recommendation** *(optional)* | The model's own best per-subject modulations. |

---

## Step 1 · Setup

Run once. Installs the Neuron ASD package from the repository plus dependencies, including **MNE**
(the library that reads EEG file formats).

**About TensorFlow:** this cell does **not** install TensorFlow. Colab ships a working build, and
installing another on top corrupts the native libraries. The cell only checks whether TensorFlow is
importable, because it is needed **only** by the optional accelerator (Step 3b).

*No adjustable parameters.*

---

## Step 2 · Load EEG (ASD subjects + TD pool)

Upload your recordings: **1, 3, or any number** of ASD subjects, and a **pool of TD** recordings
that serves as the reference.

**Accepted formats** (any the main tool reads, via MNE): EEGLAB `.set`(+`.fdt`), EDF `.edf`,
BioSemi `.bdf`, `.gdf`, BrainVision `.vhdr`(+`.eeg`/`.vmrk`), MNE `.fif`, Neuroscan `.cnt`/`.cdt`,
EGI `.mff`/`.raw`, plus `.npy`, `.csv`, `.txt`.

**Multi-file formats:** if your format uses several files (e.g. `.set` + `.fdt`), **select all parts
together** in the upload dialog. They are saved to the same folder and MNE matches them automatically.

**Processing:** each recording is conditioned with the same validated pipeline as the paper
(resample to 200 Hz, filtering, notch, 120 s window, central region of interest).

### Parameters

| Parameter | Default | What it is and how it varies |
|---|---|---|
| `reload_ASD_subjects` | `True` | When ticked, prompts you to upload ASD subjects. Untick to keep the ASD subjects already loaded. |
| `reload_TD_pool` | `True` | The TD pool is the expensive part. Tick it the first time to upload it. **Afterwards, untick it** to keep the loaded TD pool while you change only the ASD subjects — this saves a lot of time. |
| `TD_reference_source` | *Upload / keep a real TD pool* | `Upload / keep a real TD pool` uses real TD recordings (recommended). `Use the TD simulator (LEAST reliable)` uses the simulator as reference — the **least reliable** option, because the simulated reference is invalid on the aperiodic (1/f) axis. Use it only if you have no real TD data. |

**TD pool size:** the paper found the E/I read-out stabilizes around **~20 real TD recordings**.
With fewer, the tool warns you and results should be treated as provisional.

---

## Step 3 · Engine + build TD reference

Run once after loading data. It does two things that make everything else fast: it builds the TD
reference (the target profile) and pre-computes each subject's features. Both are **cached**, so
Steps 4 and 5 are then instant and repeatable.

**When to re-run:** only if you change the engine or reload data. Otherwise the cell detects the
cached reference and reuses it without recomputing.

### Parameter

| Parameter | Default | What it is and how it varies |
|---|---|---|
| `engine_mode` | *Classic (paper-validated)* | `Classic` reproduces the paper exactly (validated results). `Realistic (tissue filter, extension)` adds the 1/f tissue filter — it brings the simulator into the real-EEG regime, useful mainly with the emulator. With Classic, the validated results are unchanged. |

> **Which engine for real EEG?** Real EEG carries its own 1/f (aperiodic) physics. In **Realistic**,
> the simulated TD reference and the modulation responses also carry that 1/f structure, so they live
> on the **same physical scale** as your real recordings — the more coherent choice when comparing
> against real EEG. In **Classic**, the simulated reference is aperiodically flat (the "invalid on the
> aperiodic axis" finding of the paper), so distances mix two scales. Use **Classic** to reproduce the
> paper's validated numbers; prefer **Realistic** when interpreting modulations against your own real
> EEG. The choice is yours — the tool prints a note reminding you which regime you're in.

On completion it prints the TD reference band profile (dB) and, for a real pool, its aperiodic
exponent, plus which subjects are ready.

---

## Step 3b · Accelerator (surrogate emulator) — optional

**What it is:** a convolutional neural network (CNN) that **emulates** the simulator, roughly 2×
faster per evaluation.

**When to use it:** only if you plan **many** evaluations (large batches, parameter sweeps, or the
paper's AI component). **For interactive use with a few subjects you do not need it** — the direct
simulator in Steps 4–5 is fast enough and more accurate. The cell is placed *before* the evaluations
on purpose: if you do train it, later steps can use it.

**Robustness:** training generates data in **resumable blocks**, so a Colab restart never loses more
than one block. Fidelity scales with the number of samples (in the paper, R² ≈ 0.93 at 5000 samples
on the realistic engine).

### Parameters

| Parameter | Default | Range | What it is and how it varies |
|---|---|---|---|
| `run_training` | `False` | checkbox | Must be ticked to train. Off by default, since interactive use does not need the accelerator. |
| `n_samples` | `1500` | 300–5000 | Number of training samples. More samples = higher emulator fidelity, but more time. Reference: R²≈0.75 (700), 0.89 (1500), 0.93 (5000). |
| `epochs` | `40` | 10–80 | Training epochs. More epochs refine the fit; too many can overfit. 40 is a balanced value. |

---

## Step 4 · Modulate manually → predicted effect

The interactive core. Choose a subject, set the modulations, and run. **Repeat as many times as you
like** — change any control and re-run; it's instant because the TD reference is cached.

**Coherence with Step 5:** modulations are translated exactly as in the paper (each receptor with
its own sign and magnitude), so the effect you see here matches the Step-5 recommendations.

### Subject and seed

| Parameter | Default | What it is and how it varies |
|---|---|---|
| `subject` | `ALL (cohort average)` | Type an ASD subject name (as in Step 2) to view it individually, or leave `ALL (cohort average)` for the cohort mean. |
| `seed` | `42` (1–200) | Random seed of the simulation. Changing it shows run-to-run variability. It does not change the modulation, only that run's stochastic noise. |

### Receptor modulations

Each of the 8 receptors has three options: **None** (no modulation), **Agonist** (+) or
**Inhibitor** (−). The default for all is **None**.

> **Important:** "Agonist" does not always mean a positive value in the model. Each receptor has its
> own pharmacological sign and magnitude. For example, a **5HT2A** agonist maps to a **negative**
> model value. The tool applies these signs automatically — the same way Step 5 does.

| Receptor | Agonist → model value | Notes |
|---|---|---|
| GABA_A | +0.30 | Direct sign |
| GABA_B | +0.30 | Direct sign |
| NMDA (NR2A) | +0.30 | Direct sign |
| NMDA (NR2B) | +0.20 | Smaller magnitude |
| AMPA | +0.30 | Direct sign |
| D2_dopamine | +0.20 | Smaller magnitude |
| 5HT2A | −0.30 | Inverted sign (agonist = negative value) |
| alpha7_nAChR | +0.30 | Direct sign |

An inhibitor produces the value with the opposite sign to the table.

### Global neuromodulators

| Parameter | Default | Range | What it is |
|---|---|---|---|
| `dopamine` | `0.0` | −1 … +1 | Global dopamine level. Continuous slider (step 0.05). |
| `serotonin` | `0.0` | −1 … +1 | Global serotonin level. |
| `norepinephrine` | `0.0` | −1 … +1 | Global norepinephrine level. |

### What the result shows

- **Spectral profile:** bars for the subject, the post-modulation profile, and the TD reference,
  across the 5 bands (δ, θ, α, β, γ).
- **Distance to TD:** before and after the modulation. A decrease means moving toward TD (an
  improvement); the title reports the change in dB and percent.
- **E/I placement:** the subject, the modulated profile, and the reference on the aperiodic-exponent
  axis (steeper = lower E/I).
- **Clinical-style note:** a short printed text (not a file) summarizing the subject's baseline, the
  applied modulation, the predicted effect, and an interpretation hint.

---

## Step 5 · Per-subject recommendation (optional)

For each subject, the tool suggests the modulations the model itself predicts as best for moving the
subject toward TD, using the validated **response-projection** method.

**On timing:** the first run builds a "response ensemble" (~3–4 minutes). It is then cached, so
subsequent runs — including changing the subject — are fast.

### Parameters

| Parameter | Default | Range | What it is and how it varies |
|---|---|---|---|
| `which_subject` | `ALL subjects` | text | `ALL subjects` to process all, or a specific subject name from Step 2. |
| `top_k` | `4` | 1–8 | Maximum number of modulations the algorithm considers when projecting the subject's deviation — how many "moves" the greedy projection chains to cancel the gap to TD. Raising it allows longer combinations; lowering it restricts to the most dominant. It does **not** force that many to be shown: only moves above the confidence threshold are reported. |

### How to read the recommendation (and confidence)

The method is **stability selection**: the projection is repeated over a set of stable reference
baselines (3) combined with several seeds (8), and a move is reported with a **confidence** equal to
the fraction of the ensemble in which it was selected.

> **Reliability filter:** the tool reports only moves with confidence **≥ 60%**. This matters:
> low-confidence moves are selected only *in combination* with others and, applied in isolation in
> Step 4, may produce a contrary effect. By reporting only the ≥ 60% moves, what Step 5 recommends is
> guaranteed to be reproducible when applied manually in Step 4.

A per-subject clinical-style note summarizes the strongest reliable hypothesis and its confidence.

---

## Internal method values (not adjustable in the interface)

Fixed in the validated engine; useful to know when interpreting results:

| Constant | Value | Meaning |
|---|---|---|
| Stability threshold | 0.60 (60%) | Minimum confidence to report a modulation in Step 5. |
| Noise gate | 0.40 dB | Ignores gains below the response noise floor; avoids reporting trivial effects. |
| Reference baselines | 3 | Stable terrains on which each modulation's response is measured. |
| Response seeds | 8 | Stochastic seeds per baseline; the full ensemble is 3 × 8 = 24. |
| Sampling rate | 200 Hz | The EEG is resampled to this rate during conditioning. |
| Bands | δ, θ, α, β, γ | The five bands defining the spectral profile. |

---

## Troubleshooting

- **Widgets / controls don't appear.** The Explorer uses native Colab forms, not ipywidgets, so
  controls always render. If a form looks blank, re-run the cell.
- **TensorFlow `undefined symbol` error (only if you use Step 3b).** Your Colab session's
  TensorFlow was corrupted by a previous install of a different TensorFlow on top of Colab's build.
  Fix it with **Runtime → Restart session** (or *Disconnect and delete runtime*), then run the cells
  again from Step 1. Do **not** pip-install TensorFlow. As of v1.1.2 the package no longer lists
  TensorFlow as a dependency, so a fresh install does not trigger this. You do not need the
  accelerator for interactive use — Steps 4–5 work without it.
- **A `.set` file fails to load.** Make sure you selected its `.fdt` sidecar in the same upload.
- **Step 5 is slow the first time.** It builds the response ensemble once (~3–4 min); later runs are
  cached and fast.

---

## Citation

If you use this tool, please cite the repository via its DOI:

> Neuron ASD — an open, interactive platform for per-subject exploration of receptor modulations
> toward a typically-developing EEG reference in autism. DOI: `10.5281/zenodo.21581231`.

All outputs are **model-based predictions for hypothesis generation in research and education**.
They are not clinical advice or prescriptions.
