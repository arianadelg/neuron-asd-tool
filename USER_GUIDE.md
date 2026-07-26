# Neuron ASD — User Guide

This guide walks you through Neuron ASD from start to finish. No programming experience is needed. If you can upload a file and click a button, you can use the tool.

---

## 1. What Neuron ASD does

Neuron ASD looks at an autistic subject's resting-state EEG and answers two questions:

1. **Where does this subject sit on the excitation/inhibition (E/I) axis**, compared with a typically-developing (TD) reference?
2. **Which receptor modulation is predicted to move this subject's brain-activity profile toward the TD reference**, and by how much?

It does this per subject, because autistic subjects differ from one another; the recommended modulation is often different for different people, and that is exactly what the tool is meant to reveal.

**Important:** Neuron ASD is a research and hypothesis-generation tool. It produces predictions from a computational model, relative to a reference you supply. These are **not** medical advice or validated clinical prescriptions.

---

## 2. What you need before you start

### A typically-developing (TD) reference
A set of resting-state, eyes-closed EEG recordings from typically-developing individuals, one file per person, packaged in a single `.zip` file.

- **How many recordings?** About **20** for a reliable reference. The tool still runs with fewer, but it will warn you: with around 3 recordings, up to one-third of subjects can be placed on the wrong side of the E/I axis.
- **Which group?** Ideally the **same cohort** (same site, same equipment, same population) as the subjects you will analyze. TD groups from other datasets are **not** interchangeable, even when the recording hardware matches.
- **Can I use a simulated reference instead of real data?** No. A model-generated reference is not valid on the aperiodic axis and will produce meaningless, saturated results.

### Subject recordings
Resting-state, eyes-closed EEG for each subject you want to analyze, one file per subject.

### Supported file formats
EEGLAB `.set` (with its `.fdt`), EDF `.edf`, BioSemi `.bdf`, BrainVision `.vhdr` (with `.eeg`/`.vmrk`), FIF `.fif`, and several others supported by MNE-Python. When a format uses sidecar files (like `.set`+`.fdt`), include **all** of them in the `.zip`.

---

## 3. Using the Colab notebook (recommended)

Colab runs everything in your browser; nothing is installed on your computer.

### Step 1 — Set up
Open `Neuron_ASD.ipynb` in Colab and run the first cell. It installs the required libraries, loads Neuron ASD, and prepares its response model.

**Expect a few minutes on the first run.** Neuron ASD has to simulate the effect of every receptor modulation once before it can make recommendations. This result is deterministic, so it is saved: if you return to the notebook later, or analyze more subjects, it starts immediately. Technical messages are hidden, so the cell will look quiet while it works. Wait until it prints **"Neuron ASD is ready."**

If you use Neuron ASD from your own Python code, you can trigger this step explicitly with `app.prepare()` — for example while you are still collecting your files — so the analyses themselves return without delay.

### Step 2 — Provide the TD reference
Run the Step 2 cell. A file picker appears; choose your TD `.zip`. Neuron ASD builds the reference and prints a summary, including a **reliability note** based on how many recordings you provided. Read that note — if it says the reference is small, treat individual classifications with caution.

### Step 3 — Analyze one subject
Run the Step 3 cell and choose one subject's EEG file. Neuron ASD prints:

- the subject's **aperiodic exponent** and the quality of the fit,
- the **E/I placement** (higher E/I, lower E/I, or typical),
- the **recommended modulation** with its confidence,
- the **predicted effect** (distance to TD before and after, and the gain in dB and %),

and draws a two-panel figure: the predicted movement toward TD, and the subject's deviation from TD in each frequency band.

### Step 4 — Analyze a whole group (optional)
Run the Step 4 cell and upload a `.zip` of several subjects. Neuron ASD returns a table with one row per subject and offers it as a downloadable spreadsheet (`neuron_asd_results.csv`).

### Step 5 — Built-in example
No data yet? Run the Step 5 cell. It downloads a small synthetic example bundled with the repository and runs the full analysis so you can see the expected output.

---

## 4. Using Neuron ASD in your own Python code

```python
from neuron_asd import app

# 1. Build a TD reference from a folder, a .zip, or a list of files
reference = app.build_reference("td_folder_or_zip")
app.reference_summary(reference)

# 2. Analyze one subject
result = app.analyze_subject("subject.set", reference)
app.show(result)                       # prints a summary and draws the figure

# access individual fields
print(result.ei_class, result.top_target, result.top_direction, result.gain_pct)

# 3. Analyze a whole folder/zip at once -> pandas DataFrame
table = app.analyze_folder("subjects_folder_or_zip", reference)
table.to_csv("results.csv", index=False)
```

### Choosing the region of interest (advanced)
By default, Neuron ASD derives a common central region of interest automatically (the vertex-cluster channels present in at least 85% of your files), exactly as in the paper. To force a specific set of channels, pass `roi=[...]` to `build_reference`. The same ROI is then used for every subject compared against that reference.

---

## 5. Understanding the outputs

| Field | What it means |
|---|---|
| **Aperiodic exponent** | The slope of the 1/f part of the EEG spectrum. A steeper (larger) exponent indexes relatively stronger inhibition; a flatter (smaller) exponent, relatively stronger excitation. |
| **Fit R²** | How well the spectral model fit the recording. Values close to 1 indicate a clean fit. |
| **E/I placement** | "Higher E/I (flatter)", "lower E/I (steeper)", or "typical", relative to the TD reference and its noise floor. |
| **Recommended move** | The receptor target and direction (Agonist / Inhibitor) predicted to move the subject toward TD. |
| **Confidence** | How consistently that move was selected across the internal stability ensemble (0–1). |
| **Distance before / after** | The subject's distance from the TD reference (over the five frequency bands, in dB) before and after the recommended modulation. |
| **Predicted gain** | How much the modulation is predicted to reduce that distance, in dB and as a percentage. |
| **Deviation by band** | How far the subject is from TD in each band (δ, θ, α, β, γ); this shows which bands drive the recommendation. |

A note on interpretation: the aperiodic read-out is a **per-subject placement**, not a group diagnosis. Neuron ASD deliberately makes no group-level claim about the direction of the E/I ratio in autism. The distance-to-TD and the band-level attributions describe the behaviour of the model relative to your reference; they are not independently validated clinical biomarkers.

---

## 6. Troubleshooting

**"No EEG files found."**
Your `.zip` may contain a folder inside a folder, or only sidecar files. Make sure the primary files (e.g. `.set`, `.edf`, `.vhdr`) are present, together with their sidecars.

**"Fewer than 3 reference-ROI channels available."**
The recording does not contain enough of the central channels used for the analysis. This happens when a montage is very different from the others. Check that channel names are standard 10–20 labels; Neuron ASD renames BioSemi A/B labels automatically, but highly non-standard names may not be recognized.

**A subject is skipped in the group table.**
The reason is shown in the `E_I_class` column for that row (for example, a read error or too few ROI channels). Other subjects are unaffected.

**The reference exponent looks wrong (e.g. negative, or very different from ~1.3).**
Real resting EEG typically gives an aperiodic exponent around 1.3. A value far from that usually means the input is not real resting EEG (for example, a simulated file), or the recordings are very short or heavily filtered. Do not use a simulated reference.

**The results change when I change the reference.**
That is expected and important: every output is defined relative to the TD reference. Use a real, same-cohort reference of about 20 recordings for stable results, and do not swap in a TD group from another dataset.

**The first analysis seems frozen.**
The response model is being simulated; this happens once and takes a few minutes. Technical output is hidden, so the cell looks idle while it works. Later runs load the saved model and start immediately. If you want to see it happen explicitly, run `app.prepare()` first.

**Everything ran but the figure did not appear.**
In some environments figures render only when the cell finishes. Re-run the cell, or call `app.show(result)` again.

---

## 7. Optional realistic mode (research extension)

By default, Neuron ASD's simulator is driven by white noise and has a flat aperiodic spectrum. This is the behaviour used for every result in the paper. A known consequence is that the machine-learning surrogate, trained on that simulator, does not cover the 1/f regime of real EEG.

The optional **realistic mode** addresses this. It applies a tissue filter — a 1/f^β transfer stage representing the dendritic filtering and volume conduction that shape the aperiodic slope of scalp EEG — to the model output, so the simulated EEG carries a realistic slope (β ≈ 1.7 gives a typically-developing exponent near 1.33) while preserving the E/I band structure. A surrogate retrained on this realistic engine (included as `models/surrogate_realistic_5k.keras`, R² ≈ 0.93) then operates in the real-EEG regime.

```python
from neuron_asd import app

app.enable_realistic_mode()      # tissue filter on (β = 1.7 by default)
# ... build_reference / analyze_subject / analyze_folder as usual ...
app.disable_realistic_mode()     # back to the default, paper-validated behaviour
```

Two things to keep in mind: enabling realistic mode changes what the simulator produces, so the response model is rebuilt the first time you analyze after switching (a one-time wait). And with the filter **off** — the default — the simulator is bit-for-bit identical to the published engine, so nothing in the paper's results is affected. Realistic mode is a research extension and a proof of principle; the recommendations themselves are not yet validated on empirical outcomes.

## 8. Reproducibility and limits

- Neuron ASD applies a fixed, published pipeline; the modelling engine is not meant to be modified if you want to reproduce the paper.
- The neural-mass model is deliberately simple and does not capture detailed channel dynamics, plasticity, or cortical spatial structure.
- The machine-learning surrogate included with the project is a fast **emulator of the simulator**; real EEG falls outside its training distribution, so it is not applied to real recordings in this interface.
- Cross-dataset comparisons remain sensitive to site and montage differences despite the harmonization applied here.

For the full methods, reliability analyses, and the receptor→EEG map with references, see the accompanying paper and its supplementary material.
