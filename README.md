# Neuron ASD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**An open platform for exploring the predicted effects of receptor agonist/inhibitor modulations toward a typically-developing (TD) profile from EEG in autism.**

Neuron ASD takes an autistic subject's resting-state EEG, places the subject on the excitation/inhibition (E/I) axis relative to a TD reference, and reports the receptor modulation predicted to move that subject's spectral profile toward TD. It is built around a three-population neural-mass model and an aperiodic (1/f) read-out, with a friendly interface intended for non-programmers.

> **Scope.** Neuron ASD is a research and hypothesis-generation tool. Its outputs are model-based predictions relative to a reference, **not** validated clinical prescriptions.

---

## Quick start (no installation)

Open the notebook in Google Colab and run the cells in order:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arianadelg/neuron-asd-tool/blob/main/Neuron_ASD.ipynb)

1. **Set up** — one cell installs everything and hides technical messages.
2. **Provide a TD reference** — upload a `.zip` of real typically-developing recordings.
3. **Analyze a subject** — upload one EEG file and see the E/I placement, the recommended modulation, and the predicted effect toward TD.
4. **Analyze a group** (optional) — upload a `.zip` and download a results table.
5. **Built-in example** — run the bundled synthetic data if you just want to see it work.

## Installation (for local or programmatic use)

```bash
pip install git+https://github.com/arianadelg/neuron-asd-tool.git
```

```python
from neuron_asd import app

reference = app.build_reference("td_recordings.zip")     # real TD group
result    = app.analyze_subject("subject.set", reference) # one autistic subject
app.show(result)                                          # summary + figure

table = app.analyze_folder("subjects.zip", reference)     # a whole group -> DataFrame
```

## What you need

- **A TD reference**: real resting-state, eyes-closed EEG from a typically-developing group. The reference becomes reliable at about **20 recordings**; with fewer, the tool still runs but warns that some subjects may be misclassified. Draw the reference from the **same cohort** as your subjects when possible — references from other datasets are not interchangeable, and a simulated reference is not valid on the aperiodic axis.
- **Subject recordings**: resting-state, eyes-closed EEG, one file per subject.

Supported formats (via [MNE-Python](https://mne.tools)): EEGLAB `.set`, EDF `.edf`, BioSemi `.bdf`, BrainVision `.vhdr`, FIF `.fif`, and others.

## What it reports

| Output | Meaning |
|---|---|
| **E/I placement** | Where the subject sits on the excitation/inhibition axis (aperiodic 1/f exponent) relative to the TD reference. |
| **Recommended move** | The receptor agonist/inhibitor modulation predicted to move the subject's band profile toward TD, with a confidence. |
| **Predicted gain** | How far (in dB and %) the recommended modulation is predicted to reduce the distance to TD. |

## Optional realistic mode (research extension)

By default the simulator has a flat aperiodic spectrum (the paper-validated behaviour). An optional realistic mode applies a 1/f tissue filter so the simulated EEG carries a realistic aperiodic slope, and a surrogate retrained on it (`models/surrogate_realistic_5k.keras`, R² ≈ 0.93) operates in the real-EEG regime:

```python
app.enable_realistic_mode()      # tissue filter on
# ... analyze as usual ...
app.disable_realistic_mode()     # restore default, paper-validated behaviour
```

With the filter off (the default), the engine is bit-for-bit identical to the published version, so the paper's results are unaffected. See `USER_GUIDE.md` for details.

## The analysis pipeline

Neuron ASD applies the exact pipeline described in the paper: resampling to 200 Hz, 0.5–80 Hz band-pass and 50 Hz notch filtering, channel-name standardization (BioSemi A/B labels renamed to 10–20 names), a common central region of interest (channels present in ≥85% of the recordings), a 120 s common window, the recording's native reference, and spectral parameterization (FOOOF) for the aperiodic exponent.

## Repository contents

```
neuron-asd/
├── Neuron_ASD.ipynb        # friendly Colab notebook (start here)
├── neuron_asd/
│   ├── app.py              # user-facing interface (build_reference, analyze_subject, ...)
│   ├── engine.py           # modelling engine (do not modify to reproduce the paper)
│   └── __init__.py
├── models/                 # retrained surrogate for the optional realistic mode
├── examples/               # small synthetic example bundles (TD and subjects)
├── USER_GUIDE.md           # step-by-step guide with troubleshooting
├── PUBLISHING_GUIDE.md     # maintainer notes: GitHub + Zenodo release workflow
├── CITATION.cff            # machine-readable citation metadata
├── .zenodo.json            # metadata used when Zenodo archives a release
├── requirements.txt
├── setup.py
└── LICENSE
```

## Reproducing the paper

The published evaluation (reference stability, minimum-N, cross-dataset transferability, and the machine-learning surrogate) is in a separate evaluation notebook released with the paper. This repository provides the tool itself and the friendly interface for applying it to new data.

## Citation

If you use Neuron ASD, please cite **both** the software and the accompanying paper.

**Software** (archived on Zenodo, one DOI per release):

> Alvarado, Y. J., Cardozo-Urdaneta, A., Lossada, C., Quintero, M., Delgado, A., & González-Paz, L. (2026). *Neuron ASD* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

**Paper**: details will be added on publication.

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it under *"Cite this repository"* in the sidebar.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgements

Developed at the Instituto Venezolano de Investigaciones Científicas (IVIC), Centro de Biomedicina Molecular (CBM). Built on [MNE-Python](https://mne.tools) and [FOOOF/specparam](https://fooof-tools.github.io/fooof/).
