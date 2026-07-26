# -*- coding: utf-8 -*-
"""
Neuron ASD - friendly user interface layer.

This module wraps the modelling engine (engine.py) in a small, well-documented API
intended for non-programmers: reviewers reproducing the paper, and anyone using the
tool after publication. It hides library warnings and verbose logs, applies the exact
analysis pipeline reported in the paper (120 s common window, harmonized central ROI,
native reference), and returns tidy tables and figures.

Typical use (Colab):

    from neuron_asd import app
    ref = app.build_reference("td_folder_or_zip")        # typically-developing reference
    result = app.analyze_subject("subject.set", ref)     # one autistic subject
    app.show(result)                                     # printed summary + figure

or, for a whole folder of subjects:

    table = app.analyze_folder("asd_folder_or_zip", ref)

Nothing in this module changes the engine's numerical behaviour; it only orchestrates
the published pipeline and presents the outputs.
"""

from __future__ import annotations
import os, io, sys, glob, zipfile, tempfile, contextlib, warnings, logging
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------------
# 1) Quiet environment: silence the noisy third-party warnings/logs that would
#    otherwise flood a Colab cell. This does not affect results, only the console.
# ----------------------------------------------------------------------------------
def _quiet_environment() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # TensorFlow C++ logs off
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # hide oneDNN notice
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    warnings.filterwarnings("ignore")
    for noisy in ("mne", "tensorflow", "absl", "matplotlib", "fooof", "h5py"):
        try:
            logging.getLogger(noisy).setLevel(logging.ERROR)
        except Exception:
            pass
    try:
        import mne
        mne.set_log_level("ERROR")
    except Exception:
        pass
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass

_quiet_environment()


@contextlib.contextmanager
def _suppress_output():
    """Swallow stray stdout/stderr chatter emitted deep inside third-party libraries
    (e.g. MNE readers) so the user sees only Neuron ASD's own, tidy messages."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        devnull = io.StringIO()
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# ----------------------------------------------------------------------------------
# 2) Import the engine (quietly) and expose the constants we need.
# ----------------------------------------------------------------------------------
with _suppress_output():
    try:
        from . import engine as N          # when imported as a package
    except ImportError:                    # when the two files sit side by side
        import engine as N                 # type: ignore

import numpy as np

BANDS = list(N.BANDS.keys())
FS = N.FS

# Published pipeline defaults (Section 2.4 of the paper). Users normally never change these.
WINDOW_SECONDS = 120.0
ROI_PRESENCE_FRAC = 0.85
MIN_ROI_CHANNELS = 3
CENTRAL_ROI = ['Cz', 'C1', 'C2', 'C3', 'C4', 'FCz', 'CPz', 'Fz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2']

# BioSemi raw A/B electrode labels -> 10-20 names (standard biosemi64 order).
def _biosemi_rename_map():
    try:
        import mne
        names = mne.channels.make_standard_montage('biosemi64').ch_names
        raw = [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)]
        return dict(zip(raw, names))
    except Exception:
        return {}

_BIOSEMI_MAP = _biosemi_rename_map()

_READABLE_EXT = ('.set', '.edf', '.bdf', '.gdf', '.vhdr', '.fif', '.fif.gz',
                 '.cnt', '.cdt', '.mff', '.raw', '.nxe')


# ----------------------------------------------------------------------------------
# 3) File discovery: accept a folder, a .zip, or a list of files.
# ----------------------------------------------------------------------------------
def _gather_files(source) -> list:
    """Return a sorted list of EEG file paths from a folder, a .zip archive, or a list."""
    if isinstance(source, (list, tuple)):
        return sorted(str(s) for s in source)
    source = str(source)
    if source.lower().endswith('.zip'):
        out_dir = tempfile.mkdtemp(prefix="neuron_asd_")
        with zipfile.ZipFile(source) as z:
            z.extractall(out_dir)
        source = out_dir
    if os.path.isdir(source):
        files = []
        for ext in _READABLE_EXT:
            files += glob.glob(os.path.join(source, "**", f"*{ext}"), recursive=True)
        # BrainVision/EEGLAB sidecars must not be treated as separate recordings
        files = [f for f in files if not f.lower().endswith(('.fdt', '.eeg', '.vmrk'))]
        return sorted(set(files))
    if os.path.isfile(source):
        return [source]
    raise FileNotFoundError(f"Could not find EEG data at: {source}")


def _standardize_channels(raw):
    """Rename raw BioSemi A/B labels to 10-20 names (collision-safe) and normalize case."""
    import mne
    present = set(raw.ch_names)
    mapping = {}
    for old, new in _BIOSEMI_MAP.items():
        if old in present and new not in present and new not in mapping.values():
            mapping[old] = new
    if mapping:
        with _suppress_output():
            raw.rename_channels(mapping)
    # normalize case to the canonical 10-20 spelling for the ROI candidates
    canon = {c.lower(): c for c in CENTRAL_ROI}
    case_map = {ch: canon[ch.lower()] for ch in raw.ch_names
                if ch.lower() in canon and ch != canon[ch.lower()]}
    if case_map:
        with _suppress_output():
            raw.rename_channels(case_map)
    return raw


def _load_raw(file_path):
    import mne
    readers = {
        '.set': mne.io.read_raw_eeglab, '.edf': mne.io.read_raw_edf,
        '.bdf': mne.io.read_raw_bdf, '.gdf': mne.io.read_raw_gdf,
        '.vhdr': mne.io.read_raw_brainvision, '.fif': mne.io.read_raw_fif,
        '.cnt': mne.io.read_raw_cnt, '.cdt': mne.io.read_raw_curry,
        '.mff': mne.io.read_raw_egi, '.raw': mne.io.read_raw_egi,
        '.nxe': mne.io.read_raw_eximia,
    }
    low = file_path.lower()
    ext = '.fif' if low.endswith('.fif.gz') else os.path.splitext(low)[1]
    with _suppress_output():
        if ext in readers:
            raw = readers[ext](file_path, preload=True, verbose='ERROR')
        elif hasattr(mne.io, 'read_raw'):
            raw = mne.io.read_raw(file_path, preload=True, verbose='ERROR')
        else:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
    return _standardize_channels(raw)


def _roi_signal(raw, roi_channels):
    """Return the mean signal over the ROI channels the file actually has, conditioned
    exactly as in the paper (resample 200 Hz, 0.5-80 band-pass, 50 Hz notch, 120 s window)."""
    import mne
    have = [c for c in roi_channels if c in raw.ch_names]
    if len(have) < MIN_ROI_CHANNELS:
        return None, have
    with _suppress_output():
        raw.pick(have)
        if abs(raw.info['sfreq'] - FS) > 1e-6:
            raw.resample(FS, verbose='ERROR')
        raw.filter(0.5, 80.0, verbose='ERROR')
        try:
            raw.notch_filter(50.0, verbose='ERROR')
        except Exception:
            pass
    data = raw.get_data().mean(axis=0)              # average of retained ROI channels
    n = int(WINDOW_SECONDS * FS)
    if len(data) >= n:
        data = data[:n]
    return data, have


def _features_from_signal(signal):
    """Replicate the engine's validated real-EEG pipeline exactly (Section 2.4):
    unit-variance -> dB spectrogram -> level-align to model -> mean band powers,
    plus the FOOOF aperiodic exponent. Returns (band_vector, exponent, fit_R2)."""
    with _suppress_output():
        s = np.asarray(signal, float)
        s = (s - s.mean()) / (s.std() + 1e-12)          # real-EEG conditioning
        spec, f, t = N.spectrogram_db_of_signal(s)
        spec = N.align_level_to_model(spec)             # S1 level alignment
        bands_mean = N.band_power_mean(spec, f, t)      # {band: mean dB}
        band_vec = np.array([bands_mean[b] for b in BANDS], float)
        FOOOF = N._lazy_fooof()
        exp, sd, r2 = N._exp_of_signal(s, FOOOF)
    return band_vec, float(exp), float(r2)


def _common_roi(files):
    """Two-pass: scan headers to find channels present in >= ROI_PRESENCE_FRAC of files."""
    import mne
    counts = {c: 0 for c in CENTRAL_ROI}
    n_ok = 0
    for fp in files:
        try:
            raw = _load_raw(fp)
        except Exception:
            continue
        n_ok += 1
        present = set(raw.ch_names)
        for c in CENTRAL_ROI:
            if c in present:
                counts[c] += 1
    if n_ok == 0:
        return [], counts
    keep = [c for c in CENTRAL_ROI if counts[c] >= ROI_PRESENCE_FRAC * n_ok]
    return keep, counts


# ----------------------------------------------------------------------------------
# 4) Public data classes
# ----------------------------------------------------------------------------------
@dataclass
class Reference:
    """A typically-developing (TD) reference built from real recordings."""
    n: int
    exponent: float
    bands: np.ndarray
    roi: list
    per_file_exponents: list = field(default_factory=list)

    def reliability_note(self) -> str:
        if self.n >= 20:
            return "Reference size adequate (>=20 recordings): E/I classification is stable for ~9/10 subjects."
        if self.n >= 12:
            return "Reference usable but imprecise (12-19 recordings): expect ~80% classification stability."
        if self.n >= 3:
            return "Small reference (3-11 recordings): up to ~1/3 of subjects may be misclassified. Use with caution."
        return "Reference too small (<3 recordings): not recommended."


@dataclass
class SubjectResult:
    """Per-subject output: E/I placement and the recommended modulation toward TD."""
    name: str
    exponent: float
    fit_r2: float
    ei_class: str
    top_target: str
    top_direction: str
    confidence: float
    distance_before: float
    distance_after: float
    gain_db: float
    gain_pct: float
    band_deviation: dict
    all_recommendations: list


# ----------------------------------------------------------------------------------
# 5) Public API
# ----------------------------------------------------------------------------------
def _cache_signature() -> str:
    """Fingerprint of the engine settings the ensemble depends on, so a cached
    ensemble is never reused after the engine's configuration changes."""
    import hashlib
    parts = [
        repr(getattr(N, "RESP_SEEDS", None)),
        repr(getattr(N, "RESP_N_RUNS", None)),
        repr(getattr(N, "RESP_NOISE_GATE", None)),
        repr(len(getattr(N, "STABLE_REF_BASELINES", []) or [])),
        repr(list(getattr(N, "TARGET_KEYS", []))),
        repr(list(getattr(N, "NEUROMOD_KEYS", []))),
        repr(getattr(N.NeuralMass, "DRIVE_EXPONENT", None)),   # normal vs realistic mode
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _cache_path() -> str:
    d = os.path.join(os.path.expanduser("~"), ".neuron_asd_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"response_ensemble_{_cache_signature()}.pkl")


REALISTIC_BETA_DEFAULT = 1.7      # tissue-filter exponent giving a TD aperiodic slope ~1.33


def enable_realistic_mode(beta: float = REALISTIC_BETA_DEFAULT, verbose: bool = True) -> None:
    """Turn on the optional tissue filter (realistic mode).

    By default Neuron ASD's simulator is driven by white noise and has a flat aperiodic
    spectrum, which is why a surrogate trained on it does not cover the 1/f regime of real
    EEG. Realistic mode applies a 1/f^beta tissue/volume-conduction filter to the model
    output, so the simulated EEG carries a realistic aperiodic slope (beta ~1.7 gives a
    typically-developing exponent near 1.33) while preserving the E/I band structure.

    This is an optional research extension. It changes what the simulator produces, so any
    cached response model is rebuilt on next use. Call disable_realistic_mode() to return to
    the default, paper-validated behaviour, which is bit-for-bit identical to the published
    engine.
    """
    N.NeuralMass.DRIVE_EXPONENT = float(beta)
    N._RESP_ENSEMBLE = None            # force rebuild: the model's responses have changed
    if verbose:
        print(f"Realistic mode ON (tissue filter, beta={beta}). "
              "The response model will be rebuilt on next analysis.")


def disable_realistic_mode(verbose: bool = True) -> None:
    """Turn off the tissue filter and restore the default, paper-validated behaviour."""
    N.NeuralMass.DRIVE_EXPONENT = None
    N._RESP_ENSEMBLE = None
    if verbose:
        print("Realistic mode OFF. Simulator restored to the validated (default) behaviour.")


def prepare(verbose: bool = True, use_cache: bool = True) -> None:
    """Build (or load) the response ensemble the recommendation step needs.

    This is the slow part of Neuron ASD: it simulates the model's response to every
    receptor modulation across an ensemble of reference baselines and random seeds,
    and can take several minutes the first time. The result is deterministic, so it
    is cached on disk and reused instantly afterwards. Caching stores exactly the
    values that were computed; it does not change any result.

    Calling this is optional — the analysis functions call it automatically — but
    running it early (for example while you are still uploading data) means the
    analyses themselves return immediately.
    """
    if getattr(N, "_RESP_ENSEMBLE", None) is not None:
        return
    path = _cache_path()
    if use_cache and os.path.exists(path):
        try:
            import pickle
            with open(path, "rb") as fh:
                N._RESP_ENSEMBLE = pickle.load(fh)
            if verbose:
                print("Response model loaded from cache. Ready.")
            return
        except Exception:
            pass                                    # corrupt/incompatible cache -> rebuild
    if verbose:
        print("Preparing the response model (first run only).")
        print("This simulates the effect of every receptor modulation and takes a few")
        print("minutes. The result is saved, so later analyses start immediately...")
    with _suppress_output():
        N._build_response_ensemble()
    if use_cache:
        try:
            import pickle
            with open(path, "wb") as fh:
                pickle.dump(N._RESP_ENSEMBLE, fh)
        except Exception:
            pass                                    # cache is an optimization, never required
    if verbose:
        print("Response model ready.")


def build_reference(td_source, roi=None) -> Reference:
    """Build a TD reference from a folder, .zip, or list of real TD recordings.

    Parameters
    ----------
    td_source : str | list
        Folder path, .zip archive, or list of EEG files (typically-developing group).
    roi : list, optional
        Channels to average. If None, the common central ROI is derived automatically
        (channels present in >=85% of the files), exactly as in the paper.

    Returns
    -------
    Reference
    """
    files = _gather_files(td_source)
    if not files:
        raise FileNotFoundError("No EEG files found for the TD reference.")
    if roi is None:
        roi, _ = _common_roi(files)
        if len(roi) < MIN_ROI_CHANNELS:
            roi = [c for c in CENTRAL_ROI]      # fall back to full candidate ROI
    exps, band_list = [], []
    for fp in files:
        try:
            raw = _load_raw(fp)
            sig, have = _roi_signal(raw, roi)
            if sig is None:
                continue
            bands, exp, r2 = _features_from_signal(sig)
            exps.append(exp); band_list.append(bands)
        except Exception:
            continue
    if not exps:
        raise RuntimeError("None of the TD files could be processed. Check the ROI and formats.")
    return Reference(n=len(exps), exponent=float(np.mean(exps)),
                     bands=np.mean(np.vstack(band_list), axis=0), roi=list(roi),
                     per_file_exponents=[float(e) for e in exps])


def analyze_subject(subject_file, reference: Reference, subject_name=None) -> SubjectResult:
    """Analyze one autistic subject against a TD reference.

    Returns a SubjectResult with the E/I placement and the recommended agonist/inhibitor
    modulation predicted to move the subject's spectral profile toward TD.
    """
    prepare()                       # builds or loads the response ensemble (see prepare())
    name = subject_name or os.path.basename(str(subject_file))
    raw = _load_raw(str(subject_file))
    sig, have = _roi_signal(raw, reference.roi)
    if sig is None:
        raise RuntimeError(f"{name}: fewer than {MIN_ROI_CHANNELS} reference-ROI channels available.")
    bands, exp, r2 = _features_from_signal(sig)

    # E/I placement on the aperiodic axis
    d = exp - reference.exponent
    noise = getattr(N, "EI_NOISE", 0.05)
    ei = ("higher E/I (flatter)" if d < -noise else
          "lower E/I (steeper)" if d > noise else "typical")

    # recommendation + predicted effect toward TD
    with _suppress_output():
        recs, _ = N.recommend_by_projection(bands, reference.bands)
        d0, _, aft, _ = _projected_effect(bands, reference.bands)
    top = recs[0] if recs else ("(none)", "", 0.0)
    after = float(np.median(aft)) if len(aft) else d0
    gain = d0 - after
    dev = {b: float(reference.bands[i] - bands[i]) for i, b in enumerate(BANDS)}

    return SubjectResult(
        name=name, exponent=exp, fit_r2=r2, ei_class=ei,
        top_target=top[0], top_direction=top[1], confidence=float(top[2]),
        distance_before=float(d0), distance_after=after,
        gain_db=float(gain), gain_pct=float(100 * gain / d0) if d0 > 0 else 0.0,
        band_deviation=dev,
        all_recommendations=[(t, dr, float(fr)) for (t, dr, fr) in recs[:4]])


def _projected_effect(subject_bands, td_bands, top_k=4, gate=None):
    """Distance to TD before and after the recommended set of moves, across the
    stability ensemble. Mirrors the engine's greedy projection exactly."""
    if getattr(N, "_RESP_ENSEMBLE", None) is None:
        N._build_response_ensemble()
    gate = N.RESP_NOISE_GATE if gate is None else gate
    delta = np.asarray(td_bands, float) - np.asarray(subject_bands, float)
    d0 = float(np.linalg.norm(delta))
    after = []
    for R in N._RESP_ENSEMBLE:
        residual = delta.copy(); used = set()
        for _ in range(top_k):
            best, best_norm = None, np.linalg.norm(residual) - gate
            for (name, dirn), r in R.items():
                if name in used:
                    continue
                nn = np.linalg.norm(residual - r)
                if nn < best_norm:
                    best_norm, best = nn, (name, dirn, r)
            if best is None:
                break
            used.add(best[0]); residual = residual - best[2]
        after.append(float(np.linalg.norm(residual)))
    return d0, None, np.array(after), None


def analyze_folder(asd_source, reference: Reference):
    """Analyze every recording in a folder/zip/list against the TD reference.

    Returns a pandas DataFrame (one row per subject) with placements and predicted effects.
    """
    import pandas as pd
    prepare()                       # one-time; keeps the per-subject loop fast
    files = _gather_files(asd_source)
    rows = []
    for fp in files:
        try:
            r = analyze_subject(fp, reference)
            rows.append(dict(
                subject=r.name, exponent=round(r.exponent, 3), fit_R2=round(r.fit_r2, 3),
                E_I_class=r.ei_class, top_target=r.top_target, direction=r.top_direction,
                confidence=round(r.confidence, 2),
                distance_before_dB=round(r.distance_before, 2),
                distance_after_dB=round(r.distance_after, 2),
                predicted_gain_dB=round(r.gain_db, 2), predicted_gain_pct=round(r.gain_pct, 1)))
        except Exception as e:
            rows.append(dict(subject=os.path.basename(fp), E_I_class=f"skipped ({e})"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# 6) Presentation helpers
# ----------------------------------------------------------------------------------
def show(result: SubjectResult) -> None:
    """Print a clean, human-readable summary of one subject and draw the effect figure."""
    print(f"Subject: {result.name}")
    print(f"  Aperiodic exponent : {result.exponent:.3f}  (fit R^2 = {result.fit_r2:.3f})")
    print(f"  E/I placement      : {result.ei_class}")
    if result.top_target == "(none)":
        print("  Recommendation     : no modulation exceeded the response-noise floor.")
    else:
        print(f"  Recommended move   : {result.top_target}  {result.top_direction}  "
              f"(confidence {result.confidence:.2f})")
        print(f"  Distance to TD     : {result.distance_before:.2f} dB  ->  "
              f"{result.distance_after:.2f} dB")
        print(f"  Predicted gain     : {result.gain_db:.2f} dB "
              f"({result.gain_pct:.0f}% closer to TD)")
    _plot_subject(result)


def _plot_subject(result: SubjectResult):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2),
                                 gridspec_kw={'width_ratios': [1, 1.3]})
    a1.barh([0], [result.distance_before], color="#B03A2E", height=.5, label="before")
    a1.barh([0], [result.distance_after], color="#1E7B47", height=.5, label="after (predicted)")
    a1.set_yticks([]); a1.set_xlabel("distance to TD (dB)")
    a1.set_title("Predicted effect toward TD"); a1.legend(fontsize=8, loc="lower right")
    bands = list(result.band_deviation.keys())
    vals = [result.band_deviation[b] for b in bands]
    cols = ["#1F4E79" if v >= 0 else "#C77F0A" for v in vals]
    a2.bar(bands, vals, color=cols)
    a2.axhline(0, color="#888", lw=.8)
    a2.set_ylabel("TD - subject (dB)"); a2.set_title("Deviation from TD by band")
    plt.tight_layout(); plt.show()


def reference_summary(reference: Reference) -> None:
    print(f"TD reference built from {reference.n} recording(s).")
    print(f"  Mean aperiodic exponent : {reference.exponent:.3f}")
    print(f"  ROI channels            : {', '.join(reference.roi)}")
    print(f"  Reliability             : {reference.reliability_note()}")


__all__ = ["Reference", "SubjectResult", "prepare", "enable_realistic_mode",
           "disable_realistic_mode", "build_reference", "analyze_subject",
           "analyze_folder", "show", "reference_summary", "BANDS", "FS"]
