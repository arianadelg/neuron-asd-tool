# -*- coding: utf-8 -*-
"""
Neuron ASD - in silico pharmacological EEG screening engine.

A rate-based neural-mass model (three populations: excitatory E, fast inhibitory I1,
slow inhibitory I2) with a literature-grounded map from receptor / neuromodulator
targets to model parameters. Given EEG recordings, it places each subject on the
excitation / inhibition axis and generates per-subject hypotheses about which
agonist / inhibitor changes would move an ASD spectrum toward a neurotypical (TD)
reference.

Components:
  - Neural-mass simulator and a drug -> parameter map (with pathway and EEG-signature
    rationale, exposed by the mapping helper).
  - Per-subject analysis by response projection (optimize) and by the aperiodic 1/f
    exponent (excitation / inhibition discrimination).
  - An optional CNN surrogate that emulates the simulator for fast screening.

EEG input: many formats (EEGLAB .set/.fdt, EDF/BDF, GDF, BrainVision .vhdr, FIF,
Neuroscan .cnt, Curry .cdt, EGI .mff/.raw, Eximia .nxe ...) or a single .zip holding
two or more recordings. Real recordings are standardized to unit variance and shifted
in dB to the model reference level so the band SHAPE (which carries the E/I
information) is preserved exactly; synthetic paths are untouched. The three-parameter
model fits real EEG only approximately, so per-subject fits and recommendations are
provisional pending fit-quality review.

Scope: research prototype for hypothesis generation. The surrogate is an emulator of
the simulator, not a clinical predictor. Mapping DIRECTIONS are literature-grounded;
numeric magnitudes are nominal (sensitivity analysis). All numeric constants and the
search engine are unchanged from the validated build.
"""

# %% ------------------------------------------------------------------
# 0. Imports and soft dependency handling (does NOT crash if pip is blocked)
# --------------------------------------------------------------------
import sys, subprocess, os, argparse, warnings, tempfile, zipfile, shutil, atexit, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("neuron_asd")

# Temp dirs used to unpack .zip archives of EEGs; cleaned up automatically on exit.
_TMP_DIRS = []
def _new_tmpdir():
    d = tempfile.mkdtemp(prefix="neuron_asd_eeg_")
    _TMP_DIRS.append(d)
    return d
@atexit.register
def _cleanup_tmpdirs():
    for d in _TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)

# Primary/header EEG extensions we can load. Companion/sidecar files (.fdt for
# EEGLAB, .eeg/.vmrk for BrainVision, .dat for Curry) must sit next to their header
# and are picked up automatically by MNE -- they are NOT loaded as separate files.
EEG_EXTS = ('.set', '.edf', '.bdf', '.gdf', '.vhdr', '.fif', '.fif.gz',
            '.cnt', '.cdt', '.dap', '.mff', '.raw', '.nxe', '.mefd')

def ensure(pkg, imp=None):
    """Import `imp`; if missing, try to pip-install `pkg`. Never hard-crash."""
    try:
        __import__(imp or pkg)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
            __import__(imp or pkg)
            return True
        except Exception as e:
            log.warning("Could not auto-install '%s' (%s). Please run: pip install %s",
                        pkg, e, pkg)
            return False

for _pkg, _imp in [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                   ("pandas", "pandas"), ("scikit-learn", "sklearn")]:
    ensure(_pkg, _imp)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless backend: works without a display
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split


# %% ------------------------------------------------------------------
# 1. Global parameters (identical to the Colab build)
# --------------------------------------------------------------------
FS = 200
DURATION = 10.0
NPERSEG = 256
NOVERLAP = 128
N_RUNS = 5
BANDS = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 80)}

ASD_PARAMS = {'g_nmda_ei': 0.6, 'g_nmda_ee': 0.8, 'low_freq_amp': 0.15}
TD_PARAMS  = {'g_nmda_ei': 1.0, 'g_nmda_ee': 1.0, 'low_freq_amp': 0.0}

# =====================================================================
# LITERATURE SUPPORT
# ---------------------------------------------------------------------
# References below justify the DIRECTION of each mapping (which pathway a drug
# acts on, and the sign of its EEG effect). They do NOT justify the numeric
# MAGNITUDES (the fractional steps): those are nominal modeling choices handled
# by sensitivity analysis, not values taken from the literature. Bibliographic
# details should be re-verified against primary sources before use in a manuscript.
# ---------------------------------------------------------------------
REFERENCES = {
    # Framework: E/I balance in ASD (note: the direction of the imbalance is debated)
    "RubensteinMerzenich2003": "Rubenstein & Merzenich (2003). Model of autism: increased "
        "ratio of excitation/inhibition in key neural systems. Genes Brain Behav 2(5):255-267.",
    "SohalRubenstein2019": "Sohal & Rubenstein (2019). Excitation-inhibition balance as a "
        "framework for investigating mechanisms in neuropsychiatric disorders. Mol Psychiatry 24:1248-1257.",
    # Neural mass / rate models
    "JansenRit1995": "Jansen & Rit (1995). EEG and visual evoked potential generation in a "
        "mathematical model of coupled cortical columns. Biol Cybern 73:357-366.",
    "WilsonCowan1972": "Wilson & Cowan (1972). Excitatory and inhibitory interactions in "
        "localized populations of model neurons. Biophys J 12:1-24.",
    "DavidFriston2003": "David & Friston (2003). A neural mass model for MEG/EEG: coupling and "
        "neuronal dynamics. NeuroImage 20:1743-1755.",
    # GABA_A -> beta (benzodiazepine / GABA_A PAM signature)
    "Berro2021": "Berro et al. (2021). Alprazolam-induced EEG spectral power changes in rhesus "
        "monkeys... Psychopharmacology 238:1373-1386. (benzodiazepines increase beta, decrease alpha)",
    "vanLier2004": "van Lier et al. (2004). Effects of diazepam and zolpidem on EEG beta "
        "frequencies... Neuropharmacology. (GABA_A modulators increase beta)",
    # NMDA on interneurons -> gamma (antagonist increases gamma via disinhibition)
    "NMDAgammaNetwork": "Network model of NMDA modulation of gamma oscillations in cortex "
        "(J Neurosci): antagonists increase gamma when acting on interneuron NMDARs. [verify authors]",
    "KetamineGammaModel2025": "Computational modeling of ketamine-induced gamma changes "
        "(PLOS Comput Biol, 2025): reduced NMDA on PV/SST interneurons reproduces increased gamma "
        "(interneuron NMDA enriched in GluN2D/GluN2B).",
    # 5-HT2A -> reduced low-frequency (alpha) power / desynchronization
    "Muthukumaraswamy2013": "Muthukumaraswamy et al. (2013). Broadband cortical desynchronization "
        "underlies the human psychedelic state. J Neurosci 33(38):15171-15183. (5-HT2A excitation of "
        "deep pyramidal cells -> reduced low-freq power; used a neural mass model + DCM)",
    "Kometer2013": "Kometer et al. (2013). Activation of 5-HT2A receptors underlies psilocybin-"
        "induced effects on alpha oscillations... J Neurosci 33:10544-10551.",
    # alpha7 nAChR on interneurons / oscillations / sensory gating
    "Alpha7Hippo2021": "Trends in Neurosciences (2021). alpha7 nAChRs in the hippocampal circuit. "
        "(alpha7 depolarizes interneurons; regulates glutamatergic & GABAergic transmission)",
    "Stoiljkovic2016": "Stoiljkovic et al. (2016). Selective activation of alpha7 nAChRs augments "
        "hippocampal oscillations. Neuropharmacology 110:102-108.",
    "Freedman1997": "Freedman et al. (1997). Linkage of a neurophysiological deficit (P50 sensory "
        "gating) in schizophrenia to a chromosome 15 locus (CHRNA7). PNAS 94:587-592.",
    # GABA_B slow inhibition (classical neurophysiology)
    "GABAB_slow": "GABA_B receptors mediate slow inhibitory postsynaptic potentials "
        "(classical neurophysiology). [add a primary review]",
}

# ---------------------------------------------------------------------
# M2: neuromodulator couplings as explicit NOMINAL fractional scale factors.
# The DIRECTIONS below are qualitatively motivated; the numeric coefficients are
# illustrative and must be examined with a sensitivity analysis, not cited.
# ---------------------------------------------------------------------
NEUROMOD_COEFFS = {
    'dopamine':       {'p_base': +0.15, 'g_nmda_ei': -0.10},
    'serotonin':      {'g_nmda_ee': +0.20, 'g_nmda_ei': +0.10},
    'norepinephrine': {'g_nmda_ee': +0.15, 'tau_e': -0.20},
}

# ---------------------------------------------------------------------
# A2: protein targets -> distinct, physiologically-motivated knobs.
# Each entry documents the PATHWAY it acts on and the EEG signature reported in
# the literature (with reference keys). The fractional MAGNITUDES are nominal.
#
# 'effect_kind' flags how the EEG signature arises in THIS model:
#   'emergent'        = the signature is produced by the network dynamics (a genuine,
#                       testable check of construct validity);
#   'by_construction' = the knob is phenomenological and encodes the signature directly
#                       (matching it is NOT independent evidence).
#
# Residual, model-inherent degeneracy: AMPA (w_ee) and NMDA-NR2A (g_nmda_ee) both
# scale the single E->E excitatory term, so they form one "excitatory E->E class"
# and are indistinguishable without explicit NMDA kinetics (documented, not hidden).
# ---------------------------------------------------------------------
PROTEINS = [
    {"name": "GABA_A", "knob": "w_i1e", "base_frac": 0.30, "agonist_sign": +1,
     "pathway": "fast phasic GABA_A inhibition (I1->E)",
     "eeg_signature": "agonist/PAM increases beta (and gamma), decreases alpha",
     "effect_kind": "emergent", "refs": ["Berro2021", "vanLier2004"]},
    {"name": "GABA_B", "knob": "w_i2e", "base_frac": 0.30, "agonist_sign": +1,
     "pathway": "slow GABA_B inhibition (I2->E)",
     "eeg_signature": "slow inhibitory tone; shifts power toward lower frequencies",
     "effect_kind": "emergent", "refs": ["GABAB_slow"]},
    {"name": "NMDA (NR2A)", "knob": "g_nmda_ee", "base_frac": 0.30, "agonist_sign": +1,
     "pathway": "synaptic NMDA on pyramidal cells (E->E) [excitatory E->E class]",
     "eeg_signature": "modulates recurrent excitation / E-I balance",
     "effect_kind": "emergent", "refs": ["RubensteinMerzenich2003", "SohalRubenstein2019"]},
    {"name": "NMDA (NR2B)", "knob": "g_nmda_ei", "base_frac": 0.20, "agonist_sign": +1,
     "pathway": "NMDA onto fast interneurons (E->I1); antagonism disinhibits pyramidal cells",
     "eeg_signature": "antagonist increases gamma via interneuron disinhibition",
     "effect_kind": "emergent", "refs": ["NMDAgammaNetwork", "KetamineGammaModel2025"]},
    {"name": "AMPA", "knob": "w_ee", "base_frac": 0.30, "agonist_sign": +1,
     "pathway": "fast AMPA excitation (E->E) [excitatory E->E class]",
     "eeg_signature": "antagonist increases low-freq, decreases high-gamma",
     "effect_kind": "emergent", "refs": ["RubensteinMerzenich2003"]},
    {"name": "D2_dopamine", "knob": "p_base", "base_frac": 0.20, "agonist_sign": +1,
     "pathway": "net excitatory drive/excitability (LEAST-constrained knob)",
     "eeg_signature": "exploratory; dopaminergic modulation of cortical dynamics",
     "effect_kind": "emergent", "refs": []},
    {"name": "5HT2A", "knob": "low_freq_amp", "base_frac": 0.30, "agonist_sign": -1,
     "pathway": "phenomenological: reduces the low-frequency drive (models 5-HT2A "
                "excitation of deep pyramidal cells -> desynchronization)",
     "eeg_signature": "agonist reduces alpha/low-frequency power",
     "effect_kind": "by_construction", "refs": ["Muthukumaraswamy2013", "Kometer2013"]},
    {"name": "alpha7_nAChR", "knob": "w_ei2", "base_frac": 0.30, "agonist_sign": +1,
     "pathway": "nicotinic facilitation of drive to interneurons (E->I2); assigned to the "
                "slow-inhibitory route to stay distinct from the NR2B route (modeling choice)",
     "eeg_signature": "agonist augments oscillations/gamma; linked to P50 sensory gating",
     "effect_kind": "emergent", "refs": ["Alpha7Hippo2021", "Stoiljkovic2016", "Freedman1997"]},
]
TARGET_KEYS = [p["name"] for p in PROTEINS]
NEUROMOD_KEYS = ['dopamine', 'serotonin', 'norepinephrine']
MOD_ORDER = TARGET_KEYS + NEUROMOD_KEYS
N_MOD = len(MOD_ORDER)

W_EE0, W_EI1_0, W_EI2_0, W_I1E0, W_I2E0, P_BASE0 = 15.0, 12.0, 6.0, -10.0, -4.0, 3.0


# %% ------------------------------------------------------------------
# 2. Rate-based three-population neural mass model (Wilson-Cowan-type). M1
# --------------------------------------------------------------------
class NeuralMass:
    def __init__(self, g_nmda_ei=1.0, g_nmda_ee=1.0, low_freq_amp=0.0,
                 mod=None, dopamine=0.0, serotonin=0.0, norepinephrine=0.0,
                 w_ee=W_EE0, tau_e=0.005, fs=FS, sigma=0.4, f_slow=2.0):
        self.fs = fs
        self.dt = 1.0 / fs
        mod = mod or {}
        def m(name):
            return float(mod.get(name, 0.0))

        g_ei, g_ee, lfa, p0 = g_nmda_ei, g_nmda_ee, low_freq_amp, P_BASE0
        wee, wei1, wei2, wi1e, wi2e = w_ee, W_EI1_0, W_EI2_0, W_I1E0, W_I2E0

        # A2: one distinct knob per target
        wi1e *= (1.0 + m("GABA_A"))
        wi2e *= (1.0 + m("GABA_B"))
        g_ee *= (1.0 + m("NMDA (NR2A)"))            # }
        wee  *= (1.0 + m("AMPA"))                   # } shared E->E class (documented)
        g_ei *= (1.0 + m("NMDA (NR2B)"))            # A1: now active
        p0   *= (1.0 + m("D2_dopamine"))
        lfa   = max(0.0, lfa * (1.0 + m("5HT2A")))
        wei2 *= (1.0 + m("alpha7_nAChR"))

        # M2: global neuromodulatory tone
        for nm, val in (('dopamine', dopamine), ('serotonin', serotonin),
                        ('norepinephrine', norepinephrine)):
            if val != 0.0:
                for knob, c in NEUROMOD_COEFFS[nm].items():
                    if   knob == 'p_base':    p0   *= (1.0 + c * val)
                    elif knob == 'g_nmda_ei': g_ei *= (1.0 + c * val)
                    elif knob == 'g_nmda_ee': g_ee *= (1.0 + c * val)
                    elif knob == 'tau_e':     tau_e = tau_e * (1.0 + c * val)

        self.g_nmda_ei = g_ei
        self.g_nmda_ee = g_ee
        self.low_freq_amp = lfa
        self.p_base = p0
        self.w_ee = wee
        self.w_ei1 = wei1 * self.g_nmda_ei
        self.w_ei2 = wei2
        self.w_i1e = wi1e
        self.w_i2e = wi2e

        self.tau_e  = min(0.020, max(0.003, tau_e))   # A5: floor > dt/2
        self.tau_i1 = 0.008
        self.tau_i2 = 0.050
        self.sigma  = sigma
        self.f_slow = f_slow

    @staticmethod
    def _sig(x, thr=2.5):
        return 1.0 / (1.0 + np.exp(-(x - thr)))

    def drift(self, state, t):                        # A3: deterministic drift
        E, I1, I2 = state
        I_slow = self.low_freq_amp * np.sin(2 * np.pi * self.f_slow * t)
        p = self.p_base + I_slow
        I_E  = (self.w_ee * self.g_nmda_ee * self._sig(E)
                + self.w_i1e * self._sig(I1) + self.w_i2e * self._sig(I2) + p)
        I_I1 = self.w_ei1 * self._sig(E)
        I_I2 = self.w_ei2 * self._sig(E)
        return np.array([(-E + I_E) / self.tau_e,
                         (-I1 + I_I1) / self.tau_i1,
                         (-I2 + I_I2) / self.tau_i2])

    # OPTIONAL tissue filter (off by default). When DRIVE_EXPONENT is set to a value beta,
    # the model output is passed through a 1/f^beta tissue/volume-conduction filter so the
    # simulated EEG carries a realistic aperiodic slope. With DRIVE_EXPONENT = None the
    # simulator reproduces the validated engine EXACTLY (bit-for-bit identical output), so
    # every result in the paper is unaffected. See app.enable_realistic_mode().
    DRIVE_EXPONENT = None

    def simulate_single(self, duration_sec=DURATION, seed=None):   # A3: Euler-Maruyama
        rng = np.random.default_rng(seed)
        steps = int(duration_sec * self.fs)
        state = np.array([0.1, 0.0, 0.0])
        eeg = np.empty(steps)
        sq = np.sqrt(self.dt)
        for i in range(steps):
            eeg[i] = state[0]
            state = state + self.dt * self.drift(state, i * self.dt) \
                    + sq * self.sigma * rng.standard_normal(3)
        if self.DRIVE_EXPONENT:
            eeg = self._tissue_filter(eeg)
        return eeg

    def _tissue_filter(self, eeg):
        """OPTIONAL (realistic mode): apply a 1/f^beta tissue/volume-conduction filter to the
        model's output. This represents the dendritic low-pass filtering and volume conduction
        that shape the aperiodic (1/f) slope of scalp EEG -- physics absent from a rate model.
        It is MULTIPLICATIVE in the frequency domain, so it re-shapes the broadband slope while
        PRESERVING the relative band structure (and hence the TD/ASD separation) the model
        produced. DRIVE_EXPONENT sets the target aperiodic exponent (beta; ~1.7 gives a
        typically-developing exponent near 1.33)."""
        n = len(eeg)
        e = (eeg - eeg.mean()) / (eeg.std() + 1e-12)
        F = np.fft.rfft(e)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        H = 1.0 / (freqs ** (self.DRIVE_EXPONENT / 2.0))   # |H(f)| for a 1/f^beta PSD filter
        H = H / H[np.argmin(np.abs(freqs - 10.0))]         # normalize gain at 10 Hz (alpha)
        out = np.fft.irfft(F * H, n=n)
        return (out - out.mean()) / (out.std() + 1e-12)

    def averaged_spectrogram_db(self, seed=42, n_runs=N_RUNS, duration_sec=DURATION):  # A4
        Sxx_acc, f, t = None, None, None
        for r in range(n_runs):
            eeg = self.simulate_single(duration_sec=duration_sec, seed=seed + r)
            f, t, Sxx = spectrogram(eeg, fs=self.fs, nperseg=NPERSEG, noverlap=NOVERLAP)
            Sxx_acc = Sxx if Sxx_acc is None else Sxx_acc + Sxx
        return 10 * np.log10(Sxx_acc / n_runs + 1e-10), f, t


# %% ------------------------------------------------------------------
# 3. Auxiliary functions
# --------------------------------------------------------------------
def spectrogram_db_of_signal(signal):
    f, t, Sxx = spectrogram(signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return 10 * np.log10(Sxx + 1e-10), f, t

def band_power_temporal(spec, f, t):
    out = {}
    for band, (fmin, fmax) in BANDS.items():
        idx = (f >= fmin) & (f < fmax)
        out[band] = np.mean(spec[idx, :], axis=0) if np.any(idx) else np.zeros(spec.shape[1])
    return out

def band_power_mean(spec, f, t):
    return {b: float(np.mean(p)) for b, p in band_power_temporal(spec, f, t).items()}

def mean_from_temporal(pow_temp):
    return {b: float(np.mean(np.asarray(pow_temp[b]))) for b in BANDS if b in pow_temp}

def distance_to_td(power_mean_dict, td_mean_dict):
    diff = 0.0
    for band in BANDS:
        v1, v2 = power_mean_dict.get(band, np.nan), td_mean_dict.get(band, np.nan)
        if not np.isnan(v1) and not np.isnan(v2):
            diff += abs(v1 - v2)
    return diff

def normalize_relative(spec, spec_ref=None):
    target_mean = np.mean(spec_ref) if spec_ref is not None else np.mean(spec)
    return spec * (target_mean / np.mean(spec))

def normalize_power_relative(pow_temp_dict, target_mean_db):
    all_lin = []
    for arr in pow_temp_dict.values():
        all_lin.extend(10 ** (np.asarray(arr) / 10))
    factor = (10 ** (target_mean_db / 10)) / np.mean(all_lin)
    return {b: 10 * np.log10(10 ** (np.asarray(arr) / 10) * factor + 1e-10)
            for b, arr in pow_temp_dict.items()}

def model_normspec(model, seed=42, n_runs=N_RUNS):           # R1: single shared pipeline
    spec_raw, f, t = model.averaged_spectrogram_db(seed=seed, n_runs=n_runs)
    spec = normalize_relative(spec_raw, None)
    return spec, band_power_mean(spec, f, t), f, t

# ---- S1: real-EEG level alignment ------------------------------------------
# Real EEG absolute amplitude is ARBITRARY (electrode reference, amplifier gain,
# stored units V vs µV), so its dB level does not match the model's. Left unaligned,
# fitting the model to real band powers rails g_ei/g_ee to their bounds (RMSE ~50 dB)
# and every subject collapses to the SAME recommendation. We remove that meaningless
# offset by shifting each loaded spectrogram so its OVERALL mean dB matches the
# model's reference level. The shift is additive in dB => inter-band differences
# (the spectral SHAPE that carries the E/I information) are preserved EXACTLY; only
# the absolute level changes. Anchor = overall mean dB of a synthetic baseline;
# set MODEL_LEVEL_ANCHOR = "ASD" to anchor to the ASD baseline instead (it only sets
# the common level, not the shape). This affects real-EEG loading ONLY; every
# synthetic path is untouched, and no model constant is changed.
MODEL_LEVEL_ANCHOR = "TD"                       # "TD" or "ASD"
_MODEL_REF_MEAN_DB = None
def _model_ref_mean_db():
    global _MODEL_REF_MEAN_DB
    if _MODEL_REF_MEAN_DB is None:
        params = ASD_PARAMS if MODEL_LEVEL_ANCHOR == "ASD" else TD_PARAMS
        spec, _, _, _ = model_normspec(NeuralMass(**params), seed=43)
        _MODEL_REF_MEAN_DB = float(np.mean(spec))
    return _MODEL_REF_MEAN_DB

def align_level_to_model(spec_db):
    """Shift a real-EEG spectrogram so its overall mean dB matches the model's
    reference level (removes the arbitrary amplifier/reference/units gain). Additive
    in dB => band SHAPE (the inter-band structure) is preserved; only the meaningless
    absolute level changes. Without this, fits to real EEG rail to their bounds."""
    return spec_db + (_model_ref_mean_db() - float(np.mean(spec_db)))

def fit_model_to_real(real_pow_mean_list, max_iters=30, popsize=8, seed=0, n_runs=3):  # F1
    real_avg = {b: float(np.mean([p[b] for p in real_pow_mean_list])) for b in BANDS}
    # bounds = [g_nmda_ei, g_nmda_ee, low_freq_amp]. NOTE: fit reliability is limited by a
    # MODEL property, not the optimizer — near this parameter region the rate model sits by
    # a regime transition (oscillation amplitude collapses), so its band vector is a
    # DISCONTINUOUS function of the parameters (a ~0.02 change in g_nmda_ei can shift bands
    # by tens of dB). Individual parameter estimates are therefore ill-posed and should not
    # be over-interpreted; keep per-subject fitting exploratory and lean on GROUP-level
    # results for any claim. low_freq_amp upper is 0.8 (was 0.5) to avoid ceiling-railing.
    bounds = [(0.2, 1.5), (0.2, 1.5), (0.0, 0.8)]
    def objective(params):
        g_ei, g_ee, lf = params
        model = NeuralMass(g_nmda_ei=g_ei, g_nmda_ee=g_ee, low_freq_amp=lf)
        spec, f, t = model.averaged_spectrogram_db(seed=42, n_runs=n_runs)
        pm = band_power_mean(spec, f, t)
        return float(np.mean([(pm[b] - real_avg[b]) ** 2 for b in BANDS]))
    res = differential_evolution(objective, bounds, seed=seed, tol=1e-2,
                                 maxiter=max_iters, popsize=popsize, polish=True,
                                 mutation=(0.5, 1.0), recombination=0.7)
    return res.x, float(np.sqrt(res.fun))

def bootstrap_significance(per_sample_distances, post_distance,               # E1
                           n_boot=1000, alpha=0.05, seed=0):
    d = np.asarray(per_sample_distances, dtype=float)
    n = len(d)
    if n < 3:
        return {'ok': False, 'reason': f'n={n}<3'}
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    threshold = float(np.percentile(d, 100 * alpha))
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return {'ok': True, 'n': n, 'threshold': threshold, 'mean_ci': ci,
            'significant': bool(post_distance < threshold)}


# %% ------------------------------------------------------------------
# 4. Real EEG processing from LOCAL PATHS (mne lazy; no Colab). V1
# --------------------------------------------------------------------
def process_file_from_path(file_path):
    if not ensure("mne"):
        return None
    import mne
    low = file_path.lower()
    ext = '.fif' if low.endswith('.fif.gz') else os.path.splitext(low)[1]
    # extension -> MNE reader (many formats, not only .set/.fdt)
    readers = {
        '.set':  mne.io.read_raw_eeglab,        # EEGLAB (.set + .fdt sidecar)
        '.edf':  mne.io.read_raw_edf,           # European Data Format
        '.bdf':  mne.io.read_raw_bdf,           # BioSemi
        '.gdf':  mne.io.read_raw_gdf,           # General Data Format
        '.vhdr': mne.io.read_raw_brainvision,   # BrainVision (.vhdr + .eeg/.vmrk)
        '.fif':  mne.io.read_raw_fif,           # Elekta/MNE FIF
        '.cnt':  mne.io.read_raw_cnt,           # Neuroscan / ANT
        '.cdt':  mne.io.read_raw_curry,         # Curry
        '.dap':  mne.io.read_raw_curry,
        '.mff':  mne.io.read_raw_egi,           # EGI (folder bundle)
        '.raw':  mne.io.read_raw_egi,           # EGI simple binary
        '.nxe':  mne.io.read_raw_eximia,        # Nexstim Eximia
    }

    def _load_raw():
        """Try the extension-specific reader, then MNE's generic autodetect, then a
        few common fallbacks, so unusual or mislabeled files still open."""
        attempts = []
        if ext in readers:
            attempts.append(readers[ext])
        if hasattr(mne.io, "read_raw"):
            attempts.append(mne.io.read_raw)            # generic autodetect (MNE >= 1.0)
        attempts += [mne.io.read_raw_edf, mne.io.read_raw_eeglab, mne.io.read_raw_gdf]
        last = None
        for fn in attempts:
            try:
                return fn(file_path, preload=True, verbose=False)
            except TypeError:                            # reader without verbose= kwarg
                try:
                    return fn(file_path, preload=True)
                except Exception as e:
                    last = e
            except Exception as e:
                last = e
        raise last if last is not None else RuntimeError("no MNE reader succeeded")

    try:
        raw = _load_raw()
        raw.resample(FS)
        raw.filter(0.5, 80, fir_design='firwin', verbose=False)
        raw.notch_filter(50, verbose=False)
        channels = ['Cz', 'Fz', 'Pz', 'C3', 'C4']
        available = [ch for ch in channels if ch in raw.ch_names]
        if available:
            raw.pick_channels(available)
        sig = raw.get_data().mean(axis=0)
        need = int(DURATION * FS)
        sig = np.pad(sig, (0, need - len(sig))) if len(sig) < need else sig[:need]
        # S2: standardize signal amplitude BEFORE the spectrogram. Real EEG is in Volts
        # (~1e-5), so its PSD (~1e-11..1e-9) falls at/below the fixed 1e-10 floor in
        # spectrogram_db_of_signal, which then FLATTENS every real recording to a
        # featureless ~-100 dB line (all bands within ~0.1 dB). Unit-variance scaling
        # lifts the PSD well above the floor so the true spectral SHAPE (1/f slope,
        # alpha peak, per-subject differences) survives. EEG absolute amplitude is
        # arbitrary anyway (reference/amplifier/units), so this is standard and lossless
        # for shape. Real-EEG loading ONLY; the model's own signal is unaffected.
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-12)
        spec, f, t = spectrogram_db_of_signal(sig)
        spec = align_level_to_model(spec)         # S1: align overall level to the model
        return spec, band_power_temporal(spec, f, t), f, t
    except Exception as e:
        log.warning("Error processing %s: %s", os.path.basename(file_path), e)
        return None

def _iter_dir_headers(root_dir):
    """Recursively collect primary EEG header files under a directory. .mff bundles
    are returned as a single entry (the folder). macOS junk and sidecars are skipped."""
    found = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d != '__MACOSX']
        for d in list(dirs):                                  # .mff = folder bundle
            if d.lower().endswith('.mff'):
                found.append(os.path.join(root, d)); dirs.remove(d)
        for fn in files:
            if fn.startswith('._'):                           # macOS AppleDouble
                continue
            if fn.lower().endswith(EEG_EXTS):
                found.append(os.path.join(root, fn))
    return sorted(found)

def _expand_paths(paths):
    """Accept files, directories, or .zip archives (a .zip may hold 2+ EEGs) in many
    formats. Returns the list of primary EEG header files found (EEGLAB .set, EDF/BDF,
    GDF, BrainVision .vhdr, FIF, Neuroscan .cnt, Curry .cdt, EGI .mff/.raw, Eximia
    .nxe...). Companion/sidecar files (.fdt, .eeg, .vmrk, .dat) stay on disk next to
    their header but are NOT returned as separate recordings."""
    out = []
    for p in paths:
        low = p.lower()
        if os.path.isdir(p) and low.endswith('.mff'):         # a selected .mff bundle
            out.append(p); continue
        if os.path.isfile(p) and low.endswith('.zip'):        # archive of EEGs
            d = _new_tmpdir()
            try:
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(d)
                hdrs = _iter_dir_headers(d)
                if hdrs:
                    out.extend(hdrs)
                    log.info("Zip '%s' -> %d EEG file(s).", os.path.basename(p), len(hdrs))
                else:
                    log.warning("No EEG files found inside zip: %s", os.path.basename(p))
            except Exception as e:
                log.warning("Could not read zip %s: %s", os.path.basename(p), e)
        elif os.path.isdir(p):                                # a folder of EEGs
            hdrs = _iter_dir_headers(p)
            if hdrs:
                out.extend(hdrs)
            else:
                log.warning("No EEG files found in folder: %s", p)
        elif os.path.isfile(p):                               # a single explicit file
            out.append(p)                                     # try it, whatever the ext
        else:
            log.warning("Path not found: %s", p)
    return out

def load_eeg_from_paths(paths):
    """Return (mean_spec, mean_pow_temporal, f, t, per_sample_pow_list, n)."""
    files = _expand_paths(paths)
    specs, pows, fg, tg = [], [], None, None
    for fp in files:
        res = process_file_from_path(fp)
        if res:
            spec, pt, f, t = res
            specs.append(spec); pows.append(pt)
            if fg is None:
                fg, tg = f, t
    if not specs:
        return None, None, None, None, [], 0
    spec_mean = np.mean(specs, axis=0)          # <-- multiple files are AVERAGED here
    pow_mean_temp = {b: np.mean([p[b] for p in pows], axis=0) for b in BANDS}
    return spec_mean, pow_mean_temp, fg, tg, pows, len(specs)

def pick_files_dialog(title="Select EEG files"):
    """Open a native OS window to choose one or more EEG files (many formats) and/or
    a .zip containing several EEGs. Needs a graphical environment (native desktop, or
    WSLg/X under WSL). If unavailable, returns [] and the caller falls back to CONFIG.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        paths = filedialog.askopenfilenames(
            title=title,
            filetypes=[
                ("EEG & zip archives",
                 "*.set *.edf *.bdf *.gdf *.vhdr *.fif *.fif.gz *.cnt *.cdt *.mff "
                 "*.raw *.nxe *.zip"),
                ("EEGLAB (.set + .fdt)", "*.set"),
                ("EDF / BDF", "*.edf *.bdf"),
                ("BrainVision (.vhdr + .eeg/.vmrk)", "*.vhdr"),
                ("GDF", "*.gdf"),
                ("FIF", "*.fif *.fif.gz"),
                ("Neuroscan / Curry / EGI / Eximia", "*.cnt *.cdt *.mff *.raw *.nxe"),
                ("Zip of 2+ EEGs", "*.zip"),
                ("All files", "*.*"),
            ])
        root.destroy()
        return list(paths)
    except Exception as e:
        log.warning("File dialog unavailable (%s). Under WSL, enable WSLg or an X "
                    "server, or set CONFIG['asd_paths']/['td_paths'] instead.", e)
        return []

def _resolve_input_paths(config):
    """Return (asd_paths, td_paths), opening file windows when CONFIG['ask_files']
    is True and no explicit paths were given."""
    asd_paths = list(config.get("asd_paths") or [])
    td_paths  = list(config.get("td_paths") or [])
    if config.get("ask_files"):
        if not asd_paths:
            print("👉 A window will open to select the ASD EEG files. You can pick several "
                  "files in any supported format (.set/.edf/.bdf/.gdf/.vhdr/.fif/.cnt/"
                  ".cdt/.mff/.raw/.nxe) or a single .zip with 2+ EEGs. Cancel = synthetic.")
            asd_paths = pick_files_dialog("Select ASD EEG files or a .zip (Cancel = synthetic)")
        if not td_paths:
            print("👉 Now select the TD reference EEG files (same formats, or a .zip). "
                  "Optional — Cancel to use the synthetic TD reference.")
            td_paths = pick_files_dialog("Select TD EEG files or a .zip (optional, Cancel to skip)")
    return asd_paths, td_paths


# %% ------------------------------------------------------------------
# 5. Selection -> modulation mapping (A2) and surrogate vector helpers
# --------------------------------------------------------------------
def selections_to_mod(selecciones, neuromod_selecciones):
    mod = {}
    for p in PROTEINS:
        sel = selecciones.get(p['name'], 'None')
        if sel == 'None':
            mod[p['name']] = 0.0
        else:
            action = 1.0 if sel == 'Agonist' else -1.0
            mod[p['name']] = action * p['agonist_sign'] * p['base_frac']
    neuromod = {'dopamine':       float(neuromod_selecciones.get('Dopamine', 0.0)),
                'serotonin':      float(neuromod_selecciones.get('Serotonin', 0.0)),
                'norepinephrine': float(neuromod_selecciones.get('Norepinephrine', 0.0))}
    return mod, neuromod

def mod_to_vector(mod, neuromod):
    return np.array([mod.get(k, 0.0) for k in TARGET_KEYS]
                    + [neuromod.get(k, 0.0) for k in NEUROMOD_KEYS], dtype=float)

def vector_to_mod(vec):
    mod = {name: float(vec[i]) for i, name in enumerate(TARGET_KEYS)}
    neuromod = {k: float(vec[len(TARGET_KEYS) + j]) for j, k in enumerate(NEUROMOD_KEYS)}
    return mod, neuromod

def build_model(baseline, vec):
    mod, neuromod = vector_to_mod(vec)
    return NeuralMass(g_nmda_ei=baseline['g_nmda_ei'], g_nmda_ee=baseline['g_nmda_ee'],
                      low_freq_amp=baseline['low_freq_amp'], mod=mod,
                      dopamine=neuromod['dopamine'], serotonin=neuromod['serotonin'],
                      norepinephrine=neuromod['norepinephrine'])


# ---- S5: reliable per-subject recommendation by response projection ----------
# The per-subject MODEL INVERSION (fit_model_to_real) is ill-posed near the model's
# bifurcation: tiny parameter changes flip the spectrum, so fitted baselines and any
# recommendation read off them are unstable (they jump with seed/bound). This method
# never inverts the model. Instead it (1) takes each subject's reliable, DATA-LEVEL
# spectral deviation from TD, and (2) finds which agonist/inhibitor moves best cancel
# it, using modulation RESPONSES measured on STABLE reference baselines (the terrain
# pharmacheck validates; response noise ~0.25 dB). Robustness = stability selection:
# the greedy match is repeated over an ENSEMBLE of stable reference baselines x seeds,
# and a move is reported only if it is selected in a large fraction of the ensemble,
# that fraction being its confidence. A noise gate ignores gains within the noise
# floor. Purely forward + validated; still a hypothesis generator, not clinical advice.
STABLE_REF_BASELINES = [                              # all verified stable (no collapse)
    {'g_nmda_ei': 0.6, 'g_nmda_ee': 0.8, 'low_freq_amp': 0.15},   # ASD (pharmacheck-validated)
    {'g_nmda_ei': 0.8, 'g_nmda_ee': 0.9, 'low_freq_amp': 0.10},
    {'g_nmda_ei': 1.0, 'g_nmda_ee': 1.0, 'low_freq_amp': 0.05},
]
RESP_SEEDS = [3, 11, 23, 37, 51, 67, 83, 97]
RESP_N_RUNS = 24
RESP_NOISE_GATE = 0.4                                 # dB; ignore gains below the response noise floor
REC_STABILITY_THRESHOLD = 0.6                         # report a move if selected in >=60% of the ensemble
_RESP_ENSEMBLE = None

def _modulation_moves():
    moves = []
    for i, p in enumerate(PROTEINS):
        for act, dirn in ((+1.0, 'Agonist'), (-1.0, 'Inhibitor')):
            vec = np.zeros(len(MOD_ORDER)); signed = act * p['agonist_sign'] * p['base_frac']
            vec[i] = signed
            moves.append((p['name'], dirn, signed, vec))
    for j, k in enumerate(NEUROMOD_KEYS):
        for lvl, dirn in ((+0.5, 'Increase'), (-0.5, 'Decrease')):
            vec = np.zeros(len(MOD_ORDER)); vec[len(TARGET_KEYS) + j] = lvl
            moves.append((k.capitalize(), dirn, lvl, vec))
    return moves

_MOVE_SIGN = {(m[0], m[1]): m[2] for m in _modulation_moves()}

def _bands_vec(baseline, vec, seed, n_runs):
    sp, f, t = build_model(baseline, vec).averaged_spectrogram_db(seed=seed, n_runs=n_runs)
    pm = band_power_mean(sp, f, t)
    return np.array([pm[b] for b in BANDS])

def _build_response_ensemble(references=None, seeds=None, n_runs=None):
    global _RESP_ENSEMBLE
    references = references or STABLE_REF_BASELINES
    seeds = seeds or RESP_SEEDS
    n_runs = n_runs or RESP_N_RUNS
    moves = _modulation_moves(); ens = []
    for bl in references:
        for sd in seeds:
            base = _bands_vec(bl, np.zeros(len(MOD_ORDER)), sd, n_runs)
            ens.append({(name, dirn): _bands_vec(bl, vec, sd, n_runs) - base
                        for (name, dirn, signed, vec) in moves})
    _RESP_ENSEMBLE = ens
    return ens

def _greedy_project(delta, R, top_k, gate):
    residual = np.asarray(delta, float).copy(); chosen = []; used = set()
    for _ in range(top_k):
        best = None; best_norm = np.linalg.norm(residual) - gate
        for (name, dirn), r in R.items():
            if name in used:
                continue
            nn = np.linalg.norm(residual - r)
            if nn < best_norm:
                best_norm = nn; best = (name, dirn, r)
        if best is None:
            break
        chosen.append((best[0], best[1])); used.add(best[0]); residual = residual - best[2]
    return chosen

def recommend_by_projection(subject_bands, td_bands, top_k=4, gate=None):
    """Reliable per-subject recommendation: which agonist/inhibitor moves best cancel the
    subject's spectral deviation from TD, by stability selection over an ensemble of STABLE
    reference baselines x stochastic seeds (NO unstable model inversion). Returns
    ([(target, direction, frequency)] sorted by frequency, ensemble_size)."""
    global _RESP_ENSEMBLE
    if _RESP_ENSEMBLE is None:
        _build_response_ensemble()
    gate = RESP_NOISE_GATE if gate is None else gate
    delta = np.asarray(td_bands, float) - np.asarray(subject_bands, float)
    counts = {}
    for R in _RESP_ENSEMBLE:
        for key in _greedy_project(delta, R, top_k, gate):
            counts[key] = counts.get(key, 0) + 1
    n = len(_RESP_ENSEMBLE)
    recs = sorted(((name, dirn, c / n) for (name, dirn), c in counts.items()),
                  key=lambda x: -x[2])
    return recs, n


# %% ------------------------------------------------------------------
# 6. Surrogate neural network (emulator). tf lazy. V1 + R2 fidelity
# --------------------------------------------------------------------
class NeuronAI:
    def __init__(self, model_path=None):
        self.model = None
        self.ready = False
        self.fidelity = None
        if model_path and os.path.exists(model_path):
            if ensure("tensorflow"):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.ready = True

    def build_arch(self, input_spec_shape, n_mod_params):
        ensure("tensorflow")
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        tf.random.set_seed(42)
        spec_in = layers.Input(shape=(input_spec_shape[0], input_spec_shape[1], 1))
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spec_in)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        mod_in = layers.Input(shape=(n_mod_params,))
        y = layers.Dense(64, activation='relu')(mod_in)
        y = layers.Dense(64, activation='relu')(y)
        z = layers.concatenate([x, y])
        z = layers.Dense(128, activation='relu')(z)
        z = layers.Dropout(0.3)(z)
        out_dist  = layers.Dense(1, name='distance')(z)
        out_bands = layers.Dense(len(BANDS), name='bands')(z)
        model = Model(inputs=[spec_in, mod_in], outputs=[out_dist, out_bands])
        model.compile(optimizer='adam', loss={'distance': 'mse', 'bands': 'mse'},
                      loss_weights={'distance': 1.0, 'bands': 0.5})
        self.model = model
        return model

    def train(self, X_spec, X_mod, y_dist, y_bands, epochs=30, batch_size=64,
              validation_split=0.2, model_out='neuron_ai_model.h5'):
        if X_spec.ndim == 3:
            X_spec = np.expand_dims(X_spec, -1)
        if self.model is None:
            self.build_arch(X_spec.shape[1:3], X_mod.shape[1])
        history = self.model.fit([X_spec, X_mod], [y_dist, y_bands],
                                 epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, verbose=0)
        self.ready = True
        self.model.save(model_out)
        return history

    def evaluate_fidelity(self, X_spec, X_mod, y_dist):        # R2
        if X_spec.ndim == 3:
            X_spec = np.expand_dims(X_spec, -1)
        pred = self.model.predict([X_spec, X_mod], verbose=0)[0].ravel()
        y = np.asarray(y_dist, float).ravel()
        mae = float(np.mean(np.abs(pred - y)))
        denom = np.maximum(np.abs(y), 1e-6)
        mape = float(np.mean(np.abs((pred - y) / denom)) * 100)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        self.fidelity = {'mae_db': mae, 'mape_pct': mape, 'r2': 1.0 - ss_res / ss_tot}
        return self.fidelity

    def predict(self, spec, mod_vector):
        if not self.ready:
            raise ValueError("Surrogate not trained/loaded.")
        s = np.expand_dims(spec, axis=(0, -1))
        m = np.expand_dims(mod_vector, axis=0)
        dist, bands = self.model.predict([s, m], verbose=0)
        return float(dist[0][0]), bands[0]

    def optimize_modulation(self, spec, mod_space):
        best_mod, best_dist = None, np.inf
        for mod in mod_space:
            d, _ = self.predict(spec, mod)
            if d < best_dist:
                best_dist, best_mod = d, mod
        return best_mod, best_dist

    def get_band_sensitivity(self, spec, mod_vector, f, epsilon=0.01, seed=0):
        rng = np.random.RandomState(seed)
        sens = {}
        base, _ = self.predict(spec, mod_vector)
        for band, (fmin, fmax) in BANDS.items():
            idx = (f >= fmin) & (f < fmax)
            if np.any(idx):
                sp = spec.copy()
                sp[idx, :] = spec[idx, :] * (1 + epsilon * rng.randn(*spec[idx, :].shape))
                d, _ = self.predict(sp, mod_vector)
                sens[band] = abs(d - base)
            else:
                sens[band] = 0.0
        total = sum(sens.values())
        return {b: (v / total if total > 0 else 0.0) for b, v in sens.items()}


# %% ------------------------------------------------------------------
# 7. Synthetic training-data generation (from the simulator). 11-dim vectors.
# --------------------------------------------------------------------
def generate_training_data(n_samples=5000, seed=42, n_runs=N_RUNS, out='training_data.npz'):
    rng = np.random.default_rng(seed)
    X_spec, X_mod, y_dist, y_bands = [], [], [], []
    _, pow_mean_td, f, t = model_normspec(NeuralMass(**TD_PARAMS), seed=seed + 1, n_runs=n_runs)
    for i in range(n_samples):
        baseline = ASD_PARAMS if rng.random() > 0.5 else TD_PARAMS
        vec = np.concatenate([rng.uniform(-0.4, 0.4, size=len(TARGET_KEYS)),
                              rng.uniform(-1.0, 1.0, size=len(NEUROMOD_KEYS))])
        model = build_model(baseline, vec)
        spec, pow_mean, _, _ = model_normspec(model, seed=seed + i + 10, n_runs=n_runs)
        X_spec.append(spec); X_mod.append(vec)
        y_dist.append(distance_to_td(pow_mean, pow_mean_td))
        y_bands.append([pow_mean.get(b, 0.0) for b in BANDS])
        if (i + 1) % 500 == 0:
            print(f"  generated {i + 1}/{n_samples}")
    X_spec, X_mod = np.array(X_spec), np.array(X_mod)
    y_dist, y_bands = np.array(y_dist), np.array(y_bands)
    np.savez(out, X_spec=X_spec, X_mod=X_mod, y_dist=y_dist, y_bands=y_bands, f=f, t=t)
    print(f"  saved -> {out}")
    return X_spec, X_mod, y_dist, y_bands, f, t


# %% ------------------------------------------------------------------
# 8. Self-test guarding the fixes against regressions. V1
# --------------------------------------------------------------------
def print_mapping():
    """Print the drug->parameter map with its physiological rationale and references."""
    print("Drug -> parameter mapping (directions are literature-supported; magnitudes are nominal):\n")
    for p in PROTEINS:
        print(f"• {p['name']}  ->  {p['knob']}   [{p['effect_kind']}]")
        print(f"    pathway : {p['pathway']}")
        print(f"    EEG     : {p['eeg_signature']}")
        if p['refs']:
            for k in p['refs']:
                print(f"    ref     : {REFERENCES.get(k, k)}")
        print()
    print("Neuromodulatory tone (nominal scale factors, NOT literature magnitudes):")
    for nm, d in NEUROMOD_COEFFS.items():
        print(f"• {nm}: " + ", ".join(f"{k}{v:+.2f}" for k, v in d.items()))
    print("\nNote: 'by_construction' knobs encode their EEG signature directly; matching them is")
    print("not independent evidence. 'emergent' signatures arise from the network dynamics and")
    print("are checked by --mode pharmacheck. AMPA and NMDA-NR2A share the E->E term (documented).")


def check_pharmaco_eeg_signatures(baseline=None, n_runs=3, tol_db=0.15, verbose=True):
    """
    Exploratory CONSTRUCT-VALIDITY check: for canonical single-drug manipulations,
    does the simulated EEG move in the direction reported in the literature?

    This is not a pass/fail validation. A coarse rate model need not reproduce every
    pharmaco-EEG signature, and 'by_construction' knobs match trivially. Interpret the
    'emergent' rows as the informative ones. Uses a fixed noise seed so basal vs
    modulated comparisons are fair.
    """
    baseline = baseline or ASD_PARAMS   # ASD profile: the drugs' target features exist here
    seed = 123
    # (target, action tested, primary band, expected change, reference key)
    checks = [
        ("GABA_A",      "Agonist",   "β", "increase", "Berro2021"),
        ("NMDA (NR2B)", "Inhibitor", "γ", "increase", "KetamineGammaModel2025"),
        ("AMPA",        "Inhibitor", "γ", "decrease", "RubensteinMerzenich2003"),
        ("alpha7_nAChR","Agonist",   "γ", "increase", "Stoiljkovic2016"),
        ("5HT2A",       "Agonist",   "α", "decrease", "Muthukumaraswamy2013"),
    ]
    pinfo = {p["name"]: p for p in PROTEINS}
    _, basal, _, _ = model_normspec(NeuralMass(**baseline), seed=seed, n_runs=n_runs)

    rows = {'Target': [], 'Action': [], 'Band': [], 'Expected': [], 'Model': [],
            'Match': [], 'Type': []}
    n_emergent, n_emergent_match = 0, 0
    for name, action, band, expected, ref in checks:
        p = pinfo[name]
        frac = (1.0 if action == "Agonist" else -1.0) * p["agonist_sign"] * p["base_frac"]
        _, pm, _, _ = model_normspec(NeuralMass(**baseline, mod={name: frac}),
                                     seed=seed, n_runs=n_runs)
        delta = pm[band] - basal[band]
        got = "increase" if delta > tol_db else ("decrease" if delta < -tol_db else "no change")
        match = (got == expected)
        rows['Target'].append(name); rows['Action'].append(action); rows['Band'].append(band)
        rows['Expected'].append(expected); rows['Model'].append(f"{got} ({delta:+.2f} dB)")
        rows['Match'].append("yes" if match else "no"); rows['Type'].append(p["effect_kind"])
        if p["effect_kind"] == "emergent":
            n_emergent += 1; n_emergent_match += int(match)

    if verbose:
        print("Construct-validity check — does the model reproduce known pharmaco-EEG directions?")
        print("(baseline = ASD profile; 'emergent' rows are the informative ones)\n")
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nEmergent signatures matching the literature: {n_emergent_match}/{n_emergent}")
        print("Interpretation: high match = supportive construct validity. IMPORTANT: these")
        print("responses are STATE-DEPENDENT (they depend on the E/I operating point). On a TD")
        print("baseline some signs differ, which is physiologically plausible but means the")
        print("tool's outputs are conditioned on the assumed ASD profile. Re-run with")
        print("baseline=TD_PARAMS to see this. 'by_construction' rows are trivial.")
    return rows, (n_emergent_match, n_emergent)


def verify_invariants(verbose=True):
    ok = True
    a = NeuralMass(**ASD_PARAMS).simulate_single(duration_sec=2.0, seed=7)
    b = NeuralMass(**ASD_PARAMS).simulate_single(duration_sec=2.0, seed=7)
    det = np.allclose(a, b); ok &= det
    base = NeuralMass(**ASD_PARAMS)
    nr2b = NeuralMass(**ASD_PARAMS, mod={'NMDA (NR2B)': -0.2})
    s0, _, _ = base.averaged_spectrogram_db(seed=1, n_runs=2)
    s1, _, _ = nr2b.averaged_spectrogram_db(seed=1, n_runs=2)
    nr2b_effect = not np.allclose(s0, s1); ok &= nr2b_effect
    _, pm_asd, f, t = model_normspec(NeuralMass(**ASD_PARAMS), seed=2, n_runs=2)
    _, pm_td, _, _ = model_normspec(NeuralMass(**TD_PARAMS), seed=3, n_runs=2)
    dpos = distance_to_td(pm_asd, pm_td) > 0; ok &= dpos
    # every target must change the spectrum (no dead / fully-degenerate knobs)
    base_s, _, _ = NeuralMass(**ASD_PARAMS).averaged_spectrogram_db(seed=5, n_runs=2)
    no_dead = True
    for p in PROTEINS:
        s, _, _ = NeuralMass(**ASD_PARAMS, mod={p['name']: 0.25 * p['agonist_sign']}
                             ).averaged_spectrogram_db(seed=5, n_runs=2)
        no_dead &= (not np.allclose(base_s, s))
    ok &= no_dead
    if verbose:
        print(f"[verify] determinism={det}  NR2B_has_effect={nr2b_effect}  "
              f"ASD_basal_distance>0={dpos}  no_dead_knobs={no_dead}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# %% ------------------------------------------------------------------
# 9. CONFIG - edit this block to run your own scenario
# --------------------------------------------------------------------
CONFIG = {
    # Receptor targets: 'Agonist' | 'Inhibitor' | 'None' (omit = None)
    "modulations": {
        "GABA_A": "Agonist",
        "NMDA (NR2B)": "Inhibitor",
    },
    "neuromodulators": {"Dopamine": 0.0, "Serotonin": 0.0, "Norepinephrine": 0.0},

    # Real EEG (optional): local .set/.edf/.gdf files or folders. Empty = synthetic.
    "asd_paths": [],
    "td_paths":  [],
    # Set True (or run with --pick-files) to choose files from a pop-up window
    # instead of listing paths above. Needs a desktop/GUI (WSLg or X under WSL).
    "ask_files": False,

    "normalize": True,
    # FIX B: in --mode optimize, give ONE recommendation per ASD subject (True) or a
    # single group-averaged recommendation (False). Overridable with --per-subject /
    # --group. With no real ASD files, per-subject falls back to the synthetic group.
    "per_subject": True,
    "use_surrogate": False,                 # requires a trained model file
    "surrogate_model_path": "neuron_ai_model.h5",

    "seed": 42,
    "n_runs": N_RUNS,
    "output_dir": "neuron_asd_out",

    # training params (only used in --mode train)
    "train_samples": 5000,
    "train_epochs": 30,
}


# %% ------------------------------------------------------------------
# 10. Runnable pipelines
# --------------------------------------------------------------------
def run_simulation(config):
    outdir = config["output_dir"]; os.makedirs(outdir, exist_ok=True)
    seed = config["seed"]; n_runs = config["n_runs"]
    print("⚡ Simulation (VSCode edition)\n"
          "   NOTE: hypothesis-generation tool. Real-data examples are demonstrations,\n"
          "   not clinical evidence; significance needs n>=3 real ASD samples.\n")

    mod, neuromod = selections_to_mod(config.get("modulations", {}),
                                      config.get("neuromodulators", {}))
    mod_vector = mod_to_vector(mod, neuromod)
    active = {k: v for k, v in mod.items() if v != 0}
    print("📌 Active modulations (fractional):", active if active else "none")
    tone = {k: v for k, v in neuromod.items() if v != 0}
    if tone:
        print("   neuromodulatory tone:", tone)

    # ---- real EEG (optional; opens a window if CONFIG['ask_files']=True) ----
    real_asd_spec = real_asd_f = real_asd_t = None
    real_td_spec = None
    real_asd_pow_list = []
    asd_paths, td_paths = _resolve_input_paths(config)
    if asd_paths:
        r = load_eeg_from_paths(asd_paths)
        if r[0] is not None:
            real_asd_spec, _, real_asd_f, real_asd_t, real_asd_pow_list, n = r
            print(f"   ✅ {n} real ASD file(s) loaded and averaged")
    if td_paths:
        r = load_eeg_from_paths(td_paths)
        if r[0] is not None:
            real_td_spec, _, _, _, _, n = r
            print(f"   ✅ {n} real TD file(s) loaded and averaged")

    # ---- relative normalization of real data ----
    if config["normalize"] and real_asd_spec is not None:
        if real_td_spec is not None:
            real_asd_spec = normalize_relative(real_asd_spec, real_td_spec)
            real_td_spec = normalize_relative(real_td_spec, None)
            real_asd_pow_list = [normalize_power_relative(p, np.mean(real_td_spec))
                                 for p in real_asd_pow_list]
        else:
            real_asd_spec = normalize_relative(real_asd_spec, None)
            real_asd_pow_list = [normalize_power_relative(p, np.mean(real_asd_spec))
                                 for p in real_asd_pow_list]

    # ---- ASD baseline (real-fitted or synthetic) ----
    is_real_asd = real_asd_spec is not None
    if is_real_asd:
        pow_mean_asd = band_power_mean(real_asd_spec, real_asd_f, real_asd_t)
        (g_ei, g_ee, lf), fit_rmse = fit_model_to_real([pow_mean_asd])
        print(f"   Fit to real ASD (F1): g_nmda_ei={g_ei:.3f}, g_nmda_ee={g_ee:.3f}, "
              f"low_freq_amp={lf:.3f} | RMSE={fit_rmse:.3f} dB")
        asd_baseline = {'g_nmda_ei': g_ei, 'g_nmda_ee': g_ee, 'low_freq_amp': lf}
        spec_asd, f, t = real_asd_spec, real_asd_f, real_asd_t
        asd_label = "Real ASD (observed)"
    else:
        asd_baseline = ASD_PARAMS
        spec_asd, pow_mean_asd, f, t = model_normspec(NeuralMass(**ASD_PARAMS),
                                                      seed=seed, n_runs=n_runs)
        asd_label = "Synthetic ASD"

    # ---- TD reference ----
    if real_td_spec is not None:
        spec_td = real_td_spec
        pow_mean_td = band_power_mean(spec_td, f, t)
        td_label = "Real TD"
    else:
        spec_td, pow_mean_td, _, _ = model_normspec(NeuralMass(**TD_PARAMS),
                                                    seed=seed + 1, n_runs=n_runs)
        td_label = "Synthetic TD"

    # ---- FIX A: isolate the modulation effect --------------------------------
    # The modulation acts on the MODEL (fitted, in the real-data case). To attribute
    # a spectral change to the drugs -- and NOT to the model-fit residual -- the
    # pre/post contrast must be simulator-vs-simulator, from the SAME baseline and
    # with the SAME noise seed (paired). We therefore build a simulated baseline at
    # modulation = 0 ("sim0") and compare the modulated model against IT. This makes
    # the real and synthetic paths behave identically w.r.t. the modulation effect.
    # (Before this fix, the real path compared observed-real vs simulated-post, so
    #  the reported change conflated the fit residual with the drug effect.)
    zeros_vec = np.zeros(N_MOD)
    mod_seed = seed + 2                                    # shared seed => paired contrast
    spec_sim0, pow_mean_sim0, _, _ = model_normspec(
        build_model(asd_baseline, zeros_vec), seed=mod_seed, n_runs=n_runs)

    # ---- post-modulation (SAME seed as sim0) ----
    model_post = build_model(asd_baseline, mod_vector)
    spec_post, pow_mean_post, _, _ = model_normspec(model_post, seed=mod_seed, n_runs=n_runs)
    post_label = "Model + modulation" if is_real_asd else (asd_label + " + modulation")
    base_label = "Model baseline (mod=0)"

    # For the synthetic path the "ASD baseline" row IS the simulated baseline, so use
    # sim0 for both the table row and the figure panel (keeps them perfectly paired).
    if not is_real_asd:
        spec_asd = spec_sim0

    # Modulation reference = simulated baseline (fair "pre"). The raw observed distance
    # is kept only as context; for real data it also carries the fit residual.
    dist_sim0 = distance_to_td(pow_mean_sim0, pow_mean_td)   # fair reference for the drug effect
    dist_post = distance_to_td(pow_mean_post, pow_mean_td)
    dist_obs  = distance_to_td(pow_mean_asd,  pow_mean_td)   # observed baseline (context only)
    improvement = dist_post < dist_sim0                      # CLEAN modulation effect
    mod_delta = dist_sim0 - dist_post                        # >0 => modulation moved toward TD
    fit_gap   = abs(dist_obs - dist_sim0)                    # meaningful only for real data

    # ---- significance (E1) ----
    if real_asd_pow_list and len(real_asd_pow_list) >= 3:
        per_sample = [distance_to_td(mean_from_temporal(p), pow_mean_td)
                      for p in real_asd_pow_list]
        bs = bootstrap_significance(per_sample, dist_post)
        if bs['ok']:
            lo, hi = bs['mean_ci']
            sig_post = (f"{'p<0.05' if bs['significant'] else 'ns'} "
                        f"(post={dist_post:.1f}, thr(p5)={bs['threshold']:.1f}, "
                        f"meanCI=[{lo:.1f},{hi:.1f}])")
        else:
            sig_post = bs['reason']
    else:
        sig_post = f"n/a (n={len(real_asd_pow_list)}<3)"

    # ---- surrogate emulation (optional) ----
    emul, sens_str = "", "n/a"
    if config.get("use_surrogate"):
        ai = NeuronAI(config.get("surrogate_model_path"))
        if ai.ready:
            pred_dist, _ = ai.predict(spec_asd, mod_vector)
            emul = f"{pred_dist:.2f} dB"
            top = sorted(ai.get_band_sensitivity(spec_asd, mod_vector, f).items(),
                         key=lambda x: x[1], reverse=True)[:3]
            sens_str = ", ".join([f"{b}: {v:.3f}" for b, v in top if v > 0.01])
        else:
            print("   ⚠️ Surrogate requested but no trained model found; skipping emulation.")

    # ---- results table ----
    data = {'Condition': [], 'δ (dB)': [], 'θ (dB)': [], 'α (dB)': [], 'β (dB)': [],
            'γ (dB)': [], 'Distance to TD (dB)': [], 'Closer to TD?': [],
            'Significance (vs TD)': [], 'Emulated Dist.': [], 'Emulator Sensit.': []}
    def add_row(name, md, dist=None, closer="", sig="", em="", se=""):
        data['Condition'].append(name)
        for bnd in BANDS:
            v = md.get(bnd, np.nan)
            data[f'{bnd} (dB)'].append(f"{v:.2f}" if not np.isnan(v) else 'NaN')
        data['Distance to TD (dB)'].append(f"{dist:.2f}" if dist is not None else '-')
        data['Closer to TD?'].append(closer)
        data['Significance (vs TD)'].append(sig)
        data['Emulated Dist.'].append(em)
        data['Emulator Sensit.'].append(se)

    if is_real_asd:
        add_row(asd_label,  pow_mean_asd,  dist_obs,  "", "", "", "")   # observed real anchor
        add_row(base_label, pow_mean_sim0, dist_sim0, "", "", "", "")   # fair pre (fitted model)
    else:
        add_row(asd_label,  pow_mean_sim0, dist_sim0, "", "", "", "")   # synthetic baseline = sim0
    add_row(post_label, pow_mean_post, dist_post,
            "closer" if improvement else "farther", sig_post, emul, sens_str)
    add_row(td_label, pow_mean_td, 0.0, "", "", "", "")
    df = pd.DataFrame(data)

    print("\n📊 Results (relative normalization):")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.to_string(index=False))
    csv_path = os.path.join(outdir, "results.csv")
    df.to_csv(csv_path, index=False)

    # ---- FIX A: report the modulation effect on the fair (paired) baseline ----
    print(f"\n🔎 Modulation effect (simulator vs simulator, paired seed): "
          f"baseline={dist_sim0:.2f} dB -> modulated={dist_post:.2f} dB "
          f"(Δ {mod_delta:+.2f} dB; {'closer to' if improvement else 'farther from'} TD).")
    if is_real_asd:
        print(f"   Fit context: the observed real ASD is {dist_obs:.2f} dB from TD, the model "
              f"baseline is {dist_sim0:.2f} dB from TD")
        print(f"   (fit gap = {fit_gap:.2f} dB is model residual, NOT a drug effect; the drug "
              f"effect is the Δ above, measured on the model baseline).")

    # ---- figure ----
    labels = [asd_label, post_label, td_label]
    specs = [spec_asd, spec_post, spec_td]
    pow_temps = [band_power_temporal(s, f, t) for s in specs]
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    for i in range(n):
        ax0 = axes[0, i]
        im = ax0.pcolormesh(t, f, specs[i], shading='auto', cmap='jet')
        ax0.set_title(labels[i], fontsize=9)
        ax0.set_xlabel('Time (s)'); ax0.set_ylabel('Frequency (Hz)'); ax0.set_ylim(0, 80)
        plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        ax1 = axes[1, i]
        for band, color in zip(BANDS.keys(), ['blue', 'orange', 'green', 'red', 'purple']):
            ax1.plot(t, pow_temps[i][band], label=band, linewidth=1.5, color=color)
        ax1.set_title(labels[i], fontsize=9)
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Power (dB)')
        ax1.legend(loc='upper right', fontsize=7); ax1.grid(True, linestyle=':', alpha=0.5)
    plt.suptitle('Temporal evolution of band power (relative normalization)', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(outdir, "spectra.png")
    plt.savefig(fig_path, dpi=130); plt.close(fig)

    print(f"\n✅ Done. Saved:\n   - {csv_path}\n   - {fig_path}")
    print("📌 Validate results with independent experiments.")
    return df, {"dist_observed": dist_obs, "dist_baseline": dist_sim0,
                "dist_post": dist_post, "mod_delta": mod_delta,
                "fit_gap": fit_gap, "improvement": improvement}


def _sim_distance(asd_baseline, vec, pow_mean_td, seed, n_runs):
    """Simulated distance to TD for a given modulation vector (fixed seed => fair
    comparisons across candidates, since the noise realization is identical)."""
    _, pm, _, _ = model_normspec(build_model(asd_baseline, vec), seed=seed, n_runs=n_runs)
    return distance_to_td(pm, pow_mean_td)

def _greedy_search(asd_baseline, pow_mean_td, seed, n_runs=3, passes=3, include_tone=True):
    """Greedy coordinate search minimizing the SIMULATED distance to the TD reference.
    Baseline and every candidate are simulated from the SAME asd_baseline with the
    SAME fixed seed, so each comparison isolates the modulation effect (fair, paired).
    Returns (best_vec, basal_dist, best_dist, n_eval). Shared by the group optimizer
    and the per-subject optimizer so both use identical search logic and constants."""
    fixed_seed = seed + 5
    states_prot = [(-1.0, 'Inhibitor'), (0.0, 'None'), (+1.0, 'Agonist')]
    tone_levels = [-0.5, -0.25, 0.0, 0.25, 0.5]

    best_vec = np.zeros(N_MOD)
    basal_dist = _sim_distance(asd_baseline, best_vec, pow_mean_td, fixed_seed, n_runs)
    best_dist, n_eval = basal_dist, 0
    for _pass in range(passes):
        improved = False
        for i, p in enumerate(PROTEINS):
            for s, _nm in states_prot:
                vec = best_vec.copy()
                vec[i] = s * p['agonist_sign'] * p['base_frac']
                if np.allclose(vec, best_vec):
                    continue
                d = _sim_distance(asd_baseline, vec, pow_mean_td, fixed_seed, n_runs); n_eval += 1
                if d < best_dist - 1e-9:
                    best_dist, best_vec, improved = d, vec, True
        if include_tone:
            for j in range(len(NEUROMOD_KEYS)):
                for lvl in tone_levels:
                    vec = best_vec.copy()
                    vec[len(TARGET_KEYS) + j] = lvl
                    if np.allclose(vec, best_vec):
                        continue
                    d = _sim_distance(asd_baseline, vec, pow_mean_td, fixed_seed, n_runs); n_eval += 1
                    if d < best_dist - 1e-9:
                        best_dist, best_vec, improved = d, vec, True
        if not improved:
            break
    return best_vec, basal_dist, best_dist, n_eval

def _recommendation_df(best_vec):
    """Translate a winning modulation vector into a per-target recommendation table."""
    rows = {'Parameter': [], 'Δ (fraction)': [], 'Recommendation': []}
    for i, p in enumerate(PROTEINS):
        v = best_vec[i]
        rec = 'Neutral' if abs(v) <= 1e-9 else ('Agonist' if v * p['agonist_sign'] > 0 else 'Inhibitor')
        rows['Parameter'].append(p['name']); rows['Δ (fraction)'].append(f"{v:+.2f}")
        rows['Recommendation'].append(rec)
    for j, k in enumerate(NEUROMOD_KEYS):
        v = best_vec[len(TARGET_KEYS) + j]
        rec = 'Increase' if v > 1e-9 else ('Decrease' if v < -1e-9 else 'Neutral')
        rows['Parameter'].append(k.capitalize()); rows['Δ (fraction)'].append(f"{v:+.2f}")
        rows['Recommendation'].append(rec)
    return pd.DataFrame(rows)

def optimize_configuration(config, n_runs=3, passes=3, include_tone=True):
    """
    Estimate the modulation (agonist/inhibitor per target + neuromodulatory tone)
    that brings the (fitted) ASD profile closest to the TD reference, using the
    ACTUAL simulator (no surrogate approximation). Greedy coordinate search.

    IMPORTANT: 'best' = closest SIMULATED spectrum to the TD reference under the
    model's assumptions and the chosen drug->parameter map. This is a mechanistic
    HYPOTHESIS, not a validated therapeutic recommendation. A normalized-looking
    spectrum is not proof of clinical benefit. Greedy search finds a good (not
    guaranteed global) configuration.
    """
    # FIX B: per-subject mode -> one recommendation per ASD subject (see below).
    if config.get("per_subject"):
        return optimize_per_subject(config, n_runs=n_runs, passes=passes,
                                    include_tone=include_tone)

    outdir = config["output_dir"]; os.makedirs(outdir, exist_ok=True)
    seed = config["seed"]
    print("🎯 Estimating the best agonist/inhibitor GROUP configuration from the data")
    print("   (direct simulation; this can take ~1 minute).\n")

    asd_paths, td_paths = _resolve_input_paths(config)

    # ---- ASD baseline: fit the model to the (averaged) real data, else synthetic ----
    real_asd_pow_list = []
    if asd_paths:
        r = load_eeg_from_paths(asd_paths)
        if r[0] is not None:
            spec_asd, _, f, t, real_asd_pow_list, n = r
            if config["normalize"]:
                spec_asd = normalize_relative(spec_asd, None)
            pow_mean_asd = band_power_mean(spec_asd, f, t)
            (g_ei, g_ee, lf), rmse = fit_model_to_real([pow_mean_asd])
            asd_baseline = {'g_nmda_ei': g_ei, 'g_nmda_ee': g_ee, 'low_freq_amp': lf}
            print(f"   Fitted ASD baseline from {n} file(s): "
                  f"g_nmda_ei={g_ei:.3f}, g_nmda_ee={g_ee:.3f}, low_freq_amp={lf:.3f} "
                  f"(RMSE={rmse:.3f} dB)")
        else:
            asd_baseline = ASD_PARAMS
            print("   No readable ASD files; using the synthetic ASD baseline.")
    else:
        asd_baseline = ASD_PARAMS
        print("   No ASD files given; using the synthetic ASD baseline.")

    # ---- TD reference ----
    if td_paths:
        r = load_eeg_from_paths(td_paths)
        if r[0] is not None:
            spec_td = normalize_relative(r[0], None) if config["normalize"] else r[0]
            pow_mean_td = band_power_mean(spec_td, r[2], r[3])
            print(f"   TD reference from {r[5]} file(s).")
        else:
            _, pow_mean_td, _, _ = model_normspec(NeuralMass(**TD_PARAMS), seed=seed + 1, n_runs=n_runs)
    else:
        _, pow_mean_td, _, _ = model_normspec(NeuralMass(**TD_PARAMS), seed=seed + 1, n_runs=n_runs)

    # ---- greedy coordinate search + recommendation table (shared helpers) ----
    best_vec, basal_dist, best_dist, n_eval = _greedy_search(
        asd_baseline, pow_mean_td, seed, n_runs=n_runs, passes=passes,
        include_tone=include_tone)
    df = _recommendation_df(best_vec)

    improvement = basal_dist - best_dist
    pct = (improvement / basal_dist * 100) if basal_dist else 0.0
    print(f"\n   Evaluations: {n_eval} simulations")
    print(f"   Simulated distance to TD: basal={basal_dist:.2f} dB -> best={best_dist:.2f} dB "
          f"(Δ {improvement:+.2f} dB, {pct:+.1f}%)")
    print("\n📋 Recommended configuration (HYPOTHESIS):")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.to_string(index=False))
    csv = os.path.join(outdir, "recommended_config.csv"); df.to_csv(csv, index=False)
    print(f"\n✅ Saved -> {csv}")
    print("⚠️  Caveats: the result depends on the model, the ASD/TD baseline, and the")
    print("    drug->parameter map (which you should validate). 'Closer to the TD")
    print("    spectrum' is NOT proof of clinical benefit; it is a testable hypothesis.")
    if 0 < len(real_asd_pow_list) < 3:
        print(f"    (The significance test needs n>=3 real ASD samples; you gave "
              f"{len(real_asd_pow_list)}.)")
    return best_vec, df, {"basal": basal_dist, "best": best_dist}


def optimize_per_subject(config, n_runs=3, passes=3, include_tone=True):
    """
    Per-SUBJECT reliable recommendation (S5). For EACH ASD EEG file separately:
      1) load + standardize its spectrum (S1/S2, same as the group path),
      2) take its DATA-LEVEL spectral deviation from the TD reference,
      3) recommend the agonist/inhibitor (+ neuromodulator) moves that best cancel that
         deviation, via response projection onto the model's VALIDATED modulation
         responses (measured on stable reference baselines) with stability selection
         over a reference×seed ensemble; each move carries a confidence (selection freq).
    This deliberately AVOIDS per-subject model inversion (fit_model_to_real), which is
    ill-posed near the model's bifurcation and gives unstable baselines. Handles 1 file
    or many; recommendations may legitimately differ across subjects (ASD is
    heterogeneous). 'No robust move' = subject already near TD or deviation within noise.
    Each move is a testable HYPOTHESIS per subject, NOT clinical advice.
    """
    outdir = config["output_dir"]; os.makedirs(outdir, exist_ok=True)
    seed = config["seed"]
    print("🎯 Per-SUBJECT best-configuration search (one recommendation per ASD subject)")
    print("   (direct simulation per subject; this can take a while).\n")

    asd_paths, td_paths = _resolve_input_paths(config)
    files = _expand_paths(asd_paths)
    if not files:
        print("   ⚠️ No ASD files found. Per-subject mode needs at least one real EEG file;")
        print("      falling back to the (synthetic) group optimizer.\n")
        return optimize_configuration({**config, "per_subject": False},
                                      n_runs=n_runs, passes=passes, include_tone=include_tone)

    # ---- TD reference (identical construction to the group optimizer) ----
    if td_paths:
        r = load_eeg_from_paths(td_paths)
        if r[0] is not None:
            spec_td = normalize_relative(r[0], None) if config["normalize"] else r[0]
            pow_mean_td = band_power_mean(spec_td, r[2], r[3])
            print(f"   TD reference from {r[5]} real file(s).\n")
        else:
            _, pow_mean_td, _, _ = model_normspec(NeuralMass(**TD_PARAMS), seed=seed + 1, n_runs=n_runs)
            print("   No readable TD files; using the synthetic TD reference.\n")
    else:
        _, pow_mean_td, _, _ = model_normspec(NeuralMass(**TD_PARAMS), seed=seed + 1, n_runs=n_runs)
        print("   No TD files given; using the synthetic TD reference.\n")

    # ---- per-subject loop ----
    long_rows, summary_rows, mat_records = [], [], {}
    used_names = set()
    param_names = [p['name'] for p in PROTEINS] + [k.capitalize() for k in NEUROMOD_KEYS]
    for si, fp in enumerate(files):
        name = os.path.basename(fp)
        if name in used_names:                       # keep subjects distinct & comparable
            base, k = name, 2
            while f"{base}#{k}" in used_names:
                k += 1
            name = f"{base}#{k}"
        used_names.add(name)
        res = process_file_from_path(fp)
        if not res:
            print(f"   ⚠️ Skipping unreadable file: {name}\n"); continue
        spec_i, _, f_i, t_i = res
        if config["normalize"]:
            spec_i = normalize_relative(spec_i, None)
        pow_mean_i = band_power_mean(spec_i, f_i, t_i)
        subj_bands = np.array([pow_mean_i[b] for b in BANDS])
        td_bands = np.array([pow_mean_td[b] for b in BANDS])

        # S5: reliable recommendation by response projection + stability selection
        # (NO unstable per-subject model inversion; see recommend_by_projection).
        recs, ensemble = recommend_by_projection(subj_bands, td_bands)
        robust = [(tg, dr, fr) for (tg, dr, fr) in recs if fr >= REC_STABILITY_THRESHOLD]

        dev = td_bands - subj_bands                     # data-level deviation from TD
        dev_shape = dev - dev.mean()
        dev_l1 = float(np.sum(np.abs(dev_shape)))       # shape deviation (level removed)

        # comparable matrix column: signed direction for the robust moves, 0 otherwise
        row = {p['name']: 0.0 for p in PROTEINS}
        row.update({k.capitalize(): 0.0 for k in NEUROMOD_KEYS})
        for (tg, dr, fr) in robust:
            row[tg] = _MOVE_SIGN[(tg, dr)]
        mat_records[name] = row

        print(f"── Subject {si + 1}/{len(files)}: {name}")
        print(f"     spectral deviation from TD (shape-L1 = {dev_l1:.2f} dB) "
              f"[δθαβγ TD-subj: {np.round(dev, 2)}]")
        if robust:
            print(f"     robust recommendation(s) [stability over {ensemble} reference×seed runs]:")
            for (tg, dr, fr) in robust:
                print(f"        {tg:16s} {dr:9s}  confidence {fr * 100:3.0f}%")
        else:
            print("     no robust modulation — subject already close to TD, or its deviation "
                  "is within noise (informative: no confident recommendation).")
        print()

        for (tg, dr, fr) in recs:
            long_rows.append({'Subject': name, 'Target': tg, 'Direction': dr,
                              'Confidence_pct': f"{fr * 100:.0f}",
                              'Robust': 'yes' if fr >= REC_STABILITY_THRESHOLD else 'no'})
        rec_str = "; ".join(f"{tg} {dr} ({fr * 100:.0f}%)" for (tg, dr, fr) in robust) or "none robust"
        summary_rows.append({
            'Subject': name, 'dev_from_TD_shapeL1_dB': f"{dev_l1:.2f}",
            'robust_recommendation': rec_str})

    if not summary_rows:
        print("   ⚠️ No readable subjects; nothing to recommend.")
        return None, None, None

    df_long = pd.DataFrame(long_rows)
    df_summary = pd.DataFrame(summary_rows)
    # comparable matrix: rows = target/neuromodulator, columns = subject, cells = signed Δ
    df_matrix = pd.DataFrame(mat_records).reindex(param_names)
    df_matrix.index.name = "Parameter"

    csv_long   = os.path.join(outdir, "recommended_config_per_subject.csv")
    csv_sum    = os.path.join(outdir, "per_subject_summary.csv")
    csv_matrix = os.path.join(outdir, "per_subject_matrix.csv")
    df_long.to_csv(csv_long, index=False)
    df_summary.to_csv(csv_sum, index=False)
    df_matrix.to_csv(csv_matrix)

    print("📊 Comparable recommendation matrix (signed direction of ROBUST moves; "
          "rows=target, cols=subject; 0 = no robust move):")
    with pd.option_context('display.max_columns', None, 'display.width', 260):
        print(df_matrix.to_string())
    print("\n📋 Per-subject deviation from TD + robust recommendation "
          "(same TD reference; response-projection with stability selection):")
    with pd.option_context('display.max_columns', None, 'display.width', 260):
        print(df_summary.to_string(index=False))
    print(f"\n✅ Saved -> {csv_matrix}   (comparable matrix: target × subject)")
    print(f"✅ Saved -> {csv_sum}   (per-subject deviation + robust recommendation)")
    print(f"✅ Saved -> {csv_long}   (tidy long format: subject × move × confidence)")
    print("ℹ️  Recommendations come from projecting each subject's spectral deviation from TD")
    print("    onto the model's VALIDATED modulation responses (measured on stable reference")
    print("    baselines), kept only if selected across the reference×seed ensemble. The")
    print("    confidence % is that selection frequency. No unstable per-subject model")
    print("    inversion is used.")
    print("⚠️  Each move is a first-order, testable HYPOTHESIS per subject, NOT clinical")
    print("    advice. 'No robust move' means the subject is already near TD or its deviation")
    print("    is within noise. ASD is heterogeneous: recommendations may differ (expected).")
    return df_summary, df_matrix, df_long


def train_surrogate(config):
    outdir = config["output_dir"]; os.makedirs(outdir, exist_ok=True)
    print("🤖 Generating synthetic data and training the surrogate (offline)...")
    print("   (synthetic data from the simulator; the surrogate is a fast emulator).")
    X_spec, X_mod, y_dist, y_bands, f, t = generate_training_data(
        n_samples=config["train_samples"], seed=config["seed"],
        out=os.path.join(outdir, "training_data.npz"))
    (Xs_tr, Xs_va, Xm_tr, Xm_va, yd_tr, yd_va, yb_tr, yb_va) = train_test_split(
        X_spec, X_mod, y_dist, y_bands, test_size=0.2, random_state=42)
    ai = NeuronAI()
    ai.build_arch(X_spec.shape[1:], X_mod.shape[1])
    model_out = os.path.join(outdir, "neuron_ai_model.h5")
    hist = ai.train(Xs_tr, Xm_tr, yd_tr, yb_tr, epochs=config["train_epochs"],
                    model_out=model_out)
    fid = ai.evaluate_fidelity(Xs_va, Xm_va, yd_va)
    print(f"\n✅ Trained. Model -> {model_out}")
    print(f"📏 Surrogate fidelity to the simulator (hold-out): "
          f"MAE={fid['mae_db']:.3f} dB | MAPE={fid['mape_pct']:.2f}% | R²={fid['r2']:.4f}")
    print("   (Fidelity to the simulator within the training distribution, NOT")
    print("    predictive validity on real clinical data.)")
    # loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title('Training loss'); plt.legend(); plt.tight_layout()
    p = os.path.join(outdir, "training_loss.png"); plt.savefig(p, dpi=130); plt.close()
    print(f"   loss curve -> {p}")
    return ai, fid


# %% ------------------------------------------------------------------
# 11. Entry point
# --------------------------------------------------------------------
# ---- Aperiodic (1/f) E/I axis: per-subject discrimination + directional hypothesis ----
# The aperiodic exponent (FOOOF) is a validated, robust E/I proxy (Gao, Peterson &
# Voytek 2017) that separates subjects where broadband 5-band powers cannot. We (1)
# place each subject on the E/I axis vs TD with an uncertainty, (2) give the pharmacological
# DIRECTION from the known E/I sign of each target (not the rate model's exponent
# response, which is faithful for glutamatergic but NOT GABAergic targets), and (3) flag,
# transparently, which candidate moves THIS model reproduces on the exponent.
EI_AGONIST_SIGN = {'GABA_A': -1, 'GABA_B': -1, 'NMDA (NR2A)': +1, 'NMDA (NR2B)': -1,
                   'AMPA': +1, 'alpha7_nAChR': -1}   # +1: agonist raises E/I; -1: raises inhibition

def _lazy_fooof():
    try:
        from fooof import FOOOF
        return FOOOF
    except Exception:
        return None

def _load_signal_1d(fp):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw(fp, preload=True, verbose="ERROR")
    raw.resample(FS); raw.filter(0.5, 80, fir_design="firwin", verbose=False)
    raw.notch_filter(50, verbose=False)
    av = [c for c in ['Cz', 'Fz', 'Pz', 'C3', 'C4'] if c in raw.ch_names]
    if av:
        raw.pick(av)
    sig = raw.get_data().mean(axis=0); need = int(DURATION * FS)
    sig = np.pad(sig, (0, need - len(sig))) if len(sig) < need else sig[:need]
    return (sig - sig.mean()) / (sig.std() + 1e-12)

def _exp_of_signal(sig, FOOOF, n_windows=4):
    from scipy.signal import welch
    def one(x):
        f, p = welch(x, fs=FS, nperseg=FS * 2)
        fm = FOOOF(max_n_peaks=6, aperiodic_mode='fixed', verbose=False)
        fm.fit(f, p, [2, 40]); return fm.aperiodic_params_[-1], fm.r_squared_
    e_full, r2 = one(sig)
    w = len(sig) // n_windows
    es = [one(sig[i * w:(i + 1) * w])[0] for i in range(n_windows) if (i + 1) * w <= len(sig) and w > FS * 2]
    sd = float(np.std(es)) if len(es) >= 2 else 0.05
    return float(e_full), sd, float(r2)

_MODEL_EXP_RESP = None
def _model_exponent_responses(FOOOF, baseline=None, seeds=(1, 2, 3), n_runs=20):
    """Δexponent of the model for each single agonist/inhibitor move (transparency check:
    does THIS model reproduce the move's effect on the 1/f slope?)."""
    global _MODEL_EXP_RESP
    if _MODEL_EXP_RESP is not None:
        return _MODEL_EXP_RESP
    baseline = baseline or STABLE_REF_BASELINES[0]
    def mexp(vec, sd):
        sp, f, t = build_model(baseline, vec).averaged_spectrogram_db(seed=sd, n_runs=n_runs)
        psd = (10 ** (sp / 10.0)).mean(axis=1)
        fm = FOOOF(max_n_peaks=6, aperiodic_mode='fixed', verbose=False)
        try:
            fm.fit(f, psd, [2, 40]); return fm.aperiodic_params_[-1]
        except Exception:
            return np.nan
    base = np.mean([mexp(np.zeros(len(MOD_ORDER)), s) for s in seeds])
    resp = {}
    for i, p in enumerate(PROTEINS):
        for act, dirn in ((+1.0, 'Agonist'), (-1.0, 'Inhibitor')):
            vec = np.zeros(len(MOD_ORDER)); vec[i] = act * p['agonist_sign'] * p['base_frac']
            resp[(p['name'], dirn)] = float(np.mean([mexp(vec, s) for s in seeds]) - base)
    _MODEL_EXP_RESP = resp
    return resp

def _candidate_moves(direction):
    """direction=+1 -> raise E/I (subject too inhibited); -1 -> lower E/I (too excited)."""
    return [(tgt, 'Agonist' if sign == direction else 'Inhibitor')
            for tgt, sign in EI_AGONIST_SIGN.items()]

def analyze_aperiodic(config):
    FOOOF = _lazy_fooof()
    if FOOOF is None:
        print("⚠️ This mode needs FOOOF: pip install fooof   (or specparam). Aborting."); return
    outdir = config["output_dir"]; os.makedirs(outdir, exist_ok=True)
    print("🧭 Aperiodic-exponent E/I analysis (per-subject discrimination + directional")
    print("   pharmacological hypothesis; robust 1/f marker, Gao et al. 2017).\n")
    asd_paths, td_paths = _resolve_input_paths(config)
    files = _expand_paths(asd_paths)
    if not files:
        print("   ⚠️ No ASD files. This mode needs at least one real EEG file."); return

    td_exp, td_sd = None, 0.05
    if td_paths:
        evals = []
        for fp in _expand_paths(td_paths):
            try:
                evals.append(_exp_of_signal(_load_signal_1d(fp), FOOOF)[0])
            except Exception:
                pass
        if evals:
            td_exp = float(np.mean(evals)); td_sd = float(np.std(evals)) if len(evals) > 1 else 0.05
            print(f"   TD reference exponent = {td_exp:.3f} (±{td_sd:.3f}, n={len(evals)})\n")
    if td_exp is None:
        print("   ⚠️ No TD files; cannot place subjects relative to neurotypical. Aborting."); return

    model_resp = _model_exponent_responses(FOOOF)
    rows = []
    for si, fp in enumerate(files):
        name = os.path.basename(fp)
        try:
            e, sd, r2 = _exp_of_signal(_load_signal_1d(fp), FOOOF)
        except Exception as ex:
            print(f"── {name}: unreadable ({ex})\n"); continue
        d = e - td_exp; unc = float(np.hypot(sd, td_sd))
        significant = abs(d) > 2 * max(unc, 0.02)
        print(f"── Subject {si + 1}/{len(files)}: {name}")
        print(f"     aperiodic exponent = {e:.3f} (±{sd:.3f}, R²={r2:.2f}) | "
              f"Δ vs TD = {d:+.3f} (±{unc:.3f})")
        if not significant:
            print("     E/I within neurotypical range (Δ not beyond 2×uncertainty): "
                  "no directional hypothesis.\n")
            rows.append({'Subject': name, 'exponent': f"{e:.3f}", 'delta_vs_TD': f"{d:+.3f}",
                         'significant': 'no', 'EI': 'typical', 'hypothesis': 'none',
                         'candidate_targets': ''})
            continue
        direction = +1 if d > 0 else -1
        ei = "LOWER than TD (steeper 1/f → more inhibited)" if d > 0 else \
             "HIGHER than TD (flatter 1/f → more excited)"
        goal = "increase excitation / disinhibit" if d > 0 else "increase inhibition / reduce excitation"
        print(f"     E/I {ei}")
        print(f"     directional hypothesis: {goal}. Candidate targets "
              f"(★ = also reproduced by this model on the 1/f slope):")
        cands = []
        for tgt, move in _candidate_moves(direction):
            mr = model_resp.get((tgt, move), np.nan)
            consistent = (not np.isnan(mr)) and ((mr < 0) == (d > 0)) and abs(mr) > 0.03
            print(f"        {'★' if consistent else ' '} {tgt:14s} {move:9s}   (model Δexp={mr:+.2f})")
            cands.append(f"{tgt}:{move}{'*' if consistent else ''}")
        print()
        rows.append({'Subject': name, 'exponent': f"{e:.3f}", 'delta_vs_TD': f"{d:+.3f}",
                     'significant': 'yes', 'EI': ('lower' if d > 0 else 'higher'),
                     'hypothesis': goal, 'candidate_targets': "; ".join(cands)})

    df = pd.DataFrame(rows)
    csv = os.path.join(outdir, "aperiodic_EI_per_subject.csv"); df.to_csv(csv, index=False)
    print(f"✅ Saved -> {csv}")
    print("ℹ️  E/I placement uses the aperiodic exponent (validated E/I proxy). Direction")
    print("    comes from each target's known E/I pharmacology; ★ marks moves this rate")
    print("    model also reproduces on the slope (glutamatergic tend to; GABAergic often")
    print("    do NOT — a documented limitation of rate models for the 1/f proxy).")
    print("⚠️  Per-subject hypotheses, NOT clinical advice. Small n => illustrative.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Neuron ASD v1.3 (VSCode edition)")
    parser.add_argument("--mode",
                        choices=["simulate", "optimize", "train", "selftest",
                                 "pharmacheck", "mapping", "aperiodic"],
                        default="simulate",
                        help="simulate a given config; optimize (estimate the best "
                             "agonist/inhibitor config from data; per-subject by default); "
                             "train the surrogate; selftest (invariants); pharmacheck "
                             "(construct-validity vs literature); mapping (print the "
                             "drug->parameter map + refs). Default: simulate")
    parser.add_argument("--pick-files", action="store_true",
                        help="open a pop-up window to choose the EEG files")
    parser.add_argument("--no-selftest", action="store_true",
                        help="skip the invariants check before simulating/optimizing")
    parser.add_argument("--per-subject", action="store_true",
                        help="optimize: one recommendation per ASD subject (default)")
    parser.add_argument("--group", action="store_true",
                        help="optimize: a single group-averaged recommendation instead")
    args = parser.parse_args()

    if args.pick_files:
        CONFIG["ask_files"] = True
    if args.per_subject:
        CONFIG["per_subject"] = True
    if args.group:
        CONFIG["per_subject"] = False

    if args.mode == "selftest":
        verify_invariants(); return
    if args.mode == "mapping":
        print_mapping(); return
    if args.mode == "pharmacheck":
        check_pharmaco_eeg_signatures(); return
    if args.mode == "aperiodic":
        if not CONFIG.get("asd_paths") and not CONFIG.get("ask_files"):
            print("ℹ️ No EEG paths in CONFIG — opening the file chooser so you can load "
                  "your EEGs (ASD, then TD reference).\n")
            CONFIG["ask_files"] = True
        analyze_aperiodic(CONFIG); return
    if args.mode == "train":
        train_surrogate(CONFIG); return
    if args.mode == "optimize":
        # Optimizing (per-subject or group) needs real EEG. If no paths are set in
        # CONFIG and none were requested, offer the file window automatically so the
        # tool ASKS for the EEGs (accepts many formats and a .zip of 2+ recordings).
        if not CONFIG.get("asd_paths") and not CONFIG.get("ask_files"):
            print("ℹ️ No EEG paths in CONFIG — opening the file chooser so you can load "
                  "your EEGs (many formats, or a .zip of 2+). Cancel = synthetic demo.\n")
            CONFIG["ask_files"] = True
        if not args.no_selftest:
            verify_invariants(); print()
        optimize_configuration(CONFIG); return

    # default: simulate (with a quick self-test first, unless disabled)
    if not args.no_selftest:
        verify_invariants(); print()
    run_simulation(CONFIG)


if __name__ == "__main__":
    main()
