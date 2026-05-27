"""EDA preprocessing: decompose into tonic (SCL) and phasic (SCR) components.

Pipeline:
  1. Lowpass 3 Hz  → remove high-frequency noise (full signal)
  2. Highpass 0.05 Hz → extract phasic (SCR) component
  3. Tonic (SCL) = lowpass_signal - phasic
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _sos(cutoff, btype, fs, order=4):
    nyq = fs / 2
    return butter(order, cutoff / nyq, btype=btype, output="sos")


def preprocess(signal: np.ndarray, fs: int = 128) -> dict[str, np.ndarray]:
    """Return dict with keys 'full', 'tonic', 'phasic'.

    Args:
        signal: (n_samples,) EDA signal
        fs: sampling frequency in Hz
    """
    signal = signal.astype(np.float32).ravel()

    lp_sos = _sos(3.0, "low", fs)
    hp_sos = _sos(0.05, "high", fs)

    full = sosfiltfilt(lp_sos, signal)
    phasic = sosfiltfilt(hp_sos, full)
    tonic = full - phasic

    return {
        "full": full.astype(np.float32),
        "tonic": tonic.astype(np.float32),
        "phasic": phasic.astype(np.float32),
    }
