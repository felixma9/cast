#!/usr/bin/env python
"""
Synthetic validation for the B/A proxy algorithm in pysidecaster.py.

Tests the core algorithm against RF signals with analytically known
frequency content, so ground-truth E2/E1 ratios can be computed exactly.

Requirements: numpy only.
No probe, no PySide6, no pyclariuscast required.

If the algorithm in pysidecaster.py is changed, update compute_ba_proxy()
below to match before re-running these tests.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Algorithm -- copied verbatim from newRawImage() in pysidecaster.py.
# Update this if the production algorithm changes.
# ---------------------------------------------------------------------------

def compute_ba_proxy(arr):
    """
    Run the B/A proxy computation on a (lines, samples) float32 array.

    Returns:
        coarse_map : (lines, num_windows) float32  -- raw E2/E1 ratios
        k1         : (lines, num_windows) int      -- detected fundamental bin
        E1         : (lines, num_windows) float    -- fundamental neighbourhood energy
        E2         : (lines, num_windows) float    -- harmonic neighbourhood energy
    """
    lines, samples = arr.shape
    win_len = 64
    stride  = 16
    eps     = 1e-6

    num_windows = max(1, (samples - win_len) // stride + 1)

    window_starts  = np.arange(num_windows, dtype=np.int32) * stride
    window_indices = window_starts[:, None] + np.arange(win_len, dtype=np.int32)[None, :]

    windows = arr[:, window_indices]                              # (lines, num_windows, win_len)
    windows = windows - windows.mean(axis=2, keepdims=True)      # remove DC

    power    = np.abs(np.fft.rfft(windows, axis=2)) ** 2         # (lines, num_windows, nfft_bins)
    nfft_bins = win_len // 2 + 1
    last_bin  = nfft_bins - 1

    upper = min(max(2, win_len // 8), last_bin)
    k1 = np.argmax(power[:, :, 1:upper + 1], axis=2) + 1
    k2 = np.minimum(2 * k1, last_bin)

    L_idx = np.arange(lines)[:, None]
    W_idx = np.arange(num_windows)[None, :]

    def gather_energy(kk):
        kk_lo = np.clip(kk - 1, 0, last_bin)
        kk_hi = np.clip(kk + 1, 0, last_bin)
        return power[L_idx, W_idx, kk_lo] + power[L_idx, W_idx, kk] + power[L_idx, W_idx, kk_hi]

    E1 = gather_energy(k1)
    E2 = gather_energy(k2)

    coarse_map = (E2 / (E1 + eps)).astype(np.float32)
    coarse_map = np.where(np.isfinite(coarse_map), coarse_map, 0.0)

    return coarse_map, k1, E1, E2


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

WIN_LEN = 64  # must match win_len in compute_ba_proxy


def make_rf(lines, samples, fundamental_bin, harmonic_amplitude=0.0,
            fundamental_amplitude=1.0):
    """
    Build a (lines, samples) float32 RF array containing a pure sinusoid at
    fundamental_bin plus optionally its second harmonic.

    fundamental_bin must be an integer in [1, WIN_LEN//8] so the algorithm
    will land k1 on it.  Using exact FFT-bin frequencies eliminates spectral
    leakage, making the analytical ground truth exact.

    Frequency of fundamental: fundamental_bin / WIN_LEN  cycles per sample.
    """
    t  = np.arange(samples, dtype=np.float64)
    f1 = fundamental_bin / WIN_LEN
    f2 = 2.0 * f1

    signal = fundamental_amplitude * np.cos(2.0 * np.pi * f1 * t)
    if harmonic_amplitude > 0.0:
        signal += harmonic_amplitude * np.cos(2.0 * np.pi * f2 * t)

    return np.tile(signal.astype(np.float32), (lines, 1))


def analytical_ratio(fundamental_amplitude, harmonic_amplitude):
    """
    Ground-truth E2/E1 for exact-bin signals (no leakage).

    For FFT bin k of a real cosine with amplitude A over N samples:
        |rfft[k]|^2 = (A * N/2)^2

    With no energy at neighbouring bins, the 3-bin energy sum equals
    the single-bin power.  Therefore:
        E1 = (A * N/2)^2,  E2 = (h * N/2)^2
        ratio = E2/E1 = (h/A)^2
    """
    return (harmonic_amplitude / fundamental_amplitude) ** 2


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"


def check_near(label, actual, expected, tol_pct):
    """Pass if |actual - expected| <= |expected| * tol_pct."""
    ok = abs(actual - expected) <= abs(expected) * tol_pct
    status = PASS if ok else FAIL
    print(f"    [{status}] {label}: "
          f"actual={actual:.6f}, expected={expected:.6f}, tol=+/-{tol_pct*100:.0f}%")
    return ok


def check_below(label, actual, threshold):
    """Pass if actual < threshold."""
    ok = actual < threshold
    status = PASS if ok else FAIL
    print(f"    [{status}] {label}: actual={actual:.2e}, threshold={threshold:.2e}")
    return ok


def check_equal(label, actual, expected):
    """Pass if actual == expected (integer comparison)."""
    ok = actual == expected
    status = PASS if ok else FAIL
    print(f"    [{status}] {label}: actual={actual}, expected={expected}")
    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pure_fundamental():
    """
    A pure sinusoid contains no harmonic energy.

    E2/E1 should be essentially zero -- limited only by floating-point
    noise.  Also confirms k1 lands on the correct fundamental bin.
    """
    print("\n=== Test 1: Pure fundamental -- no harmonic energy ===")
    LINES, SAMPLES, F_BIN = 4, 512, 4

    arr = make_rf(LINES, SAMPLES, fundamental_bin=F_BIN)
    coarse_map, k1, E1, E2 = compute_ba_proxy(arr)

    max_ratio = float(coarse_map.max())
    k1_mode   = int(np.bincount(k1.ravel()).argmax())

    print(f"  max ratio across all windows : {max_ratio:.2e}  (expect ~ 0)")
    print(f"  dominant k1 bin detected     : {k1_mode}  (expect {F_BIN})")

    passed  = check_below("max E2/E1 ratio near zero", max_ratio, 1e-3)
    passed &= check_equal("k1 identifies correct fundamental bin", k1_mode, F_BIN)
    return passed


def test_known_harmonic_ratios():
    """
    Fundamental + 2nd harmonic at three known amplitude ratios.

    For amplitude ratio h/A, the expected proxy ratio is (h/A)^2.
    Tests that computed values match this analytical expectation within 2%.
    """
    print("\n=== Test 2: Known harmonic amplitude ratios ===")
    LINES, SAMPLES, F_BIN = 4, 512, 4

    cases = [
        ("h/A = 0.10", 1.0, 0.10),
        ("h/A = 0.30", 1.0, 0.30),
        ("h/A = 0.50", 1.0, 0.50),
    ]

    passed = True
    for label, A, h in cases:
        arr = make_rf(LINES, SAMPLES, fundamental_bin=F_BIN,
                      fundamental_amplitude=A, harmonic_amplitude=h)
        coarse_map, k1, _, _ = compute_ba_proxy(arr)

        mean_ratio = float(coarse_map.mean())
        expected   = analytical_ratio(A, h)
        k1_mode    = int(np.bincount(k1.ravel()).argmax())

        print(f"\n  {label}")
        passed &= check_near("  E2/E1 ratio", mean_ratio, expected, tol_pct=0.02)
        passed &= check_equal("  k1 bin", k1_mode, F_BIN)

    return passed


def test_spatial_differentiation():
    """
    Left half of scan lines: pure fundamental  (low nonlinearity).
    Right half of scan lines: fundamental + harmonic (higher nonlinearity).

    The algorithm must produce clearly different mean ratios for each half,
    matching the known analytical values for each half.
    """
    print("\n=== Test 3: Spatial differentiation -- left low, right high ===")
    LINES, SAMPLES, F_BIN = 8, 512, 4
    H_AMPLITUDE = 0.4  # expected right-half ratio: 0.4^2 = 0.16

    arr = np.zeros((LINES, SAMPLES), dtype=np.float32)
    arr[:LINES // 2] = make_rf(LINES // 2, SAMPLES, F_BIN, harmonic_amplitude=0.0)
    arr[LINES // 2:] = make_rf(LINES // 2, SAMPLES, F_BIN, harmonic_amplitude=H_AMPLITUDE)

    coarse_map, _, _, _ = compute_ba_proxy(arr)

    mean_left  = float(coarse_map[:LINES // 2].mean())
    mean_right = float(coarse_map[LINES // 2:].mean())
    expected_right = analytical_ratio(1.0, H_AMPLITUDE)

    print(f"  Left  half mean ratio : {mean_left:.6f}  (expect ~ 0)")
    print(f"  Right half mean ratio : {mean_right:.6f}  (expect ~ {expected_right:.4f})")

    passed  = check_below("left half near zero", mean_left, 1e-3)
    passed &= check_near("right half matches expected ratio", mean_right,
                         expected_right, tol_pct=0.02)
    # Separation: right should be orders of magnitude above left
    separation = mean_right / max(mean_left, 1e-12)
    passed &= check_below("right/left separation > 100x",
                           1.0 / separation, 1.0 / 100)
    return passed


def test_amplitude_invariance():
    """
    E2/E1 is a ratio, so scaling both fundamental and harmonic amplitudes
    by the same factor must leave the output unchanged.

    Tests amplitudes spanning four orders of magnitude (0.1 to 1000).
    """
    print("\n=== Test 4: Amplitude invariance ===")
    LINES, SAMPLES, F_BIN = 4, 512, 4
    H_FRACTION = 0.3  # harmonic = 30% of fundamental at every scale

    ratios = []
    for scale in [0.1, 1.0, 10.0, 1000.0]:
        arr = make_rf(LINES, SAMPLES, F_BIN,
                      fundamental_amplitude=scale,
                      harmonic_amplitude=scale * H_FRACTION)
        coarse_map, _, _, _ = compute_ba_proxy(arr)
        ratios.append(float(coarse_map.mean()))
        print(f"  scale={scale:7.1f} -> mean ratio = {ratios[-1]:.6f}")

    ratio_spread = max(ratios) - min(ratios)
    reference    = ratios[0]
    passed = check_below("spread across all scales < 1% of reference",
                         ratio_spread, reference * 0.01)
    return passed


def test_dc_invariance():
    """
    Adding a constant DC offset to the signal must not change the ratio,
    because DC is subtracted before the FFT.
    """
    print("\n=== Test 5: DC offset invariance ===")
    LINES, SAMPLES, F_BIN = 4, 512, 4
    H_AMPLITUDE = 0.3

    baseline = make_rf(LINES, SAMPLES, F_BIN, harmonic_amplitude=H_AMPLITUDE)
    with_dc   = (baseline + 500.0).astype(np.float32)

    map_baseline, _, _, _ = compute_ba_proxy(baseline)
    map_dc,       _, _, _ = compute_ba_proxy(with_dc)

    mean_baseline = float(map_baseline.mean())
    mean_dc       = float(map_dc.mean())

    print(f"  Without DC offset : {mean_baseline:.6f}")
    print(f"  With DC +500      : {mean_dc:.6f}")

    passed = check_near("DC offset does not change ratio",
                        mean_dc, mean_baseline, tol_pct=0.001)
    return passed


def test_depth_gradient():
    """
    Simulates harmonic buildup with propagation depth.

    Physical basis: as an ultrasound wave travels deeper into tissue, it
    accumulates nonlinear distortion, progressively transferring energy from
    the fundamental into the second harmonic.  Even in uniform tissue this
    produces a systematic depth-dependent trend in E2/E1.

    Signal construction:
        signal[t] = A*cos(2*pi*f1*t) + h(t)*cos(2*pi*2*f1*t)
        h(t) = H_MAX * t / (SAMPLES - 1)    -- zero at t=0, H_MAX at t=end

    Expected per-window ratio for window wi (center at t_c = wi*stride + win_len/2):
        h_eff(wi) = H_MAX * t_c / (SAMPLES - 1)
        ratio(wi) ~ (h_eff(wi) / A)^2

    Because h(t) varies within each 64-sample window (not constant), the FFT
    sees a blend rather than a pure tone.  Tolerances are therefore looser
    than for the stationary-signal tests.

    Checks:
        1. Strong positive correlation (r > 0.95) between window depth and ratio.
        2. Shallow-end windows (first 10%) have ratio near zero.
        3. Deep-end windows (last 10%) have ratio within 30% of (H_MAX/A)^2.
    """
    print("\n=== Test 6: Depth gradient -- harmonic buildup with depth ===")

    LINES   = 8
    SAMPLES = 1024
    F_BIN   = 4
    H_MAX   = 0.4   # harmonic amplitude at maximum depth; expected peak ratio = 0.16
    A       = 1.0
    STRIDE  = 16    # must match compute_ba_proxy

    t          = np.arange(SAMPLES, dtype=np.float64)
    f1         = F_BIN / WIN_LEN
    h_envelope = H_MAX * t / (SAMPLES - 1)

    signal = (A * np.cos(2.0 * np.pi * f1 * t)
              + h_envelope * np.cos(2.0 * np.pi * 2.0 * f1 * t)).astype(np.float32)
    arr = np.tile(signal, (LINES, 1))

    coarse_map, k1, _, _ = compute_ba_proxy(arr)

    num_windows = coarse_map.shape[1]

    # Average over lines to get one ratio value per depth window
    per_window = coarse_map.mean(axis=0)   # shape (num_windows,)

    # Expected ratio at each window center
    win_centers = np.array([wi * STRIDE + WIN_LEN // 2
                             for wi in range(num_windows)], dtype=np.float64)
    h_eff    = H_MAX * win_centers / (SAMPLES - 1)
    expected = (h_eff / A) ** 2            # shape (num_windows,)

    # Pearson correlation between window index and measured ratio
    idx         = np.arange(num_windows, dtype=np.float64)
    correlation = float(np.corrcoef(idx, per_window)[0, 1])

    # Shallow-end mean (first 10% of windows)
    n_tail       = max(1, num_windows // 10)
    mean_shallow = float(per_window[:n_tail].mean())
    mean_deep    = float(per_window[-n_tail:].mean())
    expected_deep = float(expected[-n_tail:].mean())

    # Print a depth profile sampled at ~10 evenly spaced windows
    print(f"  {num_windows} depth windows  |  "
          f"shallow mean={mean_shallow:.4f}  |  deep mean={mean_deep:.4f}  |  "
          f"r={correlation:.4f}")
    step = max(1, num_windows // 10)
    print(f"  {'window':>7}  {'measured':>10}  {'expected':>10}")
    for wi in range(0, num_windows, step):
        print(f"  {wi:7d}  {per_window[wi]:10.5f}  {expected[wi]:10.5f}")

    passed  = check_below("Pearson r > 0.95 (1/r check)",
                          1.0 / correlation, 1.0 / 0.95)
    passed &= check_below("shallow-end ratio near zero", mean_shallow, 5e-3)
    passed &= check_near("deep-end ratio near expected",
                         mean_deep, expected_deep, tol_pct=0.30)
    return passed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("B/A Proxy -- Synthetic Validation")
    print("=" * 52)
    print("Algorithm: win_len=64, stride=16")
    print("Signals:   exact-bin sinusoids (zero spectral leakage)")
    print("Tolerance: +/-2% on ratio values unless stated otherwise")

    results = {
        "Test 1 -- Pure fundamental":        test_pure_fundamental(),
        "Test 2 -- Known harmonic ratios":   test_known_harmonic_ratios(),
        "Test 3 -- Spatial differentiation": test_spatial_differentiation(),
        "Test 4 -- Amplitude invariance":    test_amplitude_invariance(),
        "Test 5 -- DC offset invariance":    test_dc_invariance(),
        "Test 6 -- Depth gradient":          test_depth_gradient(),
    }

    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    all_passed = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  [{status}] {name}")
        all_passed &= passed

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("One or more tests FAILED -- review output above.")
