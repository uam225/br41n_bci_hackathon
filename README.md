# Motor Imagery Decoding in Chronic Stroke EEG via Riemannian Geometry

**BR41N.IO Brain-Computer Interface Designers Hackathon (May 2025)**  
**Team G95:** Umair Arshad

---

## Overview

Decoding Left vs Right hand Motor Imagery (MI) from cortical EEG in chronic stroke patients is challenging due to lesion-induced cortical reorganisation, non-stationary signal statistics, and low signal-to-noise ratio. At the same time, tracking how classification separability and sensorimotor band power evolve across **Brain-Computer Interface Functional Electrical Stimulation (BCI-FES)** rehabilitation sessions provides a direct electrophysiological marker of motor recovery.

This project systematically benchmarks spatial filtering and manifold-geometry decoding pipelines on 16-channel stroke patient EEG recorded **before (`Pre`)** and **after (`Post`)** a 25-session BCI-FES rehabilitation programme:

- **Key Result:** Isolating the **Beta rhythm (13 to 30 Hz)** during the active task window (2 to 6 s) and projecting OAS-regularised trial covariance matrices onto the **Riemannian Tangent Space** with Logistic Regression (`Riemannian(LR)_13-30Hz`) achieves an average decoding accuracy of **89.58%** across all patient-session splits (reaching **96.25%** on `P1 Pre`, **100.00%** on `P2 Post`, and **96.25%** on `P3 Post`), outperforming standard `CSP+LDA` (**63.75%** on raw epochs; **70.17%** best configuration) by **+19.4 percentage points**.
- **Neuroplasticity & Therapy Effect:** In responsive patients (`P2` and `P3`), post-therapy decoding accuracy increased by **+20.0 percentage points** (`P2`: $80.00\% \rightarrow 100.00\%$; `P3`: $76.25\% \rightarrow 96.25\%$), accompanied by clearer t-SNE tangent-space cluster separation and focal Beta-band event-related desynchronisation/synchronisation (ERD/ERS) shifts over sensorimotor electrodes (`C3`, `Cz`, `C4`).

---

## Dataset & Experimental Paradigm

The dataset derives from the clinical stroke rehabilitation study by Sebastián-Romagosa et al.:

- **Cohort:** 3 chronic stroke patients (`P1`, `P2`, `P3`), each evaluated in two sessions: **Pre-Therapy** (before Session 1) and **Post-Therapy** (after completing 25 BCI-FES rehabilitation sessions).
- **Recording Setup:** 16-channel active EEG (`FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, Pz`, standard 10-20 placement over the bilateral sensorimotor strip) sampled at $f_s = 256\text{ Hz}$.
- **Trial Structure:** Each session comprises separate **Training** (80 trials) and **Testing** (80 unseen trials) runs of cued Left-Hand (`+1`) vs Right-Hand (`-1`) motor imagery (0 s attention cue, 2 s directional arrow, 2 to 8 s active motor imagery + FES feedback).

---

## Decoding Methodology

### 1. Preprocessing & Covariance Windowing (`phs_1.*.py`)

1. **Epoch Extraction (`phs_1.4_load_and_extract_epochs.py`):** Continuous recordings are segmented into trials aligned to trigger onsets, with optional baseline time-domain ZCA whitening (`phs_1.3_time_whitening.py`).
2. **Spectral Filtering:** 5th-order zero-phase Butterworth bandpass filters (`scipy.signal.filtfilt`) isolate the sensorimotor **Mu (8 to 12 Hz)** and **Beta (13 to 30 Hz)** rhythms.
3. **Active Task Window:** Covariance estimation is restricted to $t \in [2.0, 6.0]\text{ s}$ within each epoch ($512$ to $1536$ samples), capturing sustained motor imagery while excluding initial visual-evoked potentials from the directional cue.

### 2. Riemannian Geometry Tangent-Space Pipeline (`phs_2.3*.py`)

Rather than learning spatial filters that can overfit small clinical calibration sets, each filtered trial $X_i \in \mathbb{R}^{C \times N_t}$ ($C = 16$ channels) is represented by its regularised spatial covariance matrix on the Symmetric Positive Definite ($\mathcal{SPD}_C$) manifold:

1. **Oracle Approximating Shrinkage (OAS) Covariance Estimation:** Ensures strict positive-definiteness and well-conditioned eigenvalues even on short time windows ($\Sigma_i \in \mathcal{SPD}_{16}$).
2. **Riemannian Fréchet Mean:** Computes the geometric mean covariance $\bar{\Sigma}$ across training trials under the affine-invariant Riemannian metric:

$$\delta_R(\Sigma_1, \Sigma_2) = \left\| \log\!\left(\Sigma_1^{-1/2} \Sigma_2 \Sigma_1^{-1/2}\right) \right\|_F$$

3. **Tangent-Space Projection:** Maps each trial covariance matrix $\Sigma_i$ onto the Euclidean tangent space at $\bar{\Sigma}$ via the logarithmic map, extracting the upper-triangular vector $s_i \in \mathbb{R}^{C(C+1)/2} = \mathbb{R}^{136}$:

$$S_i = \text{upper}\!\left(\log\!\left(\bar{\Sigma}^{-1/2} \Sigma_i \bar{\Sigma}^{-1/2}\right)\right)$$

4. **Linear Classification:** An $\ell_2$-regularised Logistic Regression classifier (`liblinear`) is trained on the tangent vectors $s_i$ (or stacked $\mu + \beta$ tangent vectors $s_i \in \mathbb{R}^{272}$ in `phs_2.3b_riemannian.py`).

---

## Benchmark Results Across Pipelines

Unseen test-set classification accuracies (%) across all evaluated pipelines (`result_csvs/`):

| Pipeline | Spectral Band | P1 Pre | P1 Post | P2 Pre | P2 Post | P3 Pre | P3 Post | Mean Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Riemannian + LR** (`phs_2.3a`) | **Beta (13 to 30 Hz)** | **96.25%** | 88.75% | **80.00%** | **100.00%** | 76.25% | **96.25%** | **89.58%** |
| **Riemannian + LR** (`phs_2.3b`) | **Mu + Beta (8 to 12 & 13 to 30 Hz)** | 91.25% | **90.00%** | 68.75% | **100.00%** | **91.25%** | 91.25% | **88.75%** |
| **Riemannian + LR** (`phs_2.3a`) | Mu (8 to 12 Hz) | 77.50% | 85.00% | 60.00% | 81.25% | 80.00% | 81.25% | 77.50% |
| **Riemannian + LR** (`phs_2.3`) | Full Band | 65.00% | 75.00% | 56.25% | 87.50% | 83.75% | 83.75% | 75.21% |
| **Riemannian + SVM** (`phs_2.3`) | Full Band | 66.25% | 78.75% | 52.50% | 86.25% | 82.50% | 81.25% | 74.58% |
| **Riemannian + LR (Whitened)** | Full Band (ZCA whitened) | 68.75% | 60.00% | 52.50% | 88.75% | 91.25% | 80.00% | 73.54% |
| **CSP (4 comp) + SVM** (`phs_2.1`) | Mu + Beta | 62.50% | 68.75% | 55.00% | 70.00% | 68.75% | 72.50% | 66.25% |
| **CSP + LDA** (`phs_2.1`) | Mu + Beta | 56.25% | 73.75% | 48.75% | 70.00% | 65.00% | 68.75% | 63.75% |
| **PCA (8) + TVLDA** (`phs_2.2`) | Full Band | 77.50% | 50.00% | 57.50% | 75.00% | 66.25% | 53.75% | 63.33% |

---

## Neurophysiological Visualisations (`phs_4.*.py`)

To verify that the high decoding accuracy reflects genuine sensorimotor rhythms rather than muscular or ocular artefacts, four neurophysiological visualisation modules are included:

1. **Morlet Wavelet Time-Frequency ERD/ERS (`phs_4.1_erd_ers_vis.py`):** Computes baseline-corrected (0 to 0.5 s) percentage power changes across 8 to 35 Hz using MNE `tfr_morlet`, demonstrating stronger, more sustained Beta-band modulation post-therapy.
2. **Pipeline & Pre/Post Accuracy Benchmarks (`phs_4.2_results_plots.py`):** Generates summary bar charts comparing pipeline families and quantifying Pre vs Post therapy gains per patient.
3. **Tangent-Space t-SNE & Confusion Matrices (`phs_4.3_advanced_vis.py`):** Embeds the 136-dimensional Beta-band Riemannian tangent vectors into 2D via PCA-initialised t-SNE (`perplexity=25`), illustrating the transition from overlapping Pre-therapy distributions to cleanly separated Left/Right clusters Post-therapy (`P2 Post`: 40/40 Left, 40/40 Right; `P3 Post`: 38/40 Left, 39/40 Right).
4. **Cortical Topomaps of Beta Power Shifts (`phs_4.4_topo_maps.py`):** Projects `Post - Pre` Beta-band variance differences onto the 16-channel `standard_1020` scalp montage (`mne.viz.plot_topomap`), localising therapy-induced power changes to the lateralised motor strip (`C3`/`FC3` and `C4`/`CP4`).

---

## Repository Structure

```text
├── phs_1.1_load_data_single.py         # Single-file .mat inspection and trigger parsing
├── phs_1.2_extract_epochs_single.py    # Trial segmentation relative to cue triggers
├── phs_1.3_time_whitening.py           # Baseline covariance estimation and ZCA time whitening
├── phs_1.4_load_and_extract_epochs.py  # Batch preprocessing across all patients (P1-P3) and stages
├── phs_2.1_csp_lda_.py                 # Common Spatial Patterns (CSP) + LDA baseline
├── phs_2.1_csp_svm_.py                 # CSP + Support Vector Machine baseline
├── phs_2.2_tvlda_pca.py                # PCA + Time-Variant LDA (TVLDA) benchmark
├── phs_2.3_riemannian.py               # Full-band Riemannian covariance + Tangent Space pipeline
├── phs_2.3a_riemannian.py              # Band-specific (Mu 8-12 Hz / Beta 13-30 Hz) Riemannian pipeline
├── phs_2.3ab_riemannian.py             # Riemannian Beta pipeline exporting detailed predictions & features
├── phs_2.3b_riemannian.py              # Dual-band (Mu + Beta concatenated tangent space) Riemannian pipeline
├── phs_4.1_erd_ers_vis.py              # MNE Morlet wavelet time-frequency ERD/ERS comparisons
├── phs_4.2_results_plots.py            # Pipeline comparison and Pre vs Post therapy bar plots
├── phs_4.3_advanced_vis.py             # Confusion matrices and t-SNE embeddings of tangent-space features
├── phs_4.4_topo_maps.py                # 16-channel cortical topomaps of Post - Pre Beta power change
├── detailed_results_p2_p3.pkl          # Saved tangent-space features and test predictions for P2 & P3
├── result_csvs/                        # Benchmark accuracy tables across all 9 pipeline configurations
└── requirements.txt                    # Python dependencies
```

---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce Visualisations from Saved Features & Results

You can immediately generate the benchmark comparison charts, confusion matrices, and t-SNE tangent-space plots using the tracked `result_csvs/` and `detailed_results_p2_p3.pkl`:

```bash
python phs_4.2_results_plots.py
python phs_4.3_advanced_vis.py
```

### 3. Run Full Decoding Pipeline from Raw `.mat` Recordings

Place the stroke rehabilitation `.mat` recordings (`P1_pre_training.mat`, etc.) inside `stroke-rehab/`, generate the epoched dataset `all_processed_data.pkl`, and run the Riemannian decoding scripts:

```bash
python phs_1.4_load_and_extract_epochs.py
python phs_2.3a_riemannian.py
python phs_2.3b_riemannian.py
```

---

## Licence

Released under the [MIT Licence](LICENSE).
