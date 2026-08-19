# X-LMC: Cross-View Spatiotemporal Collateral Scoring from DSA

### 🎉 Accepted to the SWITCH Workshop at MICCAI 2026

Official implementation of **X-LMC**, a deep learning framework for automated ASITN/SIR leptomeningeal collateral (LMC) grading from time-resolved biplane digital subtraction angiography (DSA).

> **Note:** This repository accompanies a paper accepted to the SWITCH Workshop at MICCAI 2026. The README, quantitative results, figures, and citation information will be updated after the official publication.

---

## Overview

Leptomeningeal collateral status is an important prognostic indicator in acute ischemic stroke, supporting secondary treatment strategies, neurorehabilitation planning, and retrospective stroke research. DSA is the reference standard for LMC assessment, but clinical grading using the ASITN/SIR scale remains manual, time-intensive, and subject to inter-rater variability.

X-LMC formulates collateral grading as a **multi-view spatiotemporal learning problem**. Synchronized anteroposterior (AP) and lateral (LAT) DSA sequences are processed through three stages:

1. **Frame-wise spatial encoding** — AP and LAT frames are independently encoded using a frozen DINOv2 ViT-B/14 backbone.
2. **Bidirectional cross-view attention (X-Attn)** — token-level representations from the two projections interact at each timepoint to obtain cross-view enhanced representations.
3. **Temporal modeling** — the fused sequence is processed using a bidirectional GRU (Bi-GRU) to model contrast-flow dynamics over time, followed by an MLP classifier for collateral grade prediction.


## Results

X-LMC was evaluated on a multicenter cohort of patients with M1-segment MCA occlusions using cross-validation.

Detailed quantitative results and additional qualitative analyses will be added following the official publication of the SWITCH Workshop paper.

---

## Installation

```bash
git clone https://github.com/maedehafezi/X-LMC.git
cd X-LMC
pip install -r requirements.txt
