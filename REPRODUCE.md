# Reproducing Research 2

Step-by-step commands for repeating the Research 2 study (12 systems on SECOND) from this
repository. Research 1 (LEVIR-CD) was trained in the notebooks `notebooks/research_1/01–04` and
`heavy_*`; its final detectors are in `artifacts/final_models/` (Git LFS) and Research 2 starts from
them.

## Where the code is

| What | Where |
|---|---|
| SCD models | `src/satchange/models/siam_unet_resnet.py` (Siamese U-Net + ResNet-18), `scannet.py` (SCanNet + ResNet-34, reimplemented from the paper), `mamba_scd.py` + `selective_scan.py` (MambaSCD-Tiny, port of ChangeMamba, Apache-2.0) |
| Detectors (Research 1) | `src/satchange/models/fc_siam_diff.py`, `bit.py`, `siam_mit_b3_diff.py` |
| Training engine | `src/satchange/training.py` (one engine for binary and semantic tasks) |
| Data, metrics, ROI, statistics | `src/satchange/data/`, `src/satchange/semantic/` |
| Training runs | `src/scripts/run_r2_v2_scd.py` (SCD model), `run_r2_v2_detector.py` (detector) |
| HPO | `src/scripts/run_r2_v2_hpo.py` (6 trials per SCD model, each a `run_r2_v2_scd.py` run) |
| Evaluation and analysis | `src/scripts/prepare_r2_v2_val.py`, `eval_r2_v2_systems.py`, `analyze_r2_v2_systems.py`, `freeze_r2_v2.py`, `latency_r2_v2_systems.py` |

Research 2 has no training notebooks: every run is a script that writes `run_record.json` and a row
of `artifacts/research2/v2/ledger.jsonl`. Outputs go to `artifacts/research2/v2/` (not in git).

## 1. Environment

```bash
git clone https://github.com/romanustenko/satellite-change-intelligence.git
cd satellite-change-intelligence
git lfs install && git lfs pull --include="artifacts/final_models/**"
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision     # a CUDA build; the study used torch 2.12.1+cu130
pip install -e .
```

**MambaSCD CUDA kernel.** MambaSCD needs `selective_scan` from ChangeMamba
(https://github.com/ChenHongruixuan/MambaCD, commit `9ce9cec`): ChangeMamba's own instruction is
`cd kernels/selective_scan && pip install .`. This study used CUDA 13, where two edits were needed
first:

- `setup.py`: `arch=compute_70,code=sm_70` → `arch=compute_80,code=sm_80` (CUDA 13 dropped Volta);
- `csrc/selective_scan/reverse_scan.cuh`: add `#include <cuda/ptx>`, replace
  `cub::LaneId()` with `::cuda::ptx::get_sreg_laneid()` and both `cub::CTA_SYNC()` with
  `__syncthreads()` (removed in CCCL 3);

then `TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install --no-build-isolation .`. Check:
`python -c "import selective_scan_cuda_oflex"`. U-Net and SCanNet do not need the kernel.

**Hardware.** All runs used A100-80GB GPUs, batch 8 at 512×512, fp32. MambaSCD needs more than
48 GB of GPU memory in training; U-Net and SCanNet fit in less. Wall-clock time on a shared 4×A100
server, summed over runs: HPO ≈ 180 h, final SCD runs ≈ 225 h, detectors ≈ 115 h.

## 2. Data and pretrained weights

**SECOND** (https://captain-whu.github.io/SCD/): extract the official train and test sets to

```text
data/SECOND/train/{im1,im2,label1,label2}/*.png     # 2968 pairs
data/SECOND/test/{im1,im2,label1,label2}/*.png      # 1694 pairs
```

The split manifest `data/SECOND/second_split_manifest.json` is in git (train 2524 / val 444 / test
1694) and records the SHA-256 of every file. `notebooks/research_2/01_second_dataset_preparation.ipynb`
checks the installation; with `RUN_VERIFY_HASHES = True` it re-hashes every file against the
manifest (`satchange.data.second_binary.verify_second_sources`).

**ImageNet encoder weights** (checked by SHA-256 when loaded):

| File | URL | SHA-256 |
|---|---|---|
| `artifacts/research2/v2/pretrained/torchvision/resnet18-f37072fd.pth` | https://download.pytorch.org/models/resnet18-f37072fd.pth | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| `artifacts/research2/v2/pretrained/torchvision/resnet34-b627a593.pth` | https://download.pytorch.org/models/resnet34-b627a593.pth | `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` |
| `artifacts/research2/v2/pretrained/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth` | https://zenodo.org/records/15479555 (VMamba-Tiny, CC-BY-4.0) | `09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201` |

## 3. Detectors

The Research 1 detectors are fine-tuned on SECOND train as binary change detectors (budget
3 × 10¹⁶ training FLOPs, seeds 42/123/777, lr chosen by a probe over {1e-4, 3e-4, 1e-3}).

```bash
D=cuda:0
# screening of Light / Medium / Heavy; case 2 uses medium_full, case 4 may use light_full
for det in light_full medium_full heavy_full; do for s in 42 123 777; do
  python src/scripts/run_r2_v2_detector.py --detector $det --flop-budget 3e16 --lr 1e-3 \
    --seed $s --device $D --group screening --name ${det}_s$s
done; done
# case 3: same Medium architecture trained on SECOND only; case-4 partner candidate Light Resized
# (each command runs the lr probe and then the three seeds)
python src/scripts/run_r2_v2_detector_chain.py --detector medium_full --init scratch --tag medium_scratch --device $D
python src/scripts/run_r2_v2_detector_chain.py --detector light_resized --init research1 --tag light_resized --device $D
```

Optional, to repeat the detector choice (Medium):
`eval_r2_v2_detector_screening.py infer --detector <det> --seed <s> --device $D` for each run,
then `eval_r2_v2_detector_screening.py report --output <dir>` (the screening rule is the committed
`configs/SCREENING_RULE.json`).

## 4. SCD models: HPO and final runs

Equal budget for every model: 6 HPO trials × 8 × 10¹⁶ FLOPs, then 3 final runs × 3.2 × 10¹⁷ FLOPs
with the winning learning rate and weight decay. Training stops at a precomputed optimizer-step
count; the checkpoint with the best validation SeK is kept.

```bash
for m in siam_unet_resnet18 scannet mamba_scd; do
  python src/scripts/run_r2_v2_hpo.py --model $m --device $D   # -> runs/hpo/<m>/search_result.json
done
# winners in this study: U-Net 9.4e-4 / 0.05, SCanNet 2.5e-4 / 0.01, MambaSCD 4.9e-4 / 0.05
for s in 42 123 777; do
  python src/scripts/run_r2_v2_scd.py --model siam_unet_resnet18 --flop-budget 3.2e17 --lr 0.00094 \
    --weight-decay 0.05 --seed $s --device $D --group final --name unet_final_s$s
  python src/scripts/run_r2_v2_scd.py --model scannet --flop-budget 3.2e17 --lr 0.00025 \
    --weight-decay 0.01 --seed $s --device $D --group final --name scannet_final_s$s
  python src/scripts/run_r2_v2_scd.py --model mamba_scd --flop-budget 3.2e17 --lr 0.00049 \
    --weight-decay 0.05 --seed $s --device $D --group final --name mamba_final_s$s
done
```

`src/scripts/r2_v2_queue.py` runs the same HPO trials and final runs as a queue over several GPUs
and takes the winners from `search_result.json` itself.

## 5. Systems on validation, freeze, one test read

```bash
python src/scripts/prepare_r2_v2_val.py maps --device $D       # detector maps on the 444 val tiles
python src/scripts/prepare_r2_v2_val.py partner --device $D    # case-4 OR partner
python src/scripts/prepare_r2_v2_val.py policies --device $D   # ROI policies (area budgets 0.7/0.5/0.3)
python src/scripts/measure_r2_v2_inference_flops.py --device $D
python src/scripts/eval_r2_v2_systems.py --split val --devices $D
python src/scripts/analyze_r2_v2_systems.py --split val       # density edges + delta rule value
```

Freeze everything chosen on validation before reading test. `--protocol` is the text file in which
you wrote down hypotheses, comparisons and decision rules (ours is not published; the rules are in
`REPORT.md` §2.4–2.5). The freeze refuses a dirty `src/`/`tests` tree. Values of this study:

```bash
python src/scripts/freeze_r2_v2.py --delta 0.007 --edges 0.11491648356119792 0.22790145874023438 \
  --protocol <protocol.md> --output artifacts/research2/v2/v9/freeze_manifest_v2.json
python src/scripts/eval_r2_v2_systems.py --split test \
  --manifest artifacts/research2/v2/v9/freeze_manifest_v2.json --devices $D   # single test read
python src/scripts/analyze_r2_v2_systems.py --split test --delta 0.007 \
  --edges 0.11491648356119792 0.22790145874023438
python src/scripts/latency_r2_v2_systems.py --device $D      # idle GPU; refuses a busy card
python src/scripts/make_r2_v2_results.py                       # summary tables
```

## 6. Ablation H6 (optional)

Semantic models trained only on ground-truth crops (same budget and lr / weight decay; validation
only, seed 42). Needs the ROI policies, the case-1 validation evaluation and `analysis_test.json`
(frozen δ and density edges) from step 5. Results of this study: `results/research2/ablation_h6/`.

```bash
python src/scripts/run_r2_v2_ablation_b.py --model siam_unet_resnet18 --flop-budget 3.2e17 \
  --lr 0.00094 --weight-decay 0.05 --seed 42 --device $D
python src/scripts/run_r2_v2_ablation_b.py --model scannet --flop-budget 3.2e17 \
  --lr 0.00025 --weight-decay 0.01 --seed 42 --device $D
python src/scripts/run_r2_v2_ablation_b.py --model mamba_scd --flop-budget 3.2e17 \
  --lr 0.00049 --weight-decay 0.05 --seed 42 --device $D
R2_SCD_ROOT=artifacts/research2/v2/runs/ablation_b python src/scripts/eval_r2_v2_systems.py \
  --split val --devices $D --seeds 42 --out artifacts/research2/v2/ablation_b/eval/val
python src/scripts/analyze_r2_v2_ablation_b.py   # -> artifacts/research2/v2/ablation_b/analysis_val.json
```

## Determinism

Runs are seeded and use deterministic algorithms (a 10-step SCanNet run from a fresh clone of this
repository reproduced the recorded validation SeK exactly on the same server). MambaSCD's
`selective_scan` backward pass is not bitwise deterministic, so its reruns differ slightly. Other
GPUs, drivers or PyTorch versions give small numeric differences.
