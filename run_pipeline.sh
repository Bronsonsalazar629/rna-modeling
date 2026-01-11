#!/bin/bash
# Master activation script for RNA 3D Folding pipeline
# Execute step-by-step to reach Kaggle submission

set -e  # Exit on error

echo "========================================================================"
echo "RNA 3D FOLDING - ACTIVATION SEQUENCE"
echo "========================================================================"
echo ""
echo "Goal: TM-score ≥ 0.85 (easy), ≥ 0.65 (medium), ≥ 0.45 (hard)"
echo "      OpenMM pass rate ≥ 95%"
echo "      Inference < 8 GPU hours"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 0: PRE-TRAINING SANITY CHECKS (CRITICAL)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 0: Pre-Training Sanity Checks"
echo "========================================================================"
echo ""
echo "Verifying 5 critical contracts..."
echo "  1. Label-Sequence Alignment"
echo "  2. MSA Gap Stripping"
echo "  3. FAPE Implementation"
echo "  4. SE(3)-Equivariance"
echo "  5. Gradient Flow"
echo ""

python validation/pre_training_checks.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ SANITY CHECKS FAILED${NC}"
    echo ""
    echo "DO NOT PROCEED TO TRAINING!"
    echo "Fix the failing contracts and re-run:"
    echo "  python validation/pre_training_checks.py"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All sanity checks passed${NC}"
echo ""
read -p "Press Enter to continue to data preparation..."

# ============================================================================
# STEP 1: PREPARE TRAINING DATA
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: Prepare Training Data"
echo "========================================================================"
echo ""
echo "Converting train_labels.csv to multi-structure format..."
echo ""

python train/prepare_labels.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Data preparation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Data preparation complete${NC}"
echo "  Output: data/processed/train_structures.pkl"
echo ""
read -p "Press Enter to continue to debug training..."

# ============================================================================
# STEP 2: DEBUG TRAINING (10 easy targets)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Debug Training - 10 Easy Targets"
echo "========================================================================"
echo ""
echo "Purpose: Verify learning happens before scaling"
echo ""
echo "Configuration:"
echo "  - Targets: 10 (≤40 nt)"
echo "  - Evoformer blocks: 8"
echo "  - Epochs: 10"
echo "  - Learning rate: 1e-3"
echo ""
echo "Success criteria:"
echo "  - Epoch 1: TM ≈ 0.1-0.2"
echo "  - Epoch 5: TM ≥ 0.4"
echo "  - Epoch 10: TM ≥ 0.6"
echo "  - OpenMM pass ≥ 80%"
echo ""

python train/debug_train.py \
  --num_targets 10 \
  --max_length 40 \
  --evoformer_blocks 8 \
  --epochs 10 \
  --lr 1e-3 \
  --log_every 1 \
  --save_checkpoint

DEBUG_EXIT_CODE=$?

if [ $DEBUG_EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Debug training failed (TM < 0.60)${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if loss is decreasing"
    echo "  2. Verify gradients are flowing (not NaN)"
    echo "  3. Try higher LR: --lr 2e-3"
    echo "  4. Increase epochs: --epochs 20"
    echo "  5. Re-run sanity checks"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Debug training successful (TM ≥ 0.60)${NC}"
echo "  Ready to scale to full training!"
echo ""
read -p "Press Enter to see next steps..."

# ============================================================================
# NEXT STEPS (Manual execution recommended)
# ============================================================================

echo ""
echo "========================================================================"
echo "NEXT STEPS - Manual Execution Recommended"
echo "========================================================================"
echo ""
echo "Step 3: Stage 1 Training (12-24 hours)"
echo "  python train/stage1_train.py \\"
echo "    --stage 1 \\"
echo "    --max_length 50 \\"
echo "    --exclude_pseudoknots \\"
echo "    --max_samples 1000 \\"
echo "    --lr 1e-3 \\"
echo "    --batch_size 16 \\"
echo "    --epochs 20"
echo ""
echo "  Success: TM ≥ 0.85 on easy validation set"
echo "  Output: data/checkpoints/stage1_best_tm85.pkl"
echo ""

echo "Step 4: Stage 2 Training (24-48 hours)"
echo "  python train/stage2_train.py \\"
echo "    --stage 2 \\"
echo "    --max_length 150 \\"
echo "    --allow_pseudoknots \\"
echo "    --max_samples 5000 \\"
echo "    --lr 5e-4 \\"
echo "    --batch_size 8 \\"
echo "    --epochs 15 \\"
echo "    --load_checkpoint data/checkpoints/stage1_best_tm85.pkl"
echo ""
echo "  Success: TM ≥ 0.65 on medium validation set"
echo "  Output: data/checkpoints/stage2_best_tm65.pkl"
echo ""

echo "Step 5: Stages 3 & 4 Training (48-72 hours)"
echo "  python train/stage3_train.py  # Hard targets"
echo "  python train/stage4_train.py  # Full dataset"
echo ""
echo "  Success: TM ≥ 0.45 on hard validation set"
echo "  Output: data/checkpoints/final_model.pkl"
echo ""

echo "Step 6: Inference (4-8 hours)"
echo "  python inference/ensemble_predict.py \\"
echo "    --checkpoint data/checkpoints/final_model.pkl \\"
echo "    --test_sequences data/raw/test_sequences.csv \\"
echo "    --output predictions/ensemble_predictions.pkl \\"
echo "    --temperatures 0.5,0.8,1.0,1.2,1.5"
echo ""

echo "Step 7: Create Submission"
echo "  python submission/kaggle_submit.py \\"
echo "    --predictions predictions/ensemble_predictions.pkl \\"
echo "    --output submission.csv"
echo ""

echo "Step 8: Validate & Submit"
echo "  python submission/validate_submission.py --submission submission.csv"
echo "  kaggle competitions submit -c stanford-rna-folding -f submission.csv"
echo ""

echo "========================================================================"
echo "CURRENT STATUS"
echo "========================================================================"
echo ""
echo -e "${GREEN}✓ Phase 1: Data Engineering (Complete)${NC}"
echo -e "${GREEN}✓ Phase 1.5: Model Architecture (Complete)${NC}"
echo -e "${GREEN}✓ Step 0: Sanity Checks (Passed)${NC}"
echo -e "${GREEN}✓ Step 1: Data Preparation (Complete)${NC}"
echo -e "${GREEN}✓ Step 2: Debug Training (Success)${NC}"
echo ""
echo -e "${YELLOW}→ Next: Step 3 - Stage 1 Training${NC}"
echo ""
echo "Estimated time to submission: 5-7 days (if stages run sequentially)"
echo ""
