#!/bin/bash

# Master Execution Script
# Runs all validation checks in order as specified by user

echo "========================================================================"
echo "DYNAMIC CO-EVOLUTION SYSTEM - VALIDATION PIPELINE"
echo "========================================================================"
echo ""
echo "Ordered execution:"
echo "  1. CHECK 1: Validation suite"
echo "  2. CHECK 2: Reward sanity"
echo "  3. CHECK 3: Balance test"
echo "  4. CHECK 4: One generation"
echo "  5. (Optional) Add monitoring"
echo "  6. Full 20-generation training"
echo ""
echo "========================================================================"
echo ""

# CHECK 1: Validation Suite
echo "Starting CHECK 1: Validation Suite..."
python validate_dynamic_system.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHECK 1 FAILED - Fix validation errors before proceeding"
    exit 1
fi

echo ""
echo "✅ CHECK 1 PASSED"
echo ""

# CHECK 2: Reward Sanity
echo "Starting CHECK 2: Reward Sanity..."
python check_2_reward_sanity.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHECK 2 FAILED - Fix reward structure before proceeding"
    exit 1
fi

echo ""
echo "✅ CHECK 2 PASSED"
echo ""

# CHECK 3: Balance Test
echo "Starting CHECK 3: Balance Test..."
python check_3_balance_test.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHECK 3 FAILED - Fix balance issues before proceeding"
    exit 1
fi

echo ""
echo "✅ CHECK 3 PASSED"
echo ""

# CHECK 4: One Generation
echo "Starting CHECK 4: One Generation Test..."
python check_4_one_generation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHECK 4 FAILED - Fix training issues before proceeding"
    exit 1
fi

echo ""
echo "✅ CHECK 4 PASSED"
echo ""

# All checks passed
echo "========================================================================"
echo "✅ ALL CHECKS PASSED"
echo "========================================================================"
echo ""
echo "System validated and ready for full training!"
echo ""
echo "To run full 20-generation training:"
echo "  python aceac_dynamic_coevolution.py"
echo ""
echo "Estimated runtime: Several hours (depends on hardware)"
echo "Output: models/dynamic/red_final.zip, blue_final.zip, training_history.json"
echo ""
echo "========================================================================"
