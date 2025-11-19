"""
CHECK 4: One Generation Test

Run a single complete generation of co-evolution training.
Verifies:
1. Training completes without errors
2. Models save/load correctly
3. Evaluation produces metrics
4. Co-evolution dynamics are working
"""

import numpy as np
from aceac_dynamic_coevolution import DynamicCoEvolutionTrainer
from pathlib import Path
import json

def test_one_generation():
    """Run one complete generation"""
    print("\n" + "="*70)
    print("CHECK 4: ONE GENERATION TEST")
    print("="*70)
    print("Running 1 generation with reduced steps for quick validation\n")

    try:
        # Initialize trainer with reduced parameters for speed
        trainer = DynamicCoEvolutionTrainer(
            num_actions=10,         # Reduced from 25
            state_dim=16,           # Reduced from 32
            population_size=2       # Reduced from 5
        )

        print("Training configuration:")
        print(f"  Generations: 1")
        print(f"  Steps per generation: 5000")
        print(f"  Actions: 10")
        print(f"  State dimensions: 16")
        print(f"  Population size: 2")
        print()

        # Train for 1 generation
        red_model, blue_model = trainer.train(
            num_generations=1,
            steps_per_generation=5000,  # Reduced from 50000
            save_dir="models/check4_test"
        )

        print("\n" + "="*70)
        print("ANALYZING RESULTS")
        print("="*70)

        # Check that training history was recorded
        if len(trainer.generation_history) != 1:
            print(f"❌ Expected 1 generation in history, got {len(trainer.generation_history)}")
            return False

        print(f"✓ Training history recorded: {len(trainer.generation_history)} generation(s)")

        # Check generation data
        gen_data = trainer.generation_history[0]

        print(f"\nGeneration 1 metrics:")
        print(f"  State dominance: {gen_data['evaluation']['state_dominance']:.3f}")
        print(f"  Red diversity: {gen_data['evaluation']['red_diversity']:.3f}")
        print(f"  Blue diversity: {gen_data['evaluation']['blue_diversity']:.3f}")

        # Verify metrics are reasonable
        state_dom = gen_data['evaluation']['state_dominance']
        red_div = gen_data['evaluation']['red_diversity']
        blue_div = gen_data['evaluation']['blue_diversity']

        if 0.0 <= state_dom <= 1.0:
            print("✓ State dominance in valid range [0, 1]")
        else:
            print(f"⚠️  State dominance outside expected range: {state_dom}")

        if 0.0 <= red_div <= 1.0:
            print("✓ Red diversity in valid range [0, 1]")
        else:
            print(f"⚠️  Red diversity outside expected range: {red_div}")

        if 0.0 <= blue_div <= 1.0:
            print("✓ Blue diversity in valid range [0, 1]")
        else:
            print(f"⚠️  Blue diversity outside expected range: {blue_div}")

        # Check diversity is not too low (strategy collapse)
        if red_div > 0.1 and blue_div > 0.1:
            print("✓ Both agents show action diversity (no collapse)")
        else:
            print("⚠️  Low diversity detected - possible strategy collapse")

        # Check models were saved
        save_path = Path("models/check4_test")

        if (save_path / "red_final.zip").exists():
            print("\n✓ Red model saved successfully")
        else:
            print("\n❌ Red model not saved")
            return False

        if (save_path / "blue_final.zip").exists():
            print("✓ Blue model saved successfully")
        else:
            print("❌ Blue model not saved")
            return False

        if (save_path / "training_history.json").exists():
            print("✓ Training history saved successfully")

            # Load and verify history file
            with open(save_path / "training_history.json", 'r') as f:
                saved_history = json.load(f)

            if 'generations' in saved_history and len(saved_history['generations']) == 1:
                print("✓ Training history JSON is valid")
            else:
                print("⚠️  Training history JSON format unexpected")

        else:
            print("❌ Training history not saved")
            return False

        # Test model loading
        print("\nTesting model loading...")
        from stable_baselines3 import PPO

        try:
            loaded_red = PPO.load(save_path / "red_final.zip")
            print("✓ Red model loads successfully")
        except Exception as e:
            print(f"❌ Failed to load Red model: {e}")
            return False

        try:
            loaded_blue = PPO.load(save_path / "blue_final.zip")
            print("✓ Blue model loads successfully")
        except Exception as e:
            print(f"❌ Failed to load Blue model: {e}")
            return False

        # Test inference with loaded models
        print("\nTesting model inference...")
        from aceac_dynamic_coevolution import DynamicCoEvolutionEnv

        test_env = DynamicCoEvolutionEnv(
            agent_role="red",
            num_actions=10,
            state_dim=16
        )

        obs, _ = test_env.reset()

        try:
            action, _ = loaded_red.predict(obs, deterministic=True)
            print(f"✓ Red model prediction works (action: {action})")
        except Exception as e:
            print(f"❌ Red model prediction failed: {e}")
            return False

        try:
            action, _ = loaded_blue.predict(obs, deterministic=True)
            print(f"✓ Blue model prediction works (action: {action})")
        except Exception as e:
            print(f"❌ Blue model prediction failed: {e}")
            return False

        print("\n" + "="*70)
        print("✅ CHECK 4 PASSED - One generation training works!")
        print("="*70)
        print("\nSystem ready for full 20-generation training.")
        print("\nNext steps:")
        print("  1. Add monitoring code (optional)")
        print("  2. Run full 20-generation training")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ CHECK 4 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_one_generation()
    sys.exit(0 if success else 1)
