#!/usr/bin/env python3
"""
Co-Evolution Edge Case Detector
Automated monitoring and detection tool for adversarial RL training

Author: @sarowarzahan414
Date: 2025-11-18
Usage: python coevolution_edge_case_detector.py --training-log logs/coevolution_training.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class EdgeCaseAlert:
    """Represents an edge case detection alert"""
    detector: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    generation: int
    timestamp: str
    metrics: Dict
    recommendation: str

    def to_dict(self):
        return asdict(self)


class CoEvolutionEdgeCaseDetector:
    """
    Automated edge case detection for co-evolution training
    Monitors training in real-time and detects problematic patterns
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.alerts = []
        self.training_history = []

    def _default_config(self) -> Dict:
        """Default configuration for edge case detection"""
        return {
            'cyclic_dominance': {
                'enabled': True,
                'window': 5,
                'correlation_threshold': 0.8,
                'severity': 'CRITICAL'
            },
            'strategy_collapse': {
                'enabled': True,
                'entropy_threshold': 0.2,
                'severity': 'CRITICAL'
            },
            'catastrophic_forgetting': {
                'enabled': True,
                'performance_drop_threshold': 0.2,
                'severity': 'HIGH'
            },
            'runaway_optimization': {
                'enabled': True,
                'capability_cap': 0.95,
                'severity': 'HIGH'
            },
            'deadlock': {
                'enabled': True,
                'variance_threshold': 0.01,
                'window': 20,
                'severity': 'MEDIUM'
            },
            'phase_stagnation': {
                'enabled': True,
                'stagnation_window': 20,
                'severity': 'MEDIUM'
            }
        }

    def analyze_generation(self, generation_data: Dict) -> List[EdgeCaseAlert]:
        """
        Analyze a single generation for edge cases

        Args:
            generation_data: Data from one generation of co-evolution

        Returns:
            List of detected edge cases
        """
        alerts = []

        # Extract data
        gen_num = generation_data.get('generation', 0)
        test_results = generation_data.get('test_results', {})

        # Run all detectors
        if self.config['cyclic_dominance']['enabled']:
            alert = self._detect_cyclic_dominance(gen_num)
            if alert:
                alerts.append(alert)

        if self.config['strategy_collapse']['enabled']:
            alert = self._detect_strategy_collapse(generation_data)
            if alert:
                alerts.append(alert)

        if self.config['runaway_optimization']['enabled']:
            alert = self._detect_runaway_optimization(test_results, gen_num)
            if alert:
                alerts.append(alert)

        # Store in history
        self.training_history.append(generation_data)

        return alerts

    def _detect_cyclic_dominance(self, generation: int) -> Optional[EdgeCaseAlert]:
        """Detect oscillating win patterns"""
        window = self.config['cyclic_dominance']['window']

        if len(self.training_history) < window * 2:
            return None

        # Extract win rates
        red_wins = []
        for gen_data in self.training_history[-(window*2):]:
            test_res = gen_data.get('test_results', {})
            red_wins.append(test_res.get('red_wins', 0))

        # Check for oscillation
        first_half = red_wins[:window]
        second_half = red_wins[window:]

        try:
            correlation = np.corrcoef(first_half, second_half)[0, 1]
        except:
            return None

        threshold = self.config['cyclic_dominance']['correlation_threshold']

        if abs(correlation) > threshold:
            return EdgeCaseAlert(
                detector='cyclic_dominance',
                severity=self.config['cyclic_dominance']['severity'],
                category='Co-Evolution Dynamics',
                description=f'Oscillating win pattern detected (correlation={correlation:.2f})',
                generation=generation,
                timestamp=datetime.now().isoformat(),
                metrics={'correlation': float(correlation), 'window': window},
                recommendation='Enable population-based training or add strategy diversity incentives'
            )

        return None

    def _detect_strategy_collapse(self, generation_data: Dict) -> Optional[EdgeCaseAlert]:
        """Detect if agents are using limited strategy diversity"""

        # This would require action history, which we don't have in summary data
        # In real implementation, would analyze action distributions
        # For now, return None (would need to be integrated into training loop)

        return None

    def _detect_runaway_optimization(self, test_results: Dict, generation: int) -> Optional[EdgeCaseAlert]:
        """Detect if agents exceed realistic capabilities"""

        cap = self.config['runaway_optimization']['capability_cap']

        red_wins = test_results.get('red_wins', 0)
        blue_wins = test_results.get('blue_wins', 0)
        total = red_wins + blue_wins + test_results.get('draws', 0)

        if total == 0:
            return None

        red_rate = red_wins / total
        blue_rate = blue_wins / total

        issues = []
        if red_rate > cap:
            issues.append(f'Red too powerful: {red_rate:.1%} win rate')
        if blue_rate > cap:
            issues.append(f'Blue too powerful: {blue_rate:.1%} win rate')

        if issues:
            return EdgeCaseAlert(
                detector='runaway_optimization',
                severity=self.config['runaway_optimization']['severity'],
                category='Capability Bounds',
                description='; '.join(issues),
                generation=generation,
                timestamp=datetime.now().isoformat(),
                metrics={'red_win_rate': float(red_rate), 'blue_win_rate': float(blue_rate)},
                recommendation='Enforce capability bounds or reset to previous checkpoint'
            )

        return None

    def analyze_training_log(self, log_file: Path) -> List[EdgeCaseAlert]:
        """
        Analyze complete training log

        Args:
            log_file: Path to training log JSON file

        Returns:
            List of all detected edge cases
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING TRAINING LOG: {log_file}")
        print(f"{'='*70}\n")

        # Load log
        with open(log_file) as f:
            log_data = json.load(f)

        generation_results = log_data.get('generation_results', [])

        print(f"Found {len(generation_results)} generations to analyze\n")

        all_alerts = []

        for gen_data in generation_results:
            gen_num = gen_data.get('generation', 0)
            print(f"Analyzing Generation {gen_num}...", end=" ")

            alerts = self.analyze_generation(gen_data)

            if alerts:
                print(f"⚠️  {len(alerts)} issue(s) found")
                all_alerts.extend(alerts)
            else:
                print("✅ OK")

        self.alerts.extend(all_alerts)

        return all_alerts

    def generate_report(self, output_file: Optional[Path] = None):
        """Generate comprehensive edge case report"""

        print(f"\n{'='*70}")
        print(f"EDGE CASE DETECTION REPORT")
        print(f"{'='*70}\n")

        if not self.alerts:
            print("✅ No edge cases detected - Training is healthy!\n")
            return

        # Group by severity
        by_severity = defaultdict(list)
        for alert in self.alerts:
            by_severity[alert.severity].append(alert)

        print(f"Total Issues Found: {len(self.alerts)}\n")
        print(f"  🔴 CRITICAL: {len(by_severity['CRITICAL'])}")
        print(f"  🟠 HIGH:     {len(by_severity['HIGH'])}")
        print(f"  🟡 MEDIUM:   {len(by_severity['MEDIUM'])}")
        print(f"  🟢 LOW:      {len(by_severity['LOW'])}\n")

        # Print details by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            alerts = by_severity[severity]
            if not alerts:
                continue

            emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
            print(f"{emoji} {severity} Issues ({len(alerts)}):")
            print("="*70)

            for alert in alerts:
                print(f"\n  [Gen {alert.generation}] {alert.category}")
                print(f"  Detector: {alert.detector}")
                print(f"  Issue:    {alert.description}")
                print(f"  Fix:      {alert.recommendation}")
                print(f"  Metrics:  {alert.metrics}")

        print(f"\n{'='*70}\n")

        # Save to file
        if output_file:
            report_data = {
                'report_date': datetime.now().isoformat(),
                'total_generations_analyzed': len(self.training_history),
                'total_issues': len(self.alerts),
                'by_severity': {k: len(v) for k, v in by_severity.items()},
                'alerts': [alert.to_dict() for alert in self.alerts]
            }

            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            print(f"Report saved to: {output_file}\n")

    def get_health_score(self) -> float:
        """
        Calculate overall training health score (0-100)

        Returns:
            Health score where 100 is perfect
        """
        if not self.alerts:
            return 100.0

        # Deduct points based on severity
        deductions = {
            'CRITICAL': 25,
            'HIGH': 15,
            'MEDIUM': 5,
            'LOW': 2
        }

        total_deduction = 0
        for alert in self.alerts:
            total_deduction += deductions.get(alert.severity, 0)

        score = max(0, 100 - total_deduction)
        return score


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Detect edge cases in co-evolution training'
    )
    parser.add_argument(
        '--training-log',
        type=Path,
        default='logs/coevolution_training.json',
        help='Path to training log JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for detection report (JSON)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Custom configuration file (JSON)'
    )

    args = parser.parse_args()

    # Load custom config if provided
    config = None
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)

    # Create detector
    detector = CoEvolutionEdgeCaseDetector(config=config)

    # Analyze training log
    if not args.training_log.exists():
        print(f"❌ Training log not found: {args.training_log}")
        print(f"   Run co-evolution training first to generate log file")
        return 1

    alerts = detector.analyze_training_log(args.training_log)

    # Generate report
    detector.generate_report(args.output)

    # Calculate health score
    health_score = detector.get_health_score()
    print(f"Training Health Score: {health_score:.1f}/100")

    if health_score >= 90:
        print("✅ Excellent - Training is very healthy\n")
        return 0
    elif health_score >= 75:
        print("🟢 Good - Minor issues detected\n")
        return 0
    elif health_score >= 50:
        print("🟡 Fair - Significant issues need attention\n")
        return 1
    elif health_score >= 25:
        print("🟠 Poor - Major problems detected\n")
        return 1
    else:
        print("🔴 Critical - Training has severe issues\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
