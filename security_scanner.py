#!/usr/bin/env python3
"""
ACEAC Security Scanner - Automated Security Audit Tool
Scans codebase for common security vulnerabilities in adversarial RL testbeds

Author: @sarowarzahan414
Date: 2025-11-18
Usage: python security_scanner.py --path . --report security_report.json
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: str = ""

    def to_dict(self):
        return asdict(self)


class SecurityScanner:
    """Automated security scanner for adversarial RL testbeds"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0
        self.lines_scanned = 0

    def scan(self) -> List[SecurityIssue]:
        """Run all security scans"""
        print(f"\n{'='*70}")
        print("ACEAC SECURITY SCANNER")
        print(f"{'='*70}")
        print(f"Scanning: {self.base_path.absolute()}")
        print(f"Started: {datetime.now().isoformat()}")
        print(f"{'='*70}\n")

        # Find all Python files
        python_files = list(self.base_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files to scan\n")

        # Run all checks
        for py_file in python_files:
            if self._should_skip(py_file):
                continue

            print(f"Scanning: {py_file.relative_to(self.base_path)}")
            self.files_scanned += 1
            self._scan_file(py_file)

        print(f"\n{'='*70}")
        print(f"Scan complete!")
        print(f"Files scanned: {self.files_scanned}")
        print(f"Lines scanned: {self.lines_scanned}")
        print(f"Issues found: {len(self.issues)}")
        print(f"{'='*70}\n")

        return self.issues

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            "__pycache__",
            ".git",
            "venv",
            "env",
            ".pyc",
            "test_",  # Skip test files for now
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _scan_file(self, file_path: Path):
        """Scan a single Python file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.lines_scanned += len(lines)

            # Run all checks on this file
            self._check_pickle_usage(file_path, lines)
            self._check_bare_except(file_path, lines)
            self._check_division_by_zero(file_path, lines)
            self._check_path_traversal(file_path, lines)
            self._check_hardcoded_secrets(file_path, lines)
            self._check_dangerous_functions(file_path, lines)
            self._check_weak_randomness(file_path, lines)
            self._check_sql_injection(file_path, lines)
            self._check_command_injection(file_path, lines)
            self._check_insecure_deserialization(file_path, lines)

        except Exception as e:
            print(f"  Error scanning {file_path}: {e}")

    def _check_pickle_usage(self, file_path: Path, lines: List[str]):
        """Check for unsafe pickle usage (CWE-502)"""
        patterns = [
            (r'pickle\.load\(', 'Unsafe pickle.load() usage'),
            (r'pickle\.loads\(', 'Unsafe pickle.loads() usage'),
            (r'cPickle\.load\(', 'Unsafe cPickle.load() usage'),
            (r'PPO\.load\(', 'PPO.load() uses pickle internally'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line):
                    self.issues.append(SecurityIssue(
                        severity="CRITICAL",
                        category="Deserialization",
                        description=f"{description} - allows arbitrary code execution",
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use SafeTensors, ONNX, or implement HMAC verification",
                        cwe_id="CWE-502"
                    ))

    def _check_bare_except(self, file_path: Path, lines: List[str]):
        """Check for bare except clauses (CWE-396)"""
        for i, line in enumerate(lines, 1):
            # Match "except:" but not "except Exception:" or "except ValueError:"
            if re.match(r'^\s*except\s*:\s*$', line):
                self.issues.append(SecurityIssue(
                    severity="MEDIUM",
                    category="Error Handling",
                    description="Bare except clause catches all exceptions including system exit",
                    file_path=str(file_path.relative_to(self.base_path)),
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation="Use specific exception types (e.g., except ValueError:)",
                    cwe_id="CWE-396"
                ))

    def _check_division_by_zero(self, file_path: Path, lines: List[str]):
        """Check for potential division by zero (CWE-369)"""
        # Look for divisions without obvious checks
        division_pattern = r'(\w+)\s*/\s*(\w+)'

        for i, line in enumerate(lines, 1):
            matches = re.finditer(division_pattern, line)
            for match in matches:
                denominator = match.group(2)

                # Check if there's a zero check nearby
                context_start = max(0, i - 3)
                context_end = min(len(lines), i + 2)
                context = ''.join(lines[context_start:context_end])

                # If no check found, flag it
                if f'if {denominator}' not in context and f'{denominator} >' not in context:
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Numeric Safety",
                        description=f"Potential division by zero: {denominator} not checked",
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation=f"Add check: if {denominator} > 0: ... else: ...",
                        cwe_id="CWE-369"
                    ))

    def _check_path_traversal(self, file_path: Path, lines: List[str]):
        """Check for path traversal vulnerabilities (CWE-22)"""
        unsafe_patterns = [
            (r'open\([^)]*\+', 'String concatenation in file path'),
            (r'os\.path\.join\([^,]+,\s*\w+\s*\)', 'User input in path without validation'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in unsafe_patterns:
                if re.search(pattern, line) and 'resolve()' not in line and 'safe_path' not in line:
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Path Traversal",
                        description=f"{description} - may allow directory traversal",
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use Path.resolve() and validate against base directory",
                        cwe_id="CWE-22"
                    ))

    def _check_hardcoded_secrets(self, file_path: Path, lines: List[str]):
        """Check for hardcoded secrets (CWE-798)"""
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', 'Hardcoded token'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's a test file or example
                    if 'example' in line.lower() or 'test' in line.lower():
                        continue

                    self.issues.append(SecurityIssue(
                        severity="CRITICAL",
                        category="Credential Management",
                        description=f"{description} found in source code",
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet="[REDACTED]",  # Don't include actual secret
                        recommendation="Use environment variables or secret management service",
                        cwe_id="CWE-798"
                    ))

    def _check_dangerous_functions(self, file_path: Path, lines: List[str]):
        """Check for dangerous function usage"""
        dangerous_patterns = [
            (r'eval\(', 'eval() allows arbitrary code execution', 'CRITICAL', 'CWE-95'),
            (r'exec\(', 'exec() allows arbitrary code execution', 'CRITICAL', 'CWE-95'),
            (r'os\.system\(', 'os.system() vulnerable to command injection', 'HIGH', 'CWE-78'),
            (r'subprocess\.call\([^,]+,\s*shell=True', 'shell=True enables command injection', 'HIGH', 'CWE-78'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description, severity, cwe in dangerous_patterns:
                if re.search(pattern, line):
                    self.issues.append(SecurityIssue(
                        severity=severity,
                        category="Dangerous Functions",
                        description=description,
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use safer alternatives or implement strict input validation",
                        cwe_id=cwe
                    ))

    def _check_weak_randomness(self, file_path: Path, lines: List[str]):
        """Check for weak randomness in security contexts (CWE-338)"""
        # Look for random usage in security-related contexts
        security_keywords = ['token', 'key', 'secret', 'password', 'salt', 'nonce']

        for i, line in enumerate(lines, 1):
            if 'random.random' in line or 'random.randint' in line:
                # Check if used in security context
                context_start = max(0, i - 3)
                context_end = min(len(lines), i + 3)
                context = ''.join(lines[context_start:context_end]).lower()

                if any(keyword in context for keyword in security_keywords):
                    self.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        category="Cryptography",
                        description="Weak randomness used in security context",
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use secrets.SystemRandom() for security-critical randomness",
                        cwe_id="CWE-338"
                    ))

    def _check_sql_injection(self, file_path: Path, lines: List[str]):
        """Check for SQL injection vulnerabilities (CWE-89)"""
        sql_patterns = [
            (r'execute\(["\'].*%s.*["\'].*%', 'String formatting in SQL query'),
            (r'execute\(["\'].*\+.*["\']', 'String concatenation in SQL query'),
            (r'execute\(f["\']', 'f-string in SQL query'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in sql_patterns:
                if re.search(pattern, line):
                    self.issues.append(SecurityIssue(
                        severity="CRITICAL",
                        category="Injection",
                        description=f"SQL injection: {description}",
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use parameterized queries (?, ?)",
                        cwe_id="CWE-89"
                    ))

    def _check_command_injection(self, file_path: Path, lines: List[str]):
        """Check for command injection (CWE-78)"""
        for i, line in enumerate(lines, 1):
            if 'subprocess' in line and ('shell=True' in line or '+' in line or 'f"' in line):
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="Injection",
                    description="Command injection: User input in shell command",
                    file_path=str(file_path.relative_to(self.base_path)),
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation="Use shell=False and pass arguments as list",
                    cwe_id="CWE-78"
                ))

    def _check_insecure_deserialization(self, file_path: Path, lines: List[str]):
        """Check for insecure deserialization beyond pickle (CWE-502)"""
        patterns = [
            (r'yaml\.load\([^,)]+\)', 'yaml.load() without safe loader'),
            (r'marshal\.load\(', 'marshal.load() can execute code'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line) and 'safe_load' not in line.lower():
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Deserialization",
                        description=description,
                        file_path=str(file_path.relative_to(self.base_path)),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use yaml.safe_load() or safer alternatives",
                        cwe_id="CWE-502"
                    ))

    def generate_report(self, output_file: str = None):
        """Generate security report"""
        # Group by severity
        by_severity = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for issue in self.issues:
            by_severity[issue.severity].append(issue)

        # Print summary
        print(f"\n{'='*70}")
        print("SECURITY SCAN RESULTS")
        print(f"{'='*70}")
        print(f"\nTotal Issues: {len(self.issues)}")
        print(f"  🔴 CRITICAL: {len(by_severity['CRITICAL'])}")
        print(f"  🟠 HIGH:     {len(by_severity['HIGH'])}")
        print(f"  🟡 MEDIUM:   {len(by_severity['MEDIUM'])}")
        print(f"  🟢 LOW:      {len(by_severity['LOW'])}")

        # Print issues by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            issues = by_severity[severity]
            if not issues:
                continue

            emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
            print(f"\n{emoji} {severity} Issues ({len(issues)}):")
            print("="*70)

            for issue in issues:
                print(f"\n  [{issue.category}] {issue.description}")
                print(f"  File: {issue.file_path}:{issue.line_number}")
                print(f"  Code: {issue.code_snippet[:80]}")
                print(f"  Fix:  {issue.recommendation}")
                if issue.cwe_id:
                    print(f"  CWE:  {issue.cwe_id}")

        # Calculate security score
        score = self._calculate_score()
        print(f"\n{'='*70}")
        print(f"SECURITY SCORE: {score:.1f}/100")
        print(f"{'='*70}")

        if score >= 90:
            print("✅ Excellent - Well secured")
        elif score >= 75:
            print("🟢 Good - Minor improvements needed")
        elif score >= 50:
            print("🟡 Fair - Significant improvements needed")
        elif score >= 25:
            print("🟠 Poor - Major security gaps")
        else:
            print("🔴 Critical - Immediate action required")

        print(f"\n{'='*70}\n")

        # Save to file
        if output_file:
            report_data = {
                'scan_date': datetime.now().isoformat(),
                'base_path': str(self.base_path),
                'files_scanned': self.files_scanned,
                'lines_scanned': self.lines_scanned,
                'total_issues': len(self.issues),
                'by_severity': {k: len(v) for k, v in by_severity.items()},
                'security_score': score,
                'issues': [issue.to_dict() for issue in self.issues]
            }

            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            print(f"Report saved to: {output_file}")

    def _calculate_score(self) -> float:
        """Calculate security score (0-100)"""
        # Deduct points based on severity
        deductions = {
            'CRITICAL': 20,
            'HIGH': 10,
            'MEDIUM': 5,
            'LOW': 2
        }

        total_deduction = 0
        for issue in self.issues:
            total_deduction += deductions.get(issue.severity, 0)

        # Score starts at 100, capped at 0
        score = max(0, 100 - total_deduction)
        return score


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Security scanner for adversarial RL testbeds"
    )
    parser.add_argument(
        '--path',
        default='.',
        help='Path to scan (default: current directory)'
    )
    parser.add_argument(
        '--report',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--fail-on',
        choices=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
        help='Exit with error if issues of this severity or higher are found'
    )

    args = parser.parse_args()

    # Run scanner
    scanner = SecurityScanner(args.path)
    issues = scanner.scan()
    scanner.generate_report(args.report)

    # Check fail condition
    if args.fail_on:
        severity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        min_level = severity_levels.index(args.fail_on)

        has_critical_issues = any(
            severity_levels.index(issue.severity) >= min_level
            for issue in issues
        )

        if has_critical_issues:
            print(f"\n❌ Found issues at {args.fail_on} level or higher")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
