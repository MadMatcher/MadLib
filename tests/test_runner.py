#!/usr/bin/env python3
"""
Test runner script for MadMatcher test suite.

This script provides convenient commands for running different types of tests
and generating coverage reports.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MadMatcher Test Runner")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "performance", "coverage", "quick"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true", 
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel:
        base_cmd.extend(["-n", "auto"])
    
    if args.fail_fast:
        base_cmd.append("-x")
    
    success = True
    
    if args.test_type == "unit":
        cmd = base_cmd + [
            "tests/unit/",
            "-m", "unit",
            "--tb=short"
        ]
        success = run_command(cmd, "Unit Tests")
    
    elif args.test_type == "integration":
        cmd = base_cmd + [
            "tests/integration/",
            "-m", "integration",
            "--tb=short"
        ]
        success = run_command(cmd, "Integration Tests")
    
    elif args.test_type == "performance":
        cmd = base_cmd + [
            "tests/performance/",
            "-m", "performance",
            "--tb=short"
        ]
        success = run_command(cmd, "Performance Tests")
    
    elif args.test_type == "quick":
        cmd = base_cmd + [
            "tests/unit/",
            "tests/integration/",
            "-m", "not slow",
            "--tb=line"
        ]
        success = run_command(cmd, "Quick Tests (excluding slow tests)")
    
    elif args.test_type == "coverage":
        cmd = base_cmd + [
            "tests/unit/",
            "tests/integration/",
            "--cov=MadLib",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ]
        if args.html_report:
            cmd.append("--cov-report=html")
        
        success = run_command(cmd, "Coverage Tests")
        
        if success and args.html_report:
            print("\nHTML coverage report generated in htmlcov/index.html")
    
    elif args.test_type == "all":
        # Run all test types sequentially
        tests = [
            (["tests/unit/", "-m", "unit"], "Unit Tests"),
            (["tests/integration/", "-m", "integration"], "Integration Tests"),
            (["tests/performance/", "-m", "performance"], "Performance Tests")
        ]
        
        for test_args, description in tests:
            cmd = base_cmd + test_args + ["--tb=short"]
            if not run_command(cmd, description):
                success = False
                if args.fail_fast:
                    break
        
        # Generate final coverage report
        if success:
            coverage_cmd = base_cmd + [
                "tests/unit/",
                "tests/integration/",
                "--cov=MadLib",
                "--cov-report=term-missing",
                "--cov-report=xml",
                "--cov-fail-under=80"
            ]
            if args.html_report:
                coverage_cmd.append("--cov-report=html")
            
            run_command(coverage_cmd, "Final Coverage Report")
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 