#!/usr/bin/env python3
"""
Generate test summary from JUnit XML files and create GitHub step summary.
"""

import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

from junitparser import JUnitXml


def parse_test_results():
    """Parse all JUnit XML files and extract test results."""
    results = defaultdict(lambda: defaultdict(dict))

    # Find all XML files in test-results directory
    xml_files = glob.glob("test-results/*.xml")

    if not xml_files:
        print("No test result files found!")
        return results

    for xml_file in xml_files:
        print(f"Processing {xml_file}")

        # Extract OS and Python version from filename
        filename = Path(xml_file).stem
        if filename.startswith('test-results-'):
            parts = filename.replace('test-results-', '').split('-')
            if len(parts) >= 2:
                os_name = parts[0]
                python_version = parts[1]
                test_type = "unit"
        elif filename.startswith('notebook-test-results-'):
            parts = filename.replace('notebook-test-results-', '').split('-')
            if len(parts) >= 2:
                os_name = parts[0]
                python_version = parts[1]
                test_type = "notebook"
        else:
            continue

        try:
            xml = JUnitXml.fromfile(xml_file)

            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            skipped_tests = 0
            errors = 0

            for suite in xml:
                total_tests += suite.tests
                passed_tests += suite.tests - suite.failures - suite.errors - suite.skipped
                failed_tests += suite.failures
                errors += suite.errors
                skipped_tests += suite.skipped

            results[f"{os_name}-{python_version}"][test_type] = {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': errors,
                'skipped': skipped_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }

        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue

    return results


def generate_markdown_table(results):
    """Generate a markdown table from test results."""
    if not results:
        return "No test results available."

    # Table header
    table = "| Platform | Python | Unit Tests | Notebook Tests | Overall Status |\n"
    table += "|----------|--------|------------|----------------|----------------|\n"

    for platform_python, test_types in sorted(results.items()):
        os_name, python_version = platform_python.split('-', 1)

        # Format OS name
        os_display = {
            'ubuntu-latest': '🐧 Ubuntu',
            'macos-latest': '🍎 macOS',
            'windows-latest': '🪟 Windows'
        }.get(os_name, os_name)

        # Unit tests summary
        unit_summary = "N/A"
        if 'unit' in test_types:
            unit = test_types['unit']
            unit_summary = f"✅ {unit['passed']}/{unit['total']} ({unit['success_rate']:.1f}%)"
            if unit['failed'] > 0 or unit['errors'] > 0:
                unit_summary = f"❌ {unit['passed']}/{unit['total']} ({unit['success_rate']:.1f}%)"

        # Notebook tests summary
        notebook_summary = "N/A"
        if 'notebook' in test_types:
            notebook = test_types['notebook']
            notebook_summary = f"✅ {notebook['passed']}/{notebook['total']} ({notebook['success_rate']:.1f}%)"
            if notebook['failed'] > 0 or notebook['errors'] > 0:
                notebook_summary = f"❌ {notebook['passed']}/{notebook['total']} ({notebook['success_rate']:.1f}%)"

        # Overall status
        overall_status = "✅ Pass"
        if ('unit' in test_types and (test_types['unit']['failed'] > 0 or test_types['unit']['errors'] > 0)) or \
           ('notebook' in test_types and (test_types['notebook']['failed'] > 0 or test_types['notebook']['errors'] > 0)):
            overall_status = "❌ Fail"

        table += f"| {os_display} | {python_version} | {unit_summary} | {notebook_summary} | {overall_status} |\n"

    return table


def main():
    """Main function to generate test summary."""
    print("Generating test summary...")

    results = parse_test_results()

    if not results:
        print("No test results found!")
        sys.exit(1)

    # Generate markdown table
    table = generate_markdown_table(results)

    # Write test summary for GitHub step summary
    summary_content = f"""# 🧪 Test Results Summary

{table}

## Test Details

"""

    # Add detailed results
    for platform_python, test_types in sorted(results.items()):
        os_name, python_version = platform_python.split('-', 1)
        summary_content += f"### {os_name} - Python {python_version}\n\n"

        for test_type, stats in test_types.items():
            summary_content += f"**{test_type.title()} Tests:**\n"
            summary_content += f"- Total: {stats['total']}\n"
            summary_content += f"- Passed: {stats['passed']}\n"
            summary_content += f"- Failed: {stats['failed']}\n"
            summary_content += f"- Errors: {stats['errors']}\n"
            summary_content += f"- Skipped: {stats['skipped']}\n"
            summary_content += f"- Success Rate: {stats['success_rate']:.1f}%\n\n"

    # Write to files
    with open('test_summary.md', 'w') as f:
        f.write(summary_content)

    # Write just the table for README updates
    with open('test_results_table.md', 'w') as f:
        f.write(table)

    # Write to GitHub step summary if in GitHub Actions
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(summary_content)

    print("Test summary generated successfully!")
    print(f"Table preview:\n{table}")


if __name__ == "__main__":
    main()
