"""Standalone project audit for AI_Supermarket_BI.

Run anytime with ``python audit.py`` — zero dependencies beyond the
standard library (pandas is NOT required). The script never imports
project modules; it only inspects files as text, so it works even when
the codebase is incomplete or broken.

Checks performed:
  1. File inventory (expected files per category + leftover ML files)
  2. Code completeness (placeholder scan, suspiciously short files)
  3. Forbidden ML imports (sklearn / prophet / xgboost / shap / ...)
  4. Forbidden kaleido / plotly.to_image usage
  5. Wrong data-file references (cleaned_sales.parquet etc.)
  6. Data & database state (generated files, processed artefacts, SQLite)
  7. Pipeline readiness (key entry points present per module)
  8. Dashboard page completeness

Every check is wrapped in try/except — the script must NEVER crash.
"""
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent

EXPECTED_FILES = {
    "Foundation": [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "setup.py",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
    ],
    "Source (src/)": [
        "src/__init__.py",
        "src/config.py",
        "src/logger.py",
        "src/data_generator.py",
        "src/data_validation.py",
        "src/preprocessing.py",
        "src/feature_engineering.py",
        "src/visualization.py",
        "src/analytics.py",
        "src/inventory.py",
        "src/ai_assistant.py",
        "src/report_generator.py",
    ],
    "Database": [
        "database/schema.sql",
    ],
    "Tests": [
        "tests/__init__.py",
        "tests/test_preprocessing.py",
        "tests/test_data_validation.py",
        "tests/test_analytics.py",
        "tests/test_inventory.py",
    ],
    "Dashboard": [
        "dashboard/app.py",
        "dashboard/pages/1_📊_Sales_Analytics.py",
        "dashboard/pages/2_👥_Customer_Analytics.py",
        "dashboard/pages/3_📦_Inventory.py",
        "dashboard/pages/4_📈_Trends.py",
        "dashboard/pages/5_🤖_AI_Assistant.py",
    ],
    "CI/CD": [
        ".github/workflows/ci.yml",
    ],
    "Notebook": [
        "notebooks/01_EDA.ipynb",
    ],
}

# Files that should NOT exist (old ML files from original plan)
FORBIDDEN_FILES = [
    "src/forecasting.py",
    "src/segmentation.py",
    "src/recommendation.py",
    "src/anomaly_detection.py",
    "src/evaluation.py",
]

FORBIDDEN_IMPORTS = [
    "sklearn", "scikit-learn", "from sklearn",
    "prophet", "fbprophet", "from prophet",
    "xgboost", "import xgb", "from xgboost",
    "mlxtend", "from mlxtend",
    "import shap", "from shap",
    "keras", "tensorflow", "torch",
]

FORBIDDEN_KALEIDO = [
    ".to_image(",
    "pio.to_image",
    "import kaleido",
    "from kaleido",
]

WRONG_DATA_PATHS = [
    "cleaned_sales.parquet",
]

CORRECT_DATA_PATHS = [
    "sales_processed.parquet",
    "sales_features.parquet",
    "supermarket_sales.csv",
]


# --- COLOR HELPERS (Windows-safe) ---
def enable_colors():
    """Enable ANSI colors on Windows and force UTF-8 stdout for emoji."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class C:
    """Color codes."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CHECK = "\033[92m✅\033[0m"
    CROSS = "\033[91m❌\033[0m"
    WARN = "\033[93m⚠️ \033[0m"
    INFO = "\033[94mℹ️ \033[0m"
    BLOCK = "\033[91m🚫\033[0m"
    READY = "\033[92m🟢\033[0m"
    PARTIAL = "\033[93m🟡\033[0m"


# --- CHECK FUNCTIONS ---

def check_file_inventory():
    """Check which expected files exist."""
    results = {}
    total = 0
    found = 0

    for category, files in EXPECTED_FILES.items():
        cat_results = []
        for f in files:
            total += 1
            path = PROJECT_ROOT / f
            exists = path.exists()
            if exists:
                found += 1
            cat_results.append((f, exists))
        results[category] = cat_results

    # Check for forbidden files
    forbidden_found = []
    for f in FORBIDDEN_FILES:
        if (PROJECT_ROOT / f).exists():
            forbidden_found.append(f)

    return results, found, total, forbidden_found


def check_code_completeness():
    """Scan existing .py files for placeholders."""
    issues = []
    placeholder_patterns = [
        (r'^\s*pass\s*$', "bare 'pass'"),
        (r'#\s*TODO', "TODO comment"),
        (r'#\s*FIXME', "FIXME comment"),
        (r'raise\s+NotImplementedError', "NotImplementedError"),
        (r'^\s*\.\.\.\s*$', "ellipsis placeholder"),
        (r'#\s*add\s+code\s+here', "placeholder comment"),
        (r'#\s*implement\b', "implement comment"),
    ]

    src_dir = PROJECT_ROOT / "src"
    if not src_dir.exists():
        return issues

    for py_file in src_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # Check for short files
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
            if len(code_lines) < 15:
                issues.append((str(py_file.relative_to(PROJECT_ROOT)),
                              f"Suspiciously short ({len(code_lines)} code lines)"))

            # Check for placeholders
            for i, line in enumerate(lines, 1):
                for pattern, desc in placeholder_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append((str(py_file.relative_to(PROJECT_ROOT)),
                                      f"Line {i}: {desc}"))
                        break  # one issue per line
        except Exception as e:
            issues.append((str(py_file), f"Read error: {e}"))

    return issues


def check_forbidden_imports():
    """Check for ML imports that shouldn't exist."""
    violations = []

    for search_dir in ["src", "dashboard", "tests"]:
        dir_path = PROJECT_ROOT / search_dir
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    line_lower = line.lower().strip()
                    if line_lower.startswith("#"):
                        continue
                    for forbidden in FORBIDDEN_IMPORTS:
                        if forbidden.lower() in line_lower:
                            violations.append((
                                str(py_file.relative_to(PROJECT_ROOT)),
                                i, line.strip(), forbidden
                            ))
            except Exception:
                pass

    return violations


def check_kaleido_usage():
    """Check for kaleido/plotly.to_image usage."""
    violations = []

    for search_dir in ["src", "dashboard"]:
        dir_path = PROJECT_ROOT / search_dir
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    if line.strip().startswith("#"):
                        continue
                    for pattern in FORBIDDEN_KALEIDO:
                        if pattern in line:
                            violations.append((
                                str(py_file.relative_to(PROJECT_ROOT)),
                                i, line.strip()
                            ))
            except Exception:
                pass

    return violations


def check_data_references():
    """Check for wrong data file paths in code."""
    issues = []

    for search_dir in ["src", "dashboard"]:
        dir_path = PROJECT_ROOT / search_dir
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    for wrong in WRONG_DATA_PATHS:
                        if wrong in line:
                            issues.append((
                                str(py_file.relative_to(PROJECT_ROOT)),
                                i, f"Wrong path '{wrong}' — use sales_processed.parquet"
                            ))
            except Exception:
                pass

    return issues


def check_correct_data_references():
    """Positive check: which correct data paths are referenced in code."""
    counts = {path: 0 for path in CORRECT_DATA_PATHS}
    for search_dir in ["src", "dashboard"]:
        dir_path = PROJECT_ROOT / search_dir
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for path in CORRECT_DATA_PATHS:
                    counts[path] += content.count(path)
            except Exception:
                pass
    return counts


def check_data_state():
    """Check if data files and database exist."""
    checks = {
        "data/augmented/supermarket_sales.csv": "Sales data",
        "data/augmented/products.csv": "Products data",
        "data/augmented/customers.csv": "Customers data",
        "data/processed/sales_processed.parquet": "Processed sales",
        "data/processed/sales_features.parquet": "Feature-enriched sales",
        "database/supermarket.db": "SQLite database",
    }

    results = {}
    for path, desc in checks.items():
        full = PROJECT_ROOT / path
        if full.exists():
            size = full.stat().st_size
            size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.0f} KB"
            results[path] = (True, size_str, desc)
        else:
            results[path] = (False, "missing", desc)

    return results


def check_pipeline_readiness():
    """Check if key entry points exist in source files."""
    checks = [
        ("src/data_generator.py", "generate_all", "Data generation"),
        ("src/preprocessing.py", "run_preprocessing_pipeline", "Preprocessing"),
        ("src/feature_engineering.py", "run_feature_engineering", "Feature engineering"),
        ("src/analytics.py", "run_full_analytics", "Analytics"),
        ("src/inventory.py", "run_inventory_pipeline", "Inventory"),
        ("src/ai_assistant.py", "BusinessAssistant", "AI Assistant"),
        ("src/report_generator.py", "generate_executive_report", "Report generation"),
        ("dashboard/app.py", "st.set_page_config", "Dashboard"),
    ]

    results = []
    for filepath, entry_point, desc in checks:
        full = PROJECT_ROOT / filepath
        if not full.exists():
            results.append((desc, "BLOCKED", f"{filepath} missing"))
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
            if entry_point in content:
                results.append((desc, "READY", f"{entry_point} found"))
            else:
                results.append((desc, "BLOCKED", f"{entry_point} not found in {filepath}"))
        except Exception as e:
            results.append((desc, "BLOCKED", str(e)))

    return results


def check_dashboard_pages():
    """Check dashboard page completeness."""
    pages_dir = PROJECT_ROOT / "dashboard" / "pages"
    expected = [
        "1_📊_Sales_Analytics.py",
        "2_👥_Customer_Analytics.py",
        "3_📦_Inventory.py",
        "4_📈_Trends.py",
        "5_🤖_AI_Assistant.py",
    ]

    results = []
    for page in expected:
        path = pages_dir / page
        if not path.exists():
            results.append((page, False, "Missing"))
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            has_title = "st.title" in content
            has_chart = "plotly_chart" in content or "st.metric" in content
            has_cache = "cache" in content
            issues = []
            if not has_title:
                issues.append("no st.title")
            if not has_chart:
                issues.append("no charts/metrics")
            if not has_cache:
                issues.append("no caching")
            status = "OK" if not issues else ", ".join(issues)
            results.append((page, True, status))
        except Exception as e:
            results.append((page, True, f"Error: {e}"))

    return results


# --- REPORT GENERATION ---

def print_header():
    print(f"\n{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🛒 AI Supermarket BI — Project Audit Report{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*65}{C.RESET}\n")


def print_inventory(results, found, total, forbidden):
    print(f"{C.BOLD}📁 CHECK 1: File Inventory ({found}/{total}){C.RESET}")
    print(f"{'─'*50}")

    for category, files in results.items():
        cat_found = sum(1 for _, exists in files if exists)
        cat_total = len(files)
        status = C.CHECK if cat_found == cat_total else (C.PARTIAL if cat_found > 0 else C.CROSS)
        print(f"  {status} {category}: {cat_found}/{cat_total}")

        for f, exists in files:
            icon = C.CHECK if exists else C.CROSS
            print(f"      {icon} {f}")

    if forbidden:
        print(f"\n  {C.WARN} Old ML files still present (safe to delete):")
        for f in forbidden:
            print(f"      {C.WARN} {f}")

    pct = int(found / total * 100) if total > 0 else 0
    print(f"\n  File completion: {C.BOLD}{pct}%{C.RESET} ({found}/{total})\n")
    return pct


def print_completeness(issues):
    print(f"{C.BOLD}📝 CHECK 2: Code Completeness{C.RESET}")
    print(f"{'─'*50}")

    if not issues:
        print(f"  {C.CHECK} No placeholders found — all code is complete\n")
    else:
        print(f"  {C.WARN} Found {len(issues)} issue(s):")
        for filepath, desc in issues[:10]:
            print(f"      {C.WARN} {filepath}: {desc}")
        if len(issues) > 10:
            print(f"      {C.DIM}... and {len(issues)-10} more{C.RESET}")
        print()


def print_ml_imports(violations):
    print(f"{C.BOLD}🚫 CHECK 3: No ML Imports{C.RESET}")
    print(f"{'─'*50}")

    if not violations:
        print(f"  {C.CHECK} No forbidden ML imports found\n")
    else:
        print(f"  {C.BLOCK} CRITICAL: {len(violations)} ML import(s) found:")
        for filepath, line, code, pkg in violations:
            print(f"      {C.BLOCK} {filepath}:{line} → '{pkg}'")
            print(f"         {C.DIM}{code}{C.RESET}")
        print()


def print_kaleido(violations):
    print(f"{C.BOLD}🖼️  CHECK 4: No Kaleido Usage{C.RESET}")
    print(f"{'─'*50}")

    if not violations:
        print(f"  {C.CHECK} No kaleido/to_image calls found\n")
    else:
        print(f"  {C.BLOCK} CRITICAL: {len(violations)} kaleido usage(s):")
        for filepath, line, code in violations:
            print(f"      {C.BLOCK} {filepath}:{line}")
            print(f"         {C.DIM}{code}{C.RESET}")
        print()


def print_data_refs(issues, correct_counts):
    print(f"{C.BOLD}📂 CHECK 5: Data Path References{C.RESET}")
    print(f"{'─'*50}")

    if not issues:
        print(f"  {C.CHECK} All data paths are correct")
    else:
        print(f"  {C.WARN} {len(issues)} wrong path reference(s):")
        for filepath, line, desc in issues:
            print(f"      {C.WARN} {filepath}:{line} — {desc}")

    used = [f"{p} ({n}x)" for p, n in correct_counts.items() if n > 0]
    if used:
        print(f"  {C.INFO} Correct paths referenced: {', '.join(used)}")
    print()


def print_data_state(results):
    print(f"{C.BOLD}💾 CHECK 6: Data & Database State{C.RESET}")
    print(f"{'─'*50}")

    for path, (exists, size, desc) in results.items():
        if exists:
            print(f"  {C.CHECK} {desc}: {path} ({size})")
        else:
            print(f"  {C.CROSS} {desc}: {path} ({C.RED}{size}{C.RESET})")
    print()


def print_pipeline(results):
    print(f"{C.BOLD}⚙️  CHECK 7: Pipeline Readiness{C.RESET}")
    print(f"{'─'*50}")

    ready = 0
    for desc, status, detail in results:
        if status == "READY":
            ready += 1
            print(f"  {C.READY} {desc}: {C.GREEN}{status}{C.RESET} ({detail})")
        else:
            print(f"  {C.BLOCK} {desc}: {C.RED}{status}{C.RESET} ({detail})")

    print(f"\n  Pipeline: {ready}/{len(results)} steps ready\n")
    return ready, len(results)


def print_dashboard(results):
    print(f"{C.BOLD}📊 CHECK 8: Dashboard Pages{C.RESET}")
    print(f"{'─'*50}")

    ok = 0
    for page, exists, status in results:
        if exists and status == "OK":
            ok += 1
            print(f"  {C.CHECK} {page}")
        elif exists:
            print(f"  {C.WARN} {page} ({status})")
        else:
            print(f"  {C.CROSS} {page}")

    print(f"\n  Dashboard: {ok}/{len(results)} pages complete\n")
    return ok, len(results)


def print_summary(file_pct, pipeline_ready, pipeline_total, dash_ok, dash_total,
                  critical_count, warning_count):
    # Calculate overall completion
    weights = {
        "files": 0.30,
        "pipeline": 0.35,
        "dashboard": 0.20,
        "clean": 0.15,
    }

    pipeline_pct = (pipeline_ready / pipeline_total * 100) if pipeline_total > 0 else 0
    dash_pct = (dash_ok / dash_total * 100) if dash_total > 0 else 0
    clean_pct = 100 if critical_count == 0 else max(0, 100 - critical_count * 25)

    overall = int(
        file_pct * weights["files"] +
        pipeline_pct * weights["pipeline"] +
        dash_pct * weights["dashboard"] +
        clean_pct * weights["clean"]
    )

    print(f"{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")
    print(f"{C.BOLD}  📊 OVERALL PROJECT COMPLETION: {overall}%{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")
    print()
    print(f"  Files:      {file_pct:>3}%  {'█' * (file_pct // 5)}{'░' * (20 - file_pct // 5)}")
    print(f"  Pipeline:   {int(pipeline_pct):>3}%  {'█' * (int(pipeline_pct) // 5)}{'░' * (20 - int(pipeline_pct) // 5)}")
    print(f"  Dashboard:  {int(dash_pct):>3}%  {'█' * (int(dash_pct) // 5)}{'░' * (20 - int(dash_pct) // 5)}")
    print(f"  Clean Code: {clean_pct:>3}%  {'█' * (clean_pct // 5)}{'░' * (20 - clean_pct // 5)}")
    print()

    if critical_count > 0:
        print(f"  {C.BLOCK} {critical_count} CRITICAL issue(s) — fix before running")
    if warning_count > 0:
        print(f"  {C.WARN} {warning_count} warning(s) — review when possible")
    if critical_count == 0 and warning_count == 0:
        print(f"  {C.CHECK} No issues found — project is clean!")

    print()

    if overall >= 90:
        print(f"  {C.GREEN}{C.BOLD}🏁 VERDICT: READY TO RUN{C.RESET}")
        print(f"  Run: make all  OR  streamlit run dashboard/app.py")
    elif overall >= 60:
        print(f"  {C.YELLOW}{C.BOLD}🔧 VERDICT: PARTIALLY COMPLETE{C.RESET}")
        print(f"  Complete remaining modules, then re-run this audit.")
    elif overall >= 30:
        print(f"  {C.YELLOW}{C.BOLD}🚧 VERDICT: IN PROGRESS{C.RESET}")
        print(f"  Continue with the next prompt in the series.")
    else:
        print(f"  {C.RED}{C.BOLD}🏗️  VERDICT: EARLY STAGE{C.RESET}")
        print(f"  Run Prompts A-F to build remaining modules.")

    print(f"\n{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")
    print(f"  Re-run anytime: {C.BOLD}python audit.py{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*65}{C.RESET}\n")


# --- MAIN ---

def main():
    enable_colors()
    print_header()

    try:
        # Run all checks
        inv_results, found, total, forbidden = check_file_inventory()
        completeness_issues = check_code_completeness()
        ml_violations = check_forbidden_imports()
        kaleido_violations = check_kaleido_usage()
        data_ref_issues = check_data_references()
        correct_counts = check_correct_data_references()
        data_state = check_data_state()
        pipeline_results = check_pipeline_readiness()
        dashboard_results = check_dashboard_pages()

        # Print reports
        file_pct = print_inventory(inv_results, found, total, forbidden)
        print_completeness(completeness_issues)
        print_ml_imports(ml_violations)
        print_kaleido(kaleido_violations)
        print_data_refs(data_ref_issues, correct_counts)
        print_data_state(data_state)
        p_ready, p_total = print_pipeline(pipeline_results)
        d_ok, d_total = print_dashboard(dashboard_results)

        # Count issues
        critical = len(ml_violations) + len(kaleido_violations)
        warnings = len(completeness_issues) + len(data_ref_issues) + len(forbidden)

        # Summary
        print_summary(file_pct, p_ready, p_total, d_ok, d_total, critical, warnings)
    except Exception as exc:
        # Last-resort guard: the audit must NEVER crash.
        print(f"\n{C.BLOCK} Audit failed unexpectedly: {exc!r}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
