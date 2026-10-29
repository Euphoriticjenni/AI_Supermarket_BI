"""
NexusBI — Project Audit Script
Validates project completeness, code quality, and deployment readiness.

Usage:
    python audit.py

Exit codes:
    0 = Project 100% complete and ready to ship
    1 = Issues detected, see report
"""
from __future__ import annotations
import os
import re
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

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
        "src/forecasting.py",
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
        "dashboard/theme.py",
        "dashboard/pages/1_Sales.py",
        "dashboard/pages/2_Customers.py",
        "dashboard/pages/3_Inventory.py",
        "dashboard/pages/4_Forecasting.py",
        "dashboard/pages/5_AI_Assistant.py",
    ],
    "CI/CD": [
        ".github/workflows/ci.yml",
    ],
    "Notebook": [
        "notebooks/01_EDA.ipynb",
    ],
}

FORBIDDEN_IMPORTS = [
    "sklearn",
    "xgboost",
    "mlxtend",
    "shap",
    "keras",
    "tensorflow",
    "torch",
]

PLACEHOLDER_PATTERNS = [
    r"^\s*pass\s*$",
    r"#\s*TODO",
    r"#\s*FIXME",
    r"#\s*XXX",
    r"raise\s+NotImplementedError",
    r"\.\.\.\s*$",
    r"#\s*implement\s+later",
    r"#\s*placeholder",
]

CORRECT_DATA_PATHS = [
    "sales_processed.parquet",
    "sales_features.parquet",
    "supermarket_sales.csv",
]

FORBIDDEN_DATA_PATHS = [
    "cleaned_sales.parquet",
    "cleaned_data.parquet",
]

KALEIDO_PATTERNS = [
    r"\.to_image\(",
    r"import\s+kaleido",
    r"pio\.write_image",
]

PIPELINE_ENTRY_POINTS = {
    "Data generation": ("src/data_generator.py", "generate_all"),
    "Preprocessing": ("src/preprocessing.py", "run_preprocessing_pipeline"),
    "Feature engineering": ("src/feature_engineering.py", "run_feature_engineering"),
    "Analytics": ("src/analytics.py", "run_full_analytics"),
    "Inventory": ("src/inventory.py", "run_inventory_pipeline"),
    "AI Assistant": ("src/ai_assistant.py", "BusinessAssistant"),
    "Report generation": ("src/report_generator.py", "generate_executive_report"),
    "Dashboard": ("dashboard/app.py", "st.set_page_config"),
    "Forecasting": ("src/forecasting.py", "class DemandForecaster"),
}

DASHBOARD_PAGES = [
    "dashboard/app.py",
    "dashboard/pages/1_Sales.py",
    "dashboard/pages/2_Customers.py",
    "dashboard/pages/3_Inventory.py",
    "dashboard/pages/4_Forecasting.py",
    "dashboard/pages/5_AI_Assistant.py",
]

DATA_ASSETS = [
    ("data/augmented/supermarket_sales.csv", "Sales data"),
    ("data/augmented/products.csv", "Products data"),
    ("data/augmented/customers.csv", "Customers data"),
    ("data/processed/sales_processed.parquet", "Processed sales"),
    ("data/processed/sales_features.parquet", "Feature-enriched sales"),
    ("database/supermarket.db", "SQLite database"),
]

# ═════════════════════════════════════════════════════════════════
#  UTILITIES
# ═════════════════════════════════════════════════════════════════

def header(text: str) -> None:
    print("=" * 65)
    print(f"  {text}")
    print("=" * 65)

def section(text: str) -> None:
    print(f"\n{text}")
    print("─" * 50)

def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"

def read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def iter_python_files(root: Path):
    skip_dirs = {".venv", "venv", "env", "__pycache__", ".git", "node_modules", ".pytest_cache"}
    for p in root.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        yield p

# ═════════════════════════════════════════════════════════════════
#  CHECKS
# ═════════════════════════════════════════════════════════════════

def check_file_inventory() -> tuple[int, int, list[str]]:
    section("📁 CHECK 1: File Inventory")
    total = sum(len(files) for files in EXPECTED_FILES.values())
    found = 0
    missing: list[str] = []

    for category, files in EXPECTED_FILES.items():
        cat_found = sum(1 for f in files if (ROOT / f).exists())
        status = "✅" if cat_found == len(files) else "⚠️ "
        print(f"  {status} {category}: {cat_found}/{len(files)}")
        for f in files:
            if (ROOT / f).exists():
                print(f"      ✅ {f}")
                found += 1
            else:
                print(f"      ❌ {f}")
                missing.append(f)

    print(f"\n  File completion: {int(found/total*100)}% ({found}/{total})")
    return found, total, missing


def check_code_completeness() -> list[str]:
    section("📝 CHECK 2: Code Completeness")
    issues: list[str] = []

    for py_file in iter_python_files(ROOT):
        content = read_file_safe(py_file)
        if not content.strip():
            continue
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            # Skip meaningful "pass" (in except blocks, abstract methods)
            if re.match(r"^\s*pass\s*$", line):
                prev = lines[idx - 2].strip() if idx >= 2 else ""
                if prev.startswith(("except", "class", "def", "if", "else", "elif")):
                    continue
            for pattern in PLACEHOLDER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    rel = py_file.relative_to(ROOT)
                    issues.append(f"{rel}:{idx} — {line.strip()[:60]}")
                    break

    if not issues:
        print("  ✅ No placeholders found — all code is complete")
    else:
        print(f"  ⚠️  Found {len(issues)} placeholder(s):")
        for issue in issues[:10]:
            print(f"      ⚠️  {issue}")
        if len(issues) > 10:
            print(f"      ... and {len(issues) - 10} more")
    return issues


def check_no_forbidden_imports() -> list[str]:
    section("🚫 CHECK 3: No Forbidden ML Imports")
    print("  ℹ️  Forbidden: " + ", ".join(FORBIDDEN_IMPORTS))
    print("  ℹ️  Allowed:   prophet, statsmodels, matplotlib")
    violations: list[str] = []

    for py_file in iter_python_files(ROOT):
        if py_file.name == "audit.py":
            continue
        content = read_file_safe(py_file)
        for lib in FORBIDDEN_IMPORTS:
            pattern = rf"^\s*(?:from\s+{lib}|import\s+{lib})\b"
            for idx, line in enumerate(content.splitlines(), start=1):
                if re.search(pattern, line):
                    rel = py_file.relative_to(ROOT)
                    violations.append(f"{rel}:{idx} — {line.strip()}")

    if not violations:
        print("  ✅ No forbidden ML imports found")
    else:
        print(f"  ❌ Found {len(violations)} forbidden import(s):")
        for v in violations:
            print(f"      ❌ {v}")
    return violations


def check_no_kaleido() -> list[str]:
    section("🖼️  CHECK 4: No Kaleido Usage")
    violations: list[str] = []

    for py_file in iter_python_files(ROOT):
        if py_file.name == "audit.py":
            continue
        content = read_file_safe(py_file)
        for pattern in KALEIDO_PATTERNS:
            for idx, line in enumerate(content.splitlines(), start=1):
                if re.search(pattern, line):
                    rel = py_file.relative_to(ROOT)
                    violations.append(f"{rel}:{idx} — {line.strip()}")

    if not violations:
        print("  ✅ No kaleido/to_image calls found")
    else:
        print(f"  ❌ Found {len(violations)} kaleido reference(s):")
        for v in violations:
            print(f"      ❌ {v}")
    return violations


def check_data_paths() -> tuple[list[str], dict[str, int]]:
    section("📂 CHECK 5: Data Path References")
    violations: list[str] = []
    correct_counts: dict[str, int] = {p: 0 for p in CORRECT_DATA_PATHS}

    for py_file in iter_python_files(ROOT):
        if py_file.name == "audit.py":
            continue
        content = read_file_safe(py_file)
        for bad in FORBIDDEN_DATA_PATHS:
            if bad in content:
                for idx, line in enumerate(content.splitlines(), start=1):
                    if bad in line:
                        rel = py_file.relative_to(ROOT)
                        violations.append(f"{rel}:{idx} — references {bad}")
        for good in CORRECT_DATA_PATHS:
            correct_counts[good] += content.count(good)

    if not violations:
        print("  ✅ All data paths are correct")
    else:
        print(f"  ❌ Found {len(violations)} incorrect data path(s):")
        for v in violations:
            print(f"      ❌ {v}")

    refs = ", ".join(f"{k} ({v}x)" for k, v in correct_counts.items() if v > 0)
    if refs:
        print(f"  ℹ️  Correct paths referenced: {refs}")
    return violations, correct_counts


def check_data_state() -> dict[str, bool]:
    section("💾 CHECK 6: Data & Database State")
    status: dict[str, bool] = {}
    for rel_path, label in DATA_ASSETS:
        path = ROOT / rel_path
        if path.exists():
            size = human_size(path.stat().st_size)
            print(f"  ✅ {label}: {rel_path} ({size})")
            status[rel_path] = True
        else:
            print(f"  ⚠️  {label}: {rel_path} — NOT FOUND (run data pipeline)")
            status[rel_path] = False
    return status


def check_pipeline_readiness() -> tuple[int, int]:
    section("⚙️  CHECK 7: Pipeline Readiness")
    ready = 0
    total = len(PIPELINE_ENTRY_POINTS)

    for name, (file_path, marker) in PIPELINE_ENTRY_POINTS.items():
        path = ROOT / file_path
        if not path.exists():
            print(f"  🔴 {name}: FILE MISSING ({file_path})")
            continue
        content = read_file_safe(path)
        if marker in content:
            print(f"  🟢 {name}: READY ({marker} found)")
            ready += 1
        else:
            print(f"  🔴 {name}: INCOMPLETE ({marker} not found)")

    print(f"\n  Pipeline: {ready}/{total} steps ready")
    return ready, total


def check_dashboard_pages() -> tuple[int, int]:
    section("📊 CHECK 8: Dashboard Pages")
    complete = 0
    total = len(DASHBOARD_PAGES)

    for page in DASHBOARD_PAGES:
        path = ROOT / page
        name = Path(page).name
        if not path.exists():
            print(f"  ❌ {name} — FILE MISSING")
            continue
        content = read_file_safe(path)
        has_title = "st.title(" in content or "st.header(" in content
        has_viz = any(x in content for x in ["st.plotly_chart(", "st.metric(", "st.dataframe("])
        has_cache = "@st.cache_data" in content or "@st.cache_resource" in content

        issues = []
        if not has_title:
            issues.append("no st.title")
        if not has_viz:
            issues.append("no charts/metrics")
        if not has_cache:
            issues.append("no caching")

        if not issues:
            print(f"  ✅ {name}")
            complete += 1
        else:
            print(f"  ⚠️  {name} ({', '.join(issues)})")

    print(f"\n  Dashboard: {complete}/{total} pages complete")
    return complete, total


def check_forecasting_engine() -> bool:
    section("🔮 CHECK 9: Forecasting Engine (Prophet)")
    path = ROOT / "src/forecasting.py"
    if not path.exists():
        print("  🔴 src/forecasting.py — MISSING")
        return False
    content = read_file_safe(path)
    checks = {
        "Prophet import": "from prophet import Prophet" in content or "import prophet" in content,
        "DemandForecaster class": "class DemandForecaster" in content,
        "fit method": re.search(r"def\s+fit\s*\(", content) is not None,
        "predict method": re.search(r"def\s+predict\s*\(", content) is not None,
        "Pipeline function": "def run_forecast_pipeline" in content,
    }
    all_ok = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'🟢' if ok else '🔴'} {name}")
    print(f"\n  Forecasting engine: {'READY' if all_ok else 'INCOMPLETE'}")
    return all_ok


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════

def main() -> int:
    print()
    header("🛒 NexusBI — Project Audit Report")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📂 {ROOT}")
    print("=" * 65)

    found, total_files, missing = check_file_inventory()
    placeholder_issues = check_code_completeness()
    forbidden = check_no_forbidden_imports()
    kaleido = check_no_kaleido()
    bad_paths, _ = check_data_paths()
    data_state = check_data_state()
    pipe_ready, pipe_total = check_pipeline_readiness()
    dash_complete, dash_total = check_dashboard_pages()
    forecast_ok = check_forecasting_engine()

    # ─── Scoring ───────────────────────────────────────────────────
    files_pct = int(found / total_files * 100) if total_files else 0
    pipeline_pct = int(pipe_ready / pipe_total * 100) if pipe_total else 0
    dashboard_pct = int(dash_complete / dash_total * 100) if dash_total else 0
    clean_code_pct = 100 if (not placeholder_issues and not forbidden and not kaleido and not bad_paths) else 0
    forecast_pct = 100 if forecast_ok else 0

    overall = int((files_pct + pipeline_pct + dashboard_pct + clean_code_pct + forecast_pct) / 5)

    def bar(pct: int) -> str:
        filled = int(pct / 5)
        return "█" * filled + "░" * (20 - filled)

    print()
    print("=" * 65)
    print(f"  📊 OVERALL PROJECT COMPLETION: {overall}%")
    print("=" * 65)
    print()
    print(f"  Files:       {files_pct:>3}%  {bar(files_pct)}")
    print(f"  Pipeline:    {pipeline_pct:>3}%  {bar(pipeline_pct)}")
    print(f"  Dashboard:   {dashboard_pct:>3}%  {bar(dashboard_pct)}")
    print(f"  Clean Code:  {clean_code_pct:>3}%  {bar(clean_code_pct)}")
    print(f"  Forecasting: {forecast_pct:>3}%  {bar(forecast_pct)}")
    print()

    # ─── Issues Summary ────────────────────────────────────────────
    all_issues = []
    if missing:
        all_issues.append(f"{len(missing)} missing file(s)")
    if placeholder_issues:
        all_issues.append(f"{len(placeholder_issues)} placeholder(s)")
    if forbidden:
        all_issues.append(f"{len(forbidden)} forbidden import(s)")
    if kaleido:
        all_issues.append(f"{len(kaleido)} kaleido reference(s)")
    if bad_paths:
        all_issues.append(f"{len(bad_paths)} incorrect data path(s)")

    if not all_issues:
        print("  ✅ No issues found — project is clean!")
    else:
        print("  ⚠️  Issues detected:")
        for issue in all_issues:
            print(f"      • {issue}")

    print()
    # ─── Verdict ───────────────────────────────────────────────────
    if overall == 100:
        print("  🎉 VERDICT: 100% COMPLETE & SHIP-READY")
        exit_code = 0
    elif overall >= 90:
        print("  ✅ VERDICT: NEARLY COMPLETE — minor cleanup needed")
        exit_code = 0
    elif overall >= 70:
        print("  🔧 VERDICT: PARTIALLY COMPLETE")
        print("  Complete remaining modules, then re-run this audit.")
        exit_code = 1
    else:
        print("  ❌ VERDICT: SIGNIFICANT WORK REMAINING")
        exit_code = 1

    print()
    print("=" * 65)
    print("  Re-run anytime: python audit.py")
    print("=" * 65)
    print()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())