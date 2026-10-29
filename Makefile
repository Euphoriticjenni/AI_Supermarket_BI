# ============================================================
# AI_Supermarket_BI — Makefile
# Run `make` targets from the project root.
# ============================================================

.PHONY: install generate-data preprocess features analytics test lint \
        dashboard docker-build docker-run docker-stop report status clean all

install:
	pip install -r requirements.txt
	python -c "from src.config import ensure_directories; ensure_directories()"

generate-data:
	python -c "from src.data_generator import generate_all; generate_all()"

preprocess:
	python -c "from src.preprocessing import run_preprocessing_pipeline; run_preprocessing_pipeline()"

features:
	python -c "import pandas as pd; from src.feature_engineering import run_feature_engineering; df = pd.read_parquet('data/processed/sales_processed.parquet'); run_feature_engineering(df)"

analytics:
	python -c "import pandas as pd; from src.analytics import run_full_analytics; df = pd.read_parquet('data/processed/sales_features.parquet'); run_full_analytics(df)"

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

dashboard:
	streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d
	@echo "Dashboard available at http://localhost:8501"

docker-stop:
	docker-compose down

report:
	python -c "from src.report_generator import generate_executive_report; generate_executive_report()"

status:
	@echo "=== AI_Supermarket_BI project status ==="
	@echo "--- data/augmented/ ---"
	@ls -lh data/augmented/ 2>/dev/null || echo "(empty - run: make generate-data)"
	@echo "--- data/processed/ ---"
	@ls -lh data/processed/ 2>/dev/null || echo "(empty - run: make preprocess)"
	@echo "--- database/ ---"
	@ls -lh database/ 2>/dev/null || echo "(no database - run: make generate-data)"
	@echo "--- reports/ ---"
	@ls -lh reports/ 2>/dev/null || echo "(no reports - run: make report)"

clean:
	rm -rf data/processed/*
	rm -rf logs/*
	rm -rf reports/*.pdf reports/_charts
	rm -rf .pytest_cache
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned processed data, logs, reports, and caches."
	@echo "(augmented data and the SQLite database are kept - remove manually if needed)"

all: install generate-data preprocess features analytics test dashboard
