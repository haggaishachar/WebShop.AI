.PHONY: setup install-uv sync-deps install-spacy-model update-spacy-model check-uv check-search-engine run-dev run-prod run-web-agent-site run-web-agent-text run-web-agent-human run-web-agent-paper-rule run-web-agent-simple-rule run-web-agent-llm run-web-agent-custom setup-data-small setup-data-all setup-human-trajs download-spacy-model-lg setup-search-engine test clean

# Default number of episodes for web agent runs
NUM_EPISODES ?= 100

setup: install-uv sync-deps install-spacy-model
	@echo "✓ Complete setup finished!"

check-uv:
	@which uv > /dev/null 2>&1 || (echo "UV is not installed. Run 'make install-uv' first."; exit 1)
	@echo "✓ UV is installed"

check-search-engine:
	@if [ ! -d "search_engine/indexes" ] || [ -z "$$(ls -A search_engine/indexes 2>/dev/null | grep -v README)" ]; then \
		echo "⚠️  Search engine indexes not found!"; \
		echo ""; \
		echo "Before running the application, you need to:"; \
		echo "  1. Download data: make setup-data-small (or setup-data-all)"; \
		echo "  2. Build indexes: make setup-search-engine"; \
		echo ""; \
		exit 1; \
	fi
	@echo "✓ Search engine indexes found"

install-uv:
	@echo "Checking for UV installation..."
	@if command -v uv > /dev/null 2>&1; then \
		echo "✓ UV is already installed: $$(uv --version)"; \
	else \
		echo "Installing UV..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✓ UV installed successfully!"; \
		echo "Note: You may need to restart your shell or run: source ~/.bashrc (or ~/.zshrc)"; \
	fi

sync-deps: check-uv
	@echo "Syncing dependencies with UV..."
	@uv sync
	@echo "✓ Dependencies synced successfully!"

install-spacy-model: check-uv
	@echo "Installing spaCy language model (en_core_web_sm)..."
	@uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
	@echo "✓ Model installed successfully!"
	@uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ Model verification passed!')"

update-spacy-model: check-uv
	@echo "Updating spaCy language model to latest version..."
	@uv pip install --upgrade https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
	@echo "✓ Model updated successfully!"
	@uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ Model verification passed!')"

run-dev: check-search-engine
	@echo "Starting Flask app in development mode..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so FLASK_ENV=development uv run python -m web_agent_site.app --log --attrs

run-prod: check-search-engine
	@echo "Starting Flask app in production mode..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python -m web_agent_site.app --log

run-web-agent-site: check-search-engine
	@echo "Starting web agent site environment ($(NUM_EPISODES) episodes)..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py --use-site-env --observation-mode text --policy random --num-episodes $(NUM_EPISODES)

run-web-agent-text: check-search-engine
	@echo "Starting web agent text environment ($(NUM_EPISODES) episodes)..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py --observation-mode text --policy random --num-episodes $(NUM_EPISODES)

run-web-agent-human: check-search-engine
	@echo "Starting web agent text environment with human policy ($(NUM_EPISODES) episodes)..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py --observation-mode text --policy human --num-episodes $(NUM_EPISODES)

run-web-agent-paper-rule: check-search-engine
	@echo "Starting web agent text environment with paper rule-based policy ($(NUM_EPISODES) episodes)..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py --observation-mode text --policy paper_rule --num-episodes $(NUM_EPISODES)

run-web-agent-simple-rule: check-search-engine
	@echo "Starting web agent text environment with simple rule-based policy ($(NUM_EPISODES) episodes)..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py --observation-mode text --policy simple_rule --num-episodes $(NUM_EPISODES)

run-web-agent-llm: check-search-engine
	@echo "Starting web agent text environment with LLM policy ($(NUM_EPISODES) episodes)..."
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py --observation-mode text --policy llm --num-episodes $(NUM_EPISODES)

run-web-agent-custom:
	@echo "Starting web agent with custom parameters..."
	@echo "Usage: make run-web-agent-custom ARGS='--observation-mode text --num-products 100 --policy random --num-episodes 3'"
	@echo "Or override NUM_EPISODES: make run-web-agent-custom NUM_EPISODES=10 ARGS='--observation-mode text --policy random'"
	@JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so uv run python run_envs/run_web_agent_env.py $(ARGS)

setup-data-small: check-uv
	@echo "Downloading small dataset (1000 items)..."
	@mkdir -p data
	@cd data && uv run gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
	@cd data && uv run gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
	@cd data && uv run gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
	@echo "✓ Small dataset downloaded successfully!"

setup-data-all: check-uv
	@echo "Downloading full dataset..."
	@mkdir -p data
	@cd data && uv run gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB
	@cd data && uv run gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi
	@cd data && uv run gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
	@echo "✓ Full dataset downloaded successfully!"

setup-human-trajs: check-uv
	@echo "Downloading human trajectories..."
	@mkdir -p user_session_logs
	@cd user_session_logs && uv run python -c "import gdown; gdown.download_folder('https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto', quiet=True, remaining_ok=True)"
	@echo "✓ Human trajectories downloaded successfully!"

download-spacy-model-lg: check-uv
	@echo "Downloading spaCy large model (en_core_web_lg)..."
	@uv run python -m spacy download en_core_web_lg
	@echo "✓ spaCy large model installed successfully!"

setup-search-engine: check-uv
	@echo "Setting up search engine..."
	@cd search_engine && mkdir -p resources resources_100 resources_1k resources_100k
	@cd search_engine && uv run python convert_product_file_format.py
	@cd search_engine && mkdir -p indexes
	@cd search_engine && ./run_indexing.sh
	@echo "✓ Search engine setup complete!"

test: check-uv
	@echo "Running tests with pytest..."
	@uv run pytest tests/ -v
	@echo "✓ Tests completed!"

clean:
	@echo "Cleaning up temporary files..."
	@rm -rf .temp_model
	@rm -f en_core_web_sm-*.whl
	@echo "✓ Cleanup complete!"
