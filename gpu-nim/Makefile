.PHONY: setup
setup:
	@echo "NGC_API_KEY=$$(cat ~/.secrets/ngc-api-key.txt)" > .env
	#@echo "HF_TOKEN=$$(cat ~/.secrets/hf-token.txt)" >> .env
	@cat ~/.secrets/ngc-api-key.txt | docker login nvcr.io -u '$$oauthtoken' --password-stdin
	@-useradd -m -s `which bash` nvidia
	@mkdir -p nim-cache
	@chown nvidia:nvidia nim-cache
	@chmod g+ws nim-cache
	@-mkdir benchmark-artifacts

.PHONY: status
status:
	@docker compose ps --format "{{.Service}}+{{.Status}}" | awk -F'+' '{split($$1, svc, ":"); print svc[1] " " $$2}'

.PHONY: up
up:
	docker compose up -d inference-server benchmark-server

.PHONY: stop
stop:
	docker compose stop inference-server benchmark-server

.PHONY: getting-started
getting-started: setup up status

.PHONY: list-model-profiles
list-model-profiles:
	@docker compose up list-model-profiles

.PHONY: list-models
list-models:
	@docker compose exec benchmark-server /benchmark-scripts/list-inference-server-models

.PHONY: truncate-logs
truncate-logs:
	truncate -s 0 `docker inspect --format='{{.LogPath}}' inference-server`

#
# Benchmarks
#

.PHONY: run-baseline-benchmark
run-baseline-benchmark:
	@time docker compose exec benchmark-server /benchmark-scripts/baseline.sh
	(cd reports; ./collate_benchmarks.py --artifact-dir ../benchmark-artifacts/baseline/)

.PHONY: run-smoketest-benchmark
run-smoketest-benchmark:
	@time docker compose exec benchmark-server /benchmark-scripts/smoketest.sh
	(cd reports; ./collate_benchmarks.py --artifact-dir ../benchmark-artifacts/smoketest/)

