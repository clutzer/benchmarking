.PHONY: setup
setup:
	@echo "NGC_API_KEY=$$(cat ~/.secrets/ngc-api-key.txt)" > .env
	@cat ~/.secrets/ngc-api-key.txt | docker login nvcr.io -u '$$oauthtoken' --password-stdin

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

.PHONY: list-inference-server-models
list-inference-server-models:
	@docker compose exec benchmark-server /benchmark-scripts/list-inference-server-models

.PHONY: run-baseline-benchmark
run-baseline-benchmark:
	@docker compose exec benchmark-server /benchmark-scripts/baseline.sh

