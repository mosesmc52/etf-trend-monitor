PROJECT_NAME := etf_trend_monitor
COMPOSE_FILE := docker-compose.yml
DO_FN_DIR ?= infra/do-functions
DO_FN_ENV ?= .env
DO_FN_NAME ?= launcher/etf-trend-monitor
DROPLET_USER ?= root
DROPLET_LOG_FILE ?= /var/log/job.log
SPACES_ENDPOINT ?=
SPACES_BUCKET ?=

.PHONY: help build up upd shell logs restart stop down clean \
	do-fn-validate do-fn-connect do-fn-status do-fn-deploy do-fn-deploy-remote \
	do-fn-list do-fn-get do-fn-invoke do-fn-activations do-fn-logs \
	do-droplet-log do-spaces-log

help:
	@echo "Available targets:"
	@echo "  build                Build the Docker image"
	@echo "  up                   Start the app in the foreground"
	@echo "  upd                  Start the app in daemon mode"
	@echo "  shell                Open a shell in the running container"
	@echo "  logs                 Tail container logs"
	@echo "  restart              Restart the Docker service"
	@echo "  stop                 Stop the Docker service"
	@echo "  down                 Stop and remove the Docker service"
	@echo "  clean                Remove the app container and image"
	@echo "  do-fn-validate       Validate DO Functions project metadata"
	@echo "  do-fn-connect        Connect doctl to a DO Functions namespace"
	@echo "  do-fn-status         Show DO Functions connection status"
	@echo "  do-fn-deploy         Deploy infra/do-functions with runtime env"
	@echo "  do-fn-deploy-remote  Deploy infra/do-functions using remote build"
	@echo "  do-fn-list           List deployed DO functions"
	@echo "  do-fn-get            Show deployed function metadata"
	@echo "  do-fn-invoke         Invoke launcher/etf-trend-monitor"
	@echo "  do-fn-activations    List recent activations"
	@echo "  do-fn-logs           Show logs for ACTIVATION=<id>"
	@echo "  do-droplet-log       Tail droplet log over SSH with DROPLET_IP=<ip>"
	@echo "  do-spaces-log        Print uploaded log from Spaces with LOG_KEY=<key>"

build:
	docker-compose -f $(COMPOSE_FILE) build

up:
	docker-compose -f $(COMPOSE_FILE) up

upd: ## Up in daemon mode
	docker-compose -f $(COMPOSE_FILE) up -d

shell:
	docker exec -it $(PROJECT_NAME) /bin/bash

logs:
	docker logs -f $(PROJECT_NAME)

restart:
	docker-compose -f $(COMPOSE_FILE) restart

stop:
	docker-compose -f $(COMPOSE_FILE) stop

down:
	docker-compose -f $(COMPOSE_FILE) down

clean:
	docker-compose -f $(COMPOSE_FILE) rm -sfv etf_trend_monitor
	docker image rm etf-trend-monitor_etf_trend_monitor || true

do-fn-validate:
	doctl serverless get-metadata $(DO_FN_DIR)

do-fn-connect:
	doctl serverless connect

do-fn-status:
	doctl serverless status

do-fn-deploy:
	doctl serverless deploy $(DO_FN_DIR) --env $(DO_FN_ENV)

do-fn-deploy-remote:
	doctl serverless deploy $(DO_FN_DIR) --env $(DO_FN_ENV) --remote-build

do-fn-list:
	doctl serverless functions list

do-fn-get:
	doctl serverless functions get $(DO_FN_NAME)

do-fn-invoke:
	doctl serverless functions invoke $(DO_FN_NAME)

do-fn-activations:
	doctl serverless activations list

do-fn-logs:
	test -n "$(ACTIVATION)" || (echo "Set ACTIVATION=<id>" && exit 1)
	doctl serverless activations logs $(ACTIVATION)

do-droplet-log:
	test -n "$(DROPLET_IP)" || (echo "Set DROPLET_IP=<ip>" && exit 1)
	ssh $(DROPLET_USER)@$(DROPLET_IP) "sudo tail -f $(DROPLET_LOG_FILE)"

do-spaces-log:
	test -n "$(LOG_KEY)" || (echo "Set LOG_KEY=<spaces log key>" && exit 1)
	aws --endpoint-url $(SPACES_ENDPOINT) s3 cp s3://$(SPACES_BUCKET)/$(LOG_KEY) -
