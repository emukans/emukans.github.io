# Use Homebrew's Ruby 3.4 when available (Chirpy needs Ruby 3.x; macOS ships 2.6)
RUBY_BIN := /opt/homebrew/opt/ruby@3.4/bin
ifneq ($(wildcard $(RUBY_BIN)),)
  export PATH := $(RUBY_BIN):$(PATH)
  BUNDLE := $(RUBY_BIN)/bundle
else
  BUNDLE := bundle
endif
export LANG := en_US.UTF-8
export LC_ALL := en_US.UTF-8

PORT ?= 4000

.DEFAULT_GOAL := help
.PHONY: help install update serve drafts build test clean post draft publish favicons

help: ## Show available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install gems into vendor/bundle
	$(BUNDLE) config set --local path vendor/bundle
	$(BUNDLE) install

update: ## Update gems to the latest allowed versions (Chirpy, Jekyll, ...)
	$(BUNDLE) update

serve: ## Start the dev server with live reload (PORT=4000)
	$(BUNDLE) exec jekyll serve --livereload --future --port $(PORT)

drafts: ## Start the dev server including _drafts
	$(BUNDLE) exec jekyll serve --livereload --drafts --port $(PORT)

build: ## Build the production site into _site
	JEKYLL_ENV=production $(BUNDLE) exec jekyll build

test: build ## Build and check internal links/images, as CI does
	$(BUNDLE) exec htmlproofer _site \
		--disable-external \
		--ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"

favicons: ## Copy icons to the site root, where Safari and iOS look for them
	cp assets/img/favicons/favicon.ico favicon.ico
	cp assets/img/favicons/apple-touch-icon.png apple-touch-icon.png
	cp assets/img/favicons/apple-touch-icon.png apple-touch-icon-precomposed.png

clean: ## Remove build output and caches
	$(BUNDLE) exec jekyll clean

post: ## Create a new post: make post title="My title"
	@test -n "$(title)" || (echo 'Usage: make post title="My title"' && exit 1)
	$(BUNDLE) exec jekyll post "$(title)"

draft: ## Create a new draft: make draft title="My title"
	@test -n "$(title)" || (echo 'Usage: make draft title="My title"' && exit 1)
	$(BUNDLE) exec jekyll draft "$(title)"

publish: ## Publish a draft: make publish file=_drafts/my-title.md
	@test -n "$(file)" || (echo 'Usage: make publish file=_drafts/my-title.md' && exit 1)
	$(BUNDLE) exec jekyll publish "$(file)"
