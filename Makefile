.PHONY: dev build test clean

# Development server with hot reload
dev:
	cd crates/fem-web && trunk serve --open

# Build for production
build:
	cd crates/fem-web && trunk build --release

# Run tests
test:
	cargo test -p fem-core

# Run all tests
test-all:
	cargo test

# Clean build artifacts
clean:
	cargo clean
	rm -rf crates/fem-web/dist
