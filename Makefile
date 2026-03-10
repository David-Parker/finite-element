.PHONY: dev build test clean

# Development server with hot reload
dev:
	trunk serve --open

# Build for production
build:
	trunk build --release

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean
	rm -rf dist pkg
