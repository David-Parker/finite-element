# 2D FEM Soft Body Simulation

A real-time soft body physics simulation using the Finite Element Method (FEM) with Neo-Hookean hyperelastic materials, written in Rust and compiled to WebAssembly.

![FEM Soft Body Simulation](media/fem.gif)

## Features

- **FEM-based physics**: Triangular mesh elements with proper force accumulation
- **Neo-Hookean material model**: Handles large deformations correctly
- **Multiple materials**: Rubber, jello, wood, metal with distinct behaviors
- **Real-time**: 60 FPS with 64 physics substeps per frame
- **WebAssembly**: Runs in any modern browser
- **Portable core**: Physics library has no platform dependencies

## Building

Requires [Rust](https://rustup.rs/) and [Trunk](https://trunkrs.dev/):

```bash
# Install trunk
cargo install trunk

# Build and serve
cd crates/fem-web
trunk serve
```

Then open http://localhost:8080 in your browser.

### Release Build

```bash
trunk build --release
```

Output will be in the `dist/` directory.

## Controls

| Key   | Action                                 |
|-------|----------------------------------------|
| Space | Pause/Resume                           |
| R     | Reset simulation                       |
| T     | Toggle tracing (downloads CSV on stop) |

## Project Structure

```
crates/
  fem-core/           # Portable physics library (no WASM dependencies)
    src/
      lib.rs          # Library exports
      softbody.rs     # SoftBody struct, materials, physics stepping
      fem.rs          # FEM: deformation gradient, stress, forces
      mesh.rs         # Mesh generation (ring, square)
      math.rs         # 2x2 matrix and 2D vector operations
      trace.rs        # Simulation tracing/debugging
  fem-web/            # WebAssembly application
    src/
      lib.rs          # WASM entry point, simulation loop
      renderer.rs     # WebGL rendering
      trace.rs        # CSV export for browser
    index.html        # Web page
    style.css         # Styling
```

## Materials

```rust
Material::RUBBER  // Soft, bouncy, fully elastic
Material::JELLO   // Very soft, bouncy, fully elastic
Material::WOOD    // Medium stiffness, some plasticity
Material::METAL   // Stiff, maintains shape under stress
```

## Documentation

See [docs/fem-model.md](docs/fem-model.md) for the mathematical model:

- Finite Element Method discretization
- Neo-Hookean constitutive model
- Deformation gradient and stress computation
- Time integration and stability

## License

MIT
