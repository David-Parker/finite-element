# 2D Soft Body Simulation

A real-time soft body physics simulation using **XPBD (Extended Position-Based Dynamics)** for unconditionally stable simulation, written in Rust and compiled to WebAssembly.

<table>
  <tr>
    <td><img src="media/xpbd.gif" alt="XPBD Simulation" width="400"/></td>
    <td><img src="media/25.gif" alt="25 Bodies" width="400"/></td>
  </tr>
  <tr>
    <td><img src="media/collisions.gif" alt="Collisions" width="400"/></td>
    <td><img src="media/shapes.gif" alt="Mixed Shapes" width="400"/></td>
  </tr>
  <tr>
    <td><img src="media/fem.gif" alt="FEM Simulation" width="400"/></td>
  </tr>
</table>

## Features

- **XPBD physics**: Position-based constraint solving with compliance for implicit-like stability
- **Unconditionally stable**: No explosions regardless of material stiffness or timestep
- **Shape preservation**: Zero-compliance edge constraints maintain mesh structure
- **Multiple materials**: Jello, rubber, wood, metal with distinct behaviors
- **Real-time**: 60 FPS with only 4 physics substeps per frame
- **Vertex-edge collisions**: Accurate collision detection using spatial hashing
- **Irregular meshes**: Supports rings, ellipses, stars, and blob shapes
- **Mouse interaction**: Click and drag to attract bodies
- **WebAssembly**: Runs in any modern browser
- **Portable core**: Physics library has no platform dependencies

## Algorithm

This simulation uses **XPBD (Extended Position-Based Dynamics)** rather than traditional force-based FEM. Key advantages:

| Aspect | Force-based FEM | XPBD |
|--------|-----------------|------|
| Stability | Requires tiny timesteps | Unconditionally stable |
| Substeps needed | 64-128 | 4-8 |
| Stiffness handling | Limited by timestep | Any stiffness via compliance |
| Implementation | Complex stress tensors | Simple position constraints |

### XPBD Algorithm (per substep)

```
1. Pre-solve: Apply gravity, predict positions
2. Solve constraints (5 iterations):
   - Edge constraints: Maintain rest lengths between connected vertices
   - Area constraints: Preserve triangle areas (volume in 2D)
3. Handle collisions: Ground and inter-body
4. Post-solve: Derive velocities from position change
```

### Constraint Formula

```
λ = -C / (Σ wᵢ|∇Cᵢ|² + α/Δt²)
```

Where:
- `C` is the constraint violation
- `wᵢ` is inverse mass
- `∇Cᵢ` is the constraint gradient
- `α` is compliance (inverse stiffness)

**Zero compliance** = infinitely stiff (rigid edges for shape preservation)

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
| 1-5   | Switch materials                       |

## Project Structure

```
crates/
  fem-core/           # Portable physics library (no WASM dependencies)
    src/
      lib.rs          # Library exports
      xpbd.rs         # XPBD solver: constraints, collision, integration
      mesh.rs         # Mesh generation (ring, square)
      math.rs         # 2x2 matrix and 2D vector operations
      compute.rs      # SIMD-accelerated compute backends
      trace.rs        # Simulation tracing/profiling
  fem-web/            # WebAssembly application
    src/
      lib.rs          # WASM entry point, simulation loop
      renderer.rs     # WebGL rendering
    index.html        # Web page
```

## Materials

Materials are defined by constraint compliance (lower = stiffer):

| Material | Edge Compliance | Area Compliance | Behavior |
|----------|-----------------|-----------------|----------|
| Jello    | 0 (rigid)       | 1e-6            | Soft, jiggly |
| Rubber   | 0 (rigid)       | 1e-7            | Bouncy |
| Wood     | 0 (rigid)       | 1e-8            | Stiff |
| Metal    | 0 (rigid)       | 0 (rigid)       | Perfectly rigid |

All materials use **zero edge compliance** to prevent shape collapse (pancaking).

## Documentation

See [docs/fem-model.md](docs/fem-model.md) for detailed documentation:

- XPBD algorithm and constraint solving
- Edge and area constraint mathematics
- Collision handling with spatial hashing
- Material parameters and compliance

## References

1. Macklin, M., et al. (2016). XPBD: Position-Based Simulation of Compliant Constrained Dynamics. Motion in Games.
2. Müller, M., et al. (2007). Position Based Dynamics. Journal of Visual Communication and Image Representation.
3. Ten Minute Physics - XPBD tutorials

## License

MIT
