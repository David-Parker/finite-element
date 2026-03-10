# Finite Element Method (FEM) Soft Body Simulation

This document describes the mathematical model and equations used in this 2D soft body simulation.

## Overview

The simulation uses a **Neo-Hookean hyperelastic material model** with explicit time integration. This model is well-suited for large deformations of soft materials like rubber, jello, and biological tissues.

## Material Parameters

### Input Parameters

- **Young's Modulus (E)**: Stiffness of the material (Pa). Higher values = stiffer material.
- **Poisson's Ratio (ν)**: Incompressibility (0 to 0.5). Values near 0.5 = nearly incompressible.
- **Density (ρ)**: Mass per unit area (kg/m²).
- **Damping**: Velocity damping factor per frame.

### Lamé Parameters

The input parameters are converted to Lamé parameters for the constitutive model:

```
μ = E / (2(1 + ν))           # Shear modulus
λ = Eν / ((1 + ν)(1 - 2ν))   # First Lamé parameter
```

## Deformation Gradient

For each triangle element, we compute the **deformation gradient F**, which describes how the material has deformed from its rest state.

### Rest Shape Matrix (Dm)

For a triangle with vertices X₀, X₁, X₂ in the rest configuration:

```
Dm = [X₁ - X₀ | X₂ - X₀]
```

This 2x2 matrix is computed once at initialization and its inverse (Dm⁻¹) is cached.

### Deformed Shape Matrix (Ds)

For current vertex positions x₀, x₁, x₂:

```
Ds = [x₁ - x₀ | x₂ - x₀]
```

### Deformation Gradient

```
F = Ds · Dm⁻¹
```

The determinant J = det(F) represents the volume ratio (area ratio in 2D):
- J = 1: No volume change
- J > 1: Expansion
- J < 1: Compression
- J ≤ 0: Inverted element (numerical instability)

## Neo-Hookean Model

### Strain Energy Density

The Neo-Hookean strain energy density function:

```
Ψ = (μ/2)(tr(FᵀF) - 2) - μ log(J) + (λ/2)(log(J))²
```

Where:
- `tr(FᵀF)` is the squared Frobenius norm of F
- The term `- μ log(J)` penalizes volume change
- The term `(λ/2)(log(J))²` provides additional volume preservation

At rest (F = I, J = 1), the energy is zero.

### First Piola-Kirchhoff Stress

The stress tensor P is derived from the energy:

```
P = μF + (λ log(J) - μ) F⁻ᵀ
```

Where F⁻ᵀ is the inverse transpose of F.

At rest (F = I, J = 1), the stress is zero (no internal forces).

## Force Computation

### Elastic Forces

Forces on triangle vertices are computed from the stress:

```
H = -Area · P · Dm⁻ᵀ
```

The columns of H give forces on vertices 1 and 2. Force on vertex 0 is computed to maintain equilibrium:

```
f₀ = -f₁ - f₂
```

This ensures momentum conservation (forces sum to zero).

### Gravity

Applied as:

```
f_gravity = m · g
```

Where m is the vertex mass (distributed from triangle areas) and g is gravitational acceleration.

## Time Integration

### Semi-Implicit Euler

The simulation uses semi-implicit (symplectic) Euler integration:

```
v(t+dt) = v(t) + (f/m) · dt
x(t+dt) = x(t) + v(t+dt) · dt
```

Note: velocity is updated first, then position uses the new velocity. This provides better energy stability than explicit Euler.

### Substeps

Each frame is divided into multiple substeps (default: 32) for stability:

```
dt_substep = (1/60) / num_substeps
```

Smaller timesteps are needed for stiffer materials to prevent numerical explosion.

### Damping

Velocity damping is applied once per frame (not per substep):

```
v = v · (1 - damping_factor)
```

## Mesh Topology

### Ring Mesh

The ring (annulus) mesh is generated with:
- Concentric rings of vertices from inner to outer radius
- Proper wrap-around connectivity (no seam discontinuity)
- Triangle pairs forming quads between adjacent rings

### Mass Distribution

Triangle mass is distributed equally to its three vertices:

```
vertex_mass += (triangle_area · density) / 3
```

## Numerical Stability

### J Clamping

To prevent numerical issues when elements become inverted:

```
safe_J = max(J, 0.1)
```

This prevents log(J) from going to -infinity.

### Material Selection

Stiffer materials require either:
- More substeps (smaller dt)
- Lower stiffness values
- Implicit integration (not implemented)

Typical stable ranges for 32 substeps at 60 FPS:
- Rubber/Jello: E = 1e5 to 5e5
- Wood: E = 1e6
- Metal: May require 64+ substeps

## References

1. Sifakis, E., & Barbic, J. (2012). FEM Simulation of 3D Deformable Solids. SIGGRAPH Course Notes.
2. Bonet, J., & Wood, R. D. (2008). Nonlinear Continuum Mechanics for Finite Element Analysis.
3. Smith, B., et al. (2018). Stable Neo-Hookean Flesh Simulation. ACM TOG.
