# Finite Element Method (FEM) Soft Body Simulation

This document describes the mathematical model and equations used in this 2D soft body simulation.

## Overview

This simulation combines two key components:

1. **Finite Element Method (FEM)**: The spatial discretization approach
2. **Neo-Hookean Hyperelastic Model**: The constitutive law (material behavior)

Together, these provide a physically-based simulation suitable for large deformations of soft materials like rubber, jello, and biological tissues.

## Finite Element Method (FEM)

FEM is a numerical technique for solving partial differential equations by discretizing a continuous domain into smaller "finite elements."

### Why FEM?

Instead of solving the equations of elasticity over a continuous body (impossible analytically for complex shapes), we:

1. **Discretize**: Divide the body into simple elements (triangles in 2D, tetrahedra in 3D)
2. **Approximate**: Assume deformation varies linearly within each element
3. **Compute locally**: Calculate forces for each element independently
4. **Assemble globally**: Combine element contributions at shared nodes

### Elements and Nodes

- **Elements**: The triangles that make up the mesh. Each element has its own deformation state.
- **Nodes**: The vertices where elements connect. Forces from adjacent elements accumulate here.

### FEM Workflow (Per Timestep)

```
For each element (triangle):
    1. Compute deformation gradient F from current node positions
    2. Evaluate constitutive model (Neo-Hookean) to get stress P
    3. Convert stress to nodal forces
    4. Accumulate forces at shared nodes

For each node (vertex):
    1. Sum forces from all adjacent elements
    2. Add external forces (gravity)
    3. Integrate velocity and position
```

This separation of local (element) and global (node) computation is the essence of FEM.

## Constitutive Model: Neo-Hookean Hyperelasticity

The constitutive model defines the relationship between deformation and internal stress. We use the **Neo-Hookean hyperelastic model**, which is:

- **Hyperelastic**: Stress derived from a strain energy function (path-independent, energy-conserving)
- **Neo-Hookean**: A specific energy function suitable for large deformations

This model is well-suited for rubber-like materials and provides stable behavior under large stretching and compression.

## Material Parameters (Constitutive Model Inputs)

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

## Deformation Gradient (FEM Kinematics)

For each triangle element, we compute the **deformation gradient F**, which describes how the material has deformed from its rest state. This is the key quantity that connects the FEM discretization to the constitutive model.

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

## Neo-Hookean Model (Constitutive Equations)

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

## Force Computation (FEM Assembly)

This is where FEM and the constitutive model come together: stress from Neo-Hookean is converted to nodal forces using FEM shape function derivatives.

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

Each frame is divided into multiple substeps (default: 128) for stability:

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

## Plasticity (Permanent Deformation)

For materials like wood, deformation can become permanent when stress exceeds a threshold. This is modeled using **multiplicative plasticity** with a plastic deformation gradient.

### Elastic vs Plastic Materials

- **Elastic (rubber, jello, metal)**: Always returns to rest shape. `yield_stress = 0`
- **Plastic (wood)**: Permanent deformation when overstressed. `yield_stress > 0`

Note: Metal uses high damping instead of plasticity for stability reasons (see Limitations below).

### Material Parameters

- **yield_stress**: Stress threshold before plastic flow begins (Pa)
- **plasticity**: Rate at which deformation becomes permanent (0-1)

### Multiplicative Plasticity Model

We use a multiplicative decomposition of the deformation gradient, common in finite strain plasticity:

```
F_total = F_elastic · F_plastic
```

Each triangle tracks a plastic deformation matrix `Fp` (initialized to identity).

#### Algorithm

1. Compute total deformation gradient:
   ```
   F_total = Ds · Dm⁻¹
   ```

2. Extract elastic deformation by removing plastic part:
   ```
   F_elastic = F_total · Fp⁻¹
   ```

3. Compute stress from elastic deformation only (Neo-Hookean):
   ```
   P = μ·F_elastic + (λ·log(J_elastic) - μ)·F_elastic⁻ᵀ
   ```

4. Check yield condition (Frobenius norm of stress):
   ```
   if ||P|| > yield_stress:
       # Plastic flow - update Fp towards F_total
       Fp_new = lerp(Fp, F_total, plasticity · yield_ratio · rate)
   ```

5. Apply conservative limits to prevent instability:
   - Only yield when triangle is healthy: `0.7 < J_total < 1.5`
   - Limit plastic deformation: `0.85 < det(Fp) < 1.15`
   - Very slow plastic flow rate: `rate = 0.02`

### Effect on Behavior

| Material | yield_stress | plasticity | Behavior |
|----------|-------------|------------|----------|
| Rubber   | 0           | 0          | Soft, bouncy, always recovers |
| Jello    | 0           | 0          | Very soft, jiggly |
| Wood     | 2e4         | 0.2        | Can bend/deform permanently |
| Metal    | 0           | 0          | Stiff, high damping (no bounce) |

### Limitations

The multiplicative plasticity model has stability constraints:

1. **High-stress impacts**: During violent collisions, triangles can deform faster than plastic flow can accommodate, leading to numerical instability. For this reason, metal uses high damping instead of plasticity.

2. **Conservative limits**: The tight bounds on `det(Fp)` (0.85-1.15) limit the amount of permanent deformation possible, but are necessary for stability.

3. **No fracture**: The model allows plastic deformation but not material separation/tearing.

## Numerical Stability

### Compression Barrier

To prevent triangle inversion, a barrier force is added when triangles become compressed:

```
if J < 0.8:
    compression = 0.8 - J
    barrier = compression³ · 500 · μ
    # Barrier is added to stress, pushing triangle back towards healthy state
```

This cubic scaling provides aggressive resistance as J approaches zero.

### J Clamping

As a fallback, J is clamped when computing stress:

```
safe_J = max(J, 0.4)
```

This prevents log(J) from going to -infinity.

### Velocity and Acceleration Limits

Safety limits prevent numerical explosion:

```
MAX_VELOCITY = 30.0
MAX_ACCEL = 5000.0
```

### Substep Requirements

The simulation uses 128 substeps per frame (dt ≈ 0.00013s) for stability with stiffer materials.

Typical stable material stiffness ranges:
- Jello: E = 5e4
- Rubber: E = 1e5
- Wood: E = 2e5
- Metal: E = 4e5

### Stability Testing

The codebase includes end-to-end simulation tests (`simulation_tests.rs`) that verify:
- KE stays below 100,000
- Velocity stays below 60
- J stays between 0.1 and 10.0
- All materials survive 10-second simulations

## References

1. Sifakis, E., & Barbic, J. (2012). FEM Simulation of 3D Deformable Solids. SIGGRAPH Course Notes.
2. Bonet, J., & Wood, R. D. (2008). Nonlinear Continuum Mechanics for Finite Element Analysis.
3. Smith, B., et al. (2018). Stable Neo-Hookean Flesh Simulation. ACM TOG.
