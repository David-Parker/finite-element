# XPBD Soft Body Simulation

This document describes the physics model and algorithms used in this 2D soft body simulation.

## Overview

This simulation uses **XPBD (Extended Position-Based Dynamics)** for soft body physics. XPBD is a position-based method that provides:

- **Unconditional stability**: No numerical explosions regardless of stiffness or timestep
- **Compliance-based materials**: Stiffness controlled by compliance parameter (inverse stiffness)
- **Implicit-like behavior**: Stability of implicit methods with simplicity of explicit methods

## Why XPBD over Force-Based FEM?

Traditional force-based FEM computes internal forces from stress tensors and integrates velocities explicitly. This requires:
- Very small timesteps (128+ substeps per frame)
- Complex strain limiting for stability
- Careful material parameter tuning

XPBD instead works directly on positions via constraints:
- Only 4-8 substeps needed
- Any stiffness achievable via compliance
- Simpler implementation

| Aspect | Force-based FEM | XPBD |
|--------|-----------------|------|
| Substeps @ 60Hz | 64-128 | 4-8 |
| Stiffness limit | Bounded by CFL condition | Unlimited (zero compliance = rigid) |
| Complexity | Stress tensors, Lamé params | Position corrections |
| Stability | Conditional | Unconditional |

## XPBD Algorithm

### Per-Frame Update

```python
for substep in range(SUBSTEPS):  # 4 substeps
    # 1. Pre-solve: gravity + position prediction
    for vertex in body:
        vertex.prev_pos = vertex.pos
        vertex.vel += gravity * dt
        vertex.pos += vertex.vel * dt

    # 2. Constraint solving (5 iterations)
    for iteration in range(5):
        solve_edge_constraints()
        solve_area_constraints()

    # 3. Collision handling
    solve_ground_collision()
    solve_inter_body_collisions()

    # 4. Post-solve: derive velocities
    for vertex in body:
        vertex.vel = (vertex.pos - vertex.prev_pos) / dt
```

### Key Insight

Velocities are **derived** from position changes, not the other way around. This is why XPBD is unconditionally stable—position corrections are bounded, and velocities follow.

## Constraints

### Edge Constraints (Distance)

Maintains rest length between connected vertices.

**Constraint function:**
```
C = |x₁ - x₀| - rest_length
```

**Gradient:**
```
∇C₀ = -(x₁ - x₀) / |x₁ - x₀|
∇C₁ = +(x₁ - x₀) / |x₁ - x₀|
```

**XPBD correction:**
```
λ = -C / (w₀|∇C₀|² + w₁|∇C₁|² + α/dt²)
Δx₀ = -λ · w₀ · ∇C₀
Δx₁ = -λ · w₁ · ∇C₁
```

Where:
- `wᵢ` = inverse mass of vertex i
- `α` = compliance (0 = infinitely stiff)
- `dt` = substep timestep

**Zero compliance** makes edges perfectly rigid, which is essential for **shape preservation**. Without this, bodies flatten into "pancakes" under load.

### Area Constraints (2D Volume)

Preserves triangle area.

**Constraint function:**
```
C = current_area - rest_area
```

Where area is computed from the cross product:
```
area = 0.5 * |(x₁ - x₀) × (x₂ - x₀)|
```

**Gradients (perpendicular to opposite edge):**
```
∇C₀ = 0.5 * (y₁ - y₂, x₂ - x₁)
∇C₁ = 0.5 * (y₂ - y₀, x₀ - x₂)
∇C₂ = 0.5 * (y₀ - y₁, x₁ - x₀)
```

Area constraints with **non-zero compliance** allow soft body squishing while edge constraints maintain overall shape.

## Material Parameters

### Compliance

Compliance α is the inverse of stiffness:
- `α = 0`: Infinitely stiff (rigid constraint)
- `α > 0`: Soft constraint (higher = softer)

The effective stiffness depends on timestep:
```
effective_alpha = α / dt²
```

This makes constraint stiffness independent of substep count.

### Material Presets

| Material | Edge Compliance | Area Compliance | Behavior |
|----------|-----------------|-----------------|----------|
| Jello    | 0               | 1e-6            | Soft, jiggly |
| Rubber   | 0               | 1e-7            | Bouncy |
| Wood     | 0               | 1e-8            | Stiff |
| Metal    | 0               | 0               | Perfectly rigid |

All materials use **zero edge compliance** to prevent shape collapse.

### Mass Distribution

Vertex masses are computed from triangle areas:
```
vertex_mass += (triangle_area * density) / 3
```

Inverse mass is used for constraint solving:
```
inv_mass = 1 / mass  (or 0 for fixed vertices)
```

## Collision Handling

### Ground Collision

Position-based ground collision with friction and restitution:

```python
if vertex.y < ground_y:
    # Project out of ground
    vertex.y = ground_y

    # Reflection (restitution)
    if moving_down:
        penetration = ground_y - old_y
        vertex.y = ground_y + penetration * RESTITUTION

    # Friction
    horizontal_movement *= FRICTION
```

### Inter-Body Collision

Vertex-to-vertex collision with mass-weighted separation:

```python
for each vertex pair (i, j) from different bodies:
    dist = distance(i, j)
    if dist < min_dist:
        overlap = min_dist - dist
        normal = normalize(j.pos - i.pos)

        # Mass-weighted separation
        w_sum = i.inv_mass + j.inv_mass
        i.pos -= normal * overlap * (i.inv_mass / w_sum)
        j.pos += normal * overlap * (j.inv_mass / w_sum)
```

**Important:** Collisions must happen **before** post_solve so that velocities correctly reflect the collision response.

## Constraint Iterations

More iterations = better constraint convergence. The simulation uses:

- **5 iterations** per substep for stiff materials
- **4 substeps** per frame at 60Hz

This provides good shape preservation while maintaining real-time performance.

### Gauss-Seidel vs Jacobi

Constraints are solved in **Gauss-Seidel** style (immediate position updates) rather than Jacobi (batch updates). This provides faster convergence.

## Mesh Topology

### Ring Mesh

The ring (annulus) mesh is generated with:
- Concentric rings of vertices from inner to outer radius
- Proper wrap-around connectivity (no seam discontinuity)
- Triangle pairs forming quads between adjacent rings

```
Vertices: SEGMENTS * (RADIAL_DIVISIONS + 1)
Triangles: SEGMENTS * RADIAL_DIVISIONS * 2
```

### Edge Constraint Generation

Edges are extracted from triangles, avoiding duplicates:
```python
edge_set = set()
for triangle in triangles:
    for edge in triangle.edges:
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in edge_set:
            edge_set.add(sorted_edge)
            constraints.append(EdgeConstraint(edge, rest_length))
```

## Numerical Stability

### Why XPBD is Stable

1. **Position-based**: Corrections are bounded by constraint violations
2. **Compliance scaling**: `α/dt²` ensures consistent stiffness across timesteps
3. **No force accumulation**: No risk of force explosion

### Shape Preservation

The critical insight for preventing "pancaking":

- **Zero edge compliance** makes edges rigid
- Rigid edges prevent horizontal spreading under vertical compression
- Area constraints alone are insufficient (area can be preserved while flattening)

### Timestep Independence

XPBD's compliance formulation makes behavior timestep-independent:
```
λ = -C / (w_sum + α/dt²)
```

As `dt → 0`, the compliance term `α/dt²` dominates, giving consistent stiffness.

## Comparison with Neo-Hookean FEM

The codebase includes a legacy force-based FEM solver (`softbody.rs`) using Neo-Hookean hyperelasticity. Key differences:

| Aspect | Neo-Hookean FEM | XPBD |
|--------|-----------------|------|
| Physics basis | Continuum mechanics | Constraint projection |
| Stress computation | P = μF + (λlog(J) - μ)F⁻ᵀ | None (position-based) |
| Material model | Strain energy density | Compliance parameters |
| Large deformation | Requires strain limiting | Naturally handled |
| Implementation | ~200 lines | ~150 lines |

XPBD trades physical accuracy for stability and simplicity. For real-time soft body simulation, this trade-off is usually worthwhile.

## References

1. Macklin, M., Müller, M., & Chentanez, N. (2016). XPBD: Position-Based Simulation of Compliant Constrained Dynamics. Motion in Games.
2. Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007). Position Based Dynamics. Journal of Visual Communication and Image Representation.
3. Bender, J., Müller, M., & Macklin, M. (2017). A Survey on Position Based Dynamics. EG STAR.
4. Ten Minute Physics - XPBD tutorial series (https://matthias-research.github.io/pages/tenMinutePhysics/)
