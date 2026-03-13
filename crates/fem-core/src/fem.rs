//! Finite Element Method (FEM) computations for 2D soft body simulation
//! Using Neo-Hookean hyperelastic material model

use crate::math::*;

/// Lamé parameters derived from material properties
#[derive(Clone, Copy, Debug)]
pub struct LameParams {
    pub mu: f32,      // Shear modulus
    pub lambda: f32,  // First Lamé parameter
}

/// Precomputed data for a triangle
#[derive(Clone, Debug)]
pub struct TriangleData {
    pub rest_dm_inv: Mat2,  // Inverse of rest shape matrix
    pub rest_area: f32,     // Rest area
    pub plastic_def: Mat2,  // Plastic deformation (starts as identity)
}

/// Forces on triangle vertices
#[derive(Clone, Copy, Debug)]
pub struct TriangleForces {
    pub f0: Vec2,
    pub f1: Vec2,
    pub f2: Vec2,
}

/// Result of triangle force computation
#[derive(Clone, Copy, Debug)]
pub struct TriangleForceResult {
    pub f0: Vec2,
    pub f1: Vec2,
    pub f2: Vec2,
    pub j: f32,       // Volume ratio (determinant of F)
    pub energy: f32,  // Strain energy
}

/// Force computation statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct ForceStats {
    pub total_energy: f32,
    pub min_j: f32,
    pub max_j: f32,
}

/// Convert Young's modulus and Poisson ratio to Lamé parameters
pub fn compute_lame_parameters(e: f32, nu: f32) -> LameParams {
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    LameParams { mu, lambda }
}

/// Compute the rest shape matrix Dm for a triangle
/// Dm = [X1 - X0, X2 - X0] where Xi are rest positions
#[inline]
pub fn compute_rest_shape_matrix(
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
) -> Mat2 {
    mat2_create(x1 - x0, y1 - y0, x2 - x0, y2 - y0)
}

/// Compute the deformed shape matrix Ds for a triangle
#[inline]
pub fn compute_deformed_shape_matrix(
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
) -> Mat2 {
    mat2_create(x1 - x0, y1 - y0, x2 - x0, y2 - y0)
}

/// Compute the deformation gradient F = Ds * Dm^(-1)
#[inline]
pub fn compute_deformation_gradient(ds: &Mat2, dm_inv: &Mat2) -> Mat2 {
    mat2_mul(ds, dm_inv)
}

/// Compute the volume ratio J = det(F)
#[inline]
pub fn compute_volume_ratio(f: &Mat2) -> f32 {
    mat2_det(f)
}

/// Compute Neo-Hookean strain energy density
/// Ψ = (μ/2)(tr(F^T F) - 2) - μ log(J) + (λ/2)(log(J))^2
pub fn compute_neo_hookean_energy(f: &Mat2, j: f32, mu: f32, lambda: f32) -> f32 {
    let ftf_trace = mat2_frobenius_norm_sq(f);
    let log_j = j.max(1e-10).ln();
    (mu / 2.0) * (ftf_trace - 2.0) - mu * log_j + (lambda / 2.0) * log_j * log_j
}

/// Compute Neo-Hookean first Piola-Kirchhoff stress tensor P
/// P = μF + (λ log(J) - μ) F^(-T)
/// Uses Stable Neo-Hookean formulation with log barrier for inversion prevention
pub fn compute_neo_hookean_stress(f: &Mat2, j: f32, mu: f32, lambda: f32) -> Mat2 {
    // Safety: if J is invalid, return zero stress (let strain limiting fix it)
    if !j.is_finite() || j <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    // Stable Neo-Hookean: clamp J away from zero with smooth barrier
    const J_MIN: f32 = 0.2;  // Hard floor
    const J_BARRIER_START: f32 = 0.7;  // Barrier starts here

    let safe_j = j.max(J_MIN);
    let log_j = safe_j.ln();

    let f_inv_t = mat2_inv_transpose(f);

    // Check for invalid inverse (degenerate triangle)
    if !f_inv_t[0].is_finite() || !f_inv_t[1].is_finite() ||
       !f_inv_t[2].is_finite() || !f_inv_t[3].is_finite() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    // P = μF + (λ log(J) - μ) F^(-T)
    let term1 = mat2_scale(f, mu);
    let mut coeff = lambda * log_j - mu;

    // Log barrier for inversion prevention
    // Energy barrier: -k * log(J - J_min) which gives force: k / (J - J_min)
    // This smoothly increases to infinity as J approaches J_min
    if j < J_BARRIER_START {
        // Smooth transition using quadratic blend into log barrier
        let t = (J_BARRIER_START - j) / (J_BARRIER_START - J_MIN);
        let t_clamped = t.min(0.95);  // Never fully reach the singularity

        // Barrier strength increases with material stiffness
        let barrier_strength = mu * 2.0;

        // Quadratic-to-log barrier: smooth at start, steep near J_min
        // derivative = barrier_strength * (2t + t²/(1-t))
        let barrier_force = barrier_strength * (2.0 * t_clamped + t_clamped * t_clamped / (1.0 - t_clamped + 0.01));
        coeff -= barrier_force;
    }

    // Clamp coefficient to prevent extreme values
    coeff = coeff.clamp(-mu * 100.0, mu * 100.0);

    let term2 = mat2_scale(&f_inv_t, coeff);
    let result = mat2_add(&term1, &term2);

    // Final safety check
    if !result[0].is_finite() || !result[1].is_finite() ||
       !result[2].is_finite() || !result[3].is_finite() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    result
}

/// Compute elastic forces on triangle vertices from stress
/// H = -Area * P * Dm^(-T)
pub fn compute_elastic_forces(p: &Mat2, dm_inv: &Mat2, area: f32) -> TriangleForces {
    let dm_inv_t = mat2_transpose(dm_inv);
    let p_dm_inv_t = mat2_mul(p, &dm_inv_t);
    let h = mat2_scale(&p_dm_inv_t, -area);

    let f1: Vec2 = [h[0], h[1]];
    let f2: Vec2 = [h[2], h[3]];
    let f0: Vec2 = [-f1[0] - f2[0], -f1[1] - f2[1]];

    TriangleForces { f0, f1, f2 }
}

/// Compute triangle area from vertex positions
pub fn compute_triangle_area(
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
) -> f32 {
    let e1x = x1 - x0;
    let e1y = y1 - y0;
    let e2x = x2 - x0;
    let e2y = y2 - y0;
    let cross = e1x * e2y - e1y * e2x;
    cross.abs() * 0.5
}

/// Compute precomputed triangle data from vertex positions
pub fn compute_triangle_data(
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
) -> TriangleData {
    let dm = compute_rest_shape_matrix(x0, y0, x1, y1, x2, y2);
    let rest_dm_inv = mat2_inv(&dm);
    let rest_area = compute_triangle_area(x0, y0, x1, y1, x2, y2);

    TriangleData {
        rest_dm_inv,
        rest_area,
        plastic_def: mat2_identity(),
    }
}

/// Compute all forces for a single triangle - main FEM computation
/// Now with plasticity support: uses plastic_def to compute elastic deformation
pub fn compute_triangle_forces(
    tri: &TriangleData,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    mu: f32,
    lambda: f32,
) -> TriangleForceResult {
    // Step 1: Deformed shape matrix
    let ds = compute_deformed_shape_matrix(x0, y0, x1, y1, x2, y2);

    // Step 2: Total deformation gradient
    let f_total = compute_deformation_gradient(&ds, &tri.rest_dm_inv);

    // Step 3: Elastic deformation (remove plastic part): Fe = F * Fp^(-1)
    let fp_inv = mat2_inv(&tri.plastic_def);
    let f = mat2_mul(&f_total, &fp_inv);

    // Step 4: Volume ratio (of elastic deformation)
    let j = compute_volume_ratio(&f);

    // Step 5: Stress
    let p = compute_neo_hookean_stress(&f, j, mu, lambda);

    // Step 6: Forces
    let forces = compute_elastic_forces(&p, &tri.rest_dm_inv, tri.rest_area);

    // Step 7: Energy
    let energy = compute_neo_hookean_energy(&f, j, mu, lambda);

    TriangleForceResult {
        f0: forces.f0,
        f1: forces.f1,
        f2: forces.f2,
        j,
        energy,
    }
}

/// Compute stress magnitude (Frobenius norm) for yield checking
pub fn compute_stress_magnitude(
    tri: &TriangleData,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    mu: f32,
    lambda: f32,
) -> (f32, Mat2) {
    let ds = compute_deformed_shape_matrix(x0, y0, x1, y1, x2, y2);
    let f_total = compute_deformation_gradient(&ds, &tri.rest_dm_inv);
    let fp_inv = mat2_inv(&tri.plastic_def);
    let f = mat2_mul(&f_total, &fp_inv);
    let j = compute_volume_ratio(&f);
    let p = compute_neo_hookean_stress(&f, j, mu, lambda);
    let stress_mag = mat2_frobenius_norm_sq(&p).sqrt();
    (stress_mag, f_total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_close(actual: f32, expected: f32, msg: &str) {
        assert!((actual - expected).abs() < EPSILON, "{}: expected {}, got {}", msg, expected, actual);
    }

    #[test]
    fn test_lame_parameters() {
        let params = compute_lame_parameters(200e9, 0.3);
        assert_close(params.mu, 200e9 / (2.0 * 1.3), "mu");
    }

    #[test]
    fn test_triangle_area() {
        let area = compute_triangle_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        assert_close(area, 0.5, "unit right triangle");
    }

    #[test]
    fn test_neo_hookean_energy_at_rest() {
        let f = mat2_identity();
        let energy = compute_neo_hookean_energy(&f, 1.0, 1000.0, 2000.0);
        assert_close(energy, 0.0, "energy at rest");
    }

    #[test]
    fn test_neo_hookean_stress_at_rest() {
        let f = mat2_identity();
        let p = compute_neo_hookean_stress(&f, 1.0, 1000.0, 2000.0);
        for i in 0..4 {
            assert_close(p[i], 0.0, &format!("P[{}] at rest", i));
        }
    }

    #[test]
    fn test_force_equilibrium() {
        let p = [100.0, 50.0, 30.0, 80.0];
        let dm_inv = [1.0, 0.1, -0.1, 1.0];
        let forces = compute_elastic_forces(&p, &dm_inv, 0.5);

        let sum_x = forces.f0[0] + forces.f1[0] + forces.f2[0];
        let sum_y = forces.f0[1] + forces.f1[1] + forces.f2[1];

        assert_close(sum_x, 0.0, "force sum X");
        assert_close(sum_y, 0.0, "force sum Y");
    }

    #[test]
    fn test_shape_matrix() {
        // Triangle: (0,0), (1,0), (0,1)
        let dm = compute_rest_shape_matrix(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        // Dm = [x1-x0, y1-y0, x2-x0, y2-y0] = [1, 0, 0, 1]
        assert_close(dm[0], 1.0, "dm[0]");
        assert_close(dm[1], 0.0, "dm[1]");
        assert_close(dm[2], 0.0, "dm[2]");
        assert_close(dm[3], 1.0, "dm[3]");
    }

    #[test]
    fn test_deformation_gradient_identity() {
        // Same rest and deformed shape = identity deformation gradient
        let dm = compute_rest_shape_matrix(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        let dm_inv = mat2_inv(&dm);
        let ds = compute_deformed_shape_matrix(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        let f = compute_deformation_gradient(&ds, &dm_inv);

        assert_close(f[0], 1.0, "F[0]");
        assert_close(f[1], 0.0, "F[1]");
        assert_close(f[2], 0.0, "F[2]");
        assert_close(f[3], 1.0, "F[3]");
    }

    #[test]
    fn test_volume_ratio() {
        let f_identity = mat2_identity();
        assert_close(compute_volume_ratio(&f_identity), 1.0, "J at rest");

        // Uniform scaling by 2 = area ratio of 4
        let f_scaled = [2.0, 0.0, 0.0, 2.0];
        assert_close(compute_volume_ratio(&f_scaled), 4.0, "J scaled 2x");
    }

    #[test]
    fn test_triangle_forces_at_rest() {
        // Triangle at rest should have zero forces
        let dm = compute_rest_shape_matrix(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        let tri_data = TriangleData {
            rest_dm_inv: mat2_inv(&dm),
            rest_area: 0.5,
            plastic_def: mat2_identity(),
        };

        let result = compute_triangle_forces(
            &tri_data,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            1000.0, 2000.0,
        );

        assert_close(result.f0[0], 0.0, "f0.x at rest");
        assert_close(result.f0[1], 0.0, "f0.y at rest");
        assert_close(result.f1[0], 0.0, "f1.x at rest");
        assert_close(result.f1[1], 0.0, "f1.y at rest");
        assert_close(result.j, 1.0, "J at rest");
        assert_close(result.energy, 0.0, "energy at rest");
    }
}
