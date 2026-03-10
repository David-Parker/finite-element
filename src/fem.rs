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
pub fn compute_neo_hookean_stress(f: &Mat2, j: f32, mu: f32, lambda: f32) -> Mat2 {
    // Clamp J to prevent numerical issues
    let safe_j = j.max(0.1);
    let log_j = safe_j.ln();

    let f_inv_t = mat2_inv_transpose(f);

    // P = μF + (λ log(J) - μ) F^(-T)
    let term1 = mat2_scale(f, mu);
    let coeff = lambda * log_j - mu;
    let term2 = mat2_scale(&f_inv_t, coeff);
    mat2_add(&term1, &term2)
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

/// Compute all forces for a single triangle - main FEM computation
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

    // Step 2: Deformation gradient
    let f = compute_deformation_gradient(&ds, &tri.rest_dm_inv);

    // Step 3: Volume ratio
    let j = compute_volume_ratio(&f);

    // Step 4: Stress
    let p = compute_neo_hookean_stress(&f, j, mu, lambda);

    // Step 5: Forces
    let forces = compute_elastic_forces(&p, &tri.rest_dm_inv, tri.rest_area);

    // Step 6: Energy
    let energy = compute_neo_hookean_energy(&f, j, mu, lambda);

    TriangleForceResult {
        f0: forces.f0,
        f1: forces.f1,
        f2: forces.f2,
        j,
        energy,
    }
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
}
