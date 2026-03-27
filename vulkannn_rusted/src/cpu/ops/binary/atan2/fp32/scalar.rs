//! Baseline Scalar Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Utilizes a 3rd-order Minimax Polynomial (Remez) for 3.6e-5 error bounds.

#[inline(always)]
pub fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    assert_eq!(n, x.len());
    assert_eq!(n, res.len());

    // Minimax Polynomial Constants (Remez Algorithm)
    const C0: f32 = 0.99978784;
    const C1: f32 = -0.32580840;
    const C2: f32 = 0.15557865;
    const C3: f32 = -0.04432655;
    
    const PI: f32 = core::f32::consts::PI;
    const PI_2: f32 = core::f32::consts::FRAC_PI_2;

    for i in 0..n {
        let yi = y[i];
        let xi = x[i];

        let y_abs = yi.abs();
        let x_abs = xi.abs();

        // 1. Argument Reduction to [0, 1]
        let (a, swap) = if y_abs > x_abs {
            (x_abs / y_abs, true)
        } else {
            (y_abs / x_abs, false)
        };

        // 2. Polynomial Evaluation (Horner's Scheme)
        let s = a * a;
        let mut p = a * (C0 + s * (C1 + s * (C2 + s * C3)));

        // 3. Quadrant Restoration
        if swap {
            p = PI_2 - p;
        }
        
        if xi < 0.0 {
            p = PI - p;
        }

        if yi < 0.0 {
            p = -p;
        }

        res[i] = p;
    }
}
