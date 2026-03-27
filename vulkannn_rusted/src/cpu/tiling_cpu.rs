//! tiling_cpu.rs — Cache-Oblivious CPU Algorithms
//!
//! Implements recursive divide-and-conquer strategies for Matrix Multiplication.
//! These algorithms are "cache-oblivious" because they perform well across
//! all levels of the memory hierarchy without explicit knowledge of cache sizes.

/// Recursive Cache-Oblivious Matrix Multiplication: C = A * B
/// Threshold defines the base case size where we stop recursing and use scalar loops.
pub fn matmul_recursive(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, k: usize, n: usize,
    rsa: usize, csa: usize,
    rsb: usize, csb: usize,
    rsc: usize, csc: usize,
) {
    let threshold = 32;

    if m <= threshold && n <= threshold && k <= threshold {
        // Base case: simple triple loop
        for i in 0..m {
            for l in 0..k {
                let s = a[i * rsa + l * csa];
                for j in 0..n {
                    c[i * rsc + j * csc] += s * b[l * rsb + j * csb];
                }
            }
        }
        return;
    }

    // Recursive case: split the largest dimension
    if m >= n && m >= k {
        let m2 = m / 2;
        // Top half
        matmul_recursive(&a[0..], b, &mut c[0..], m2, k, n, rsa, csa, rsb, csb, rsc, csc);
        // Bottom half
        matmul_recursive(&a[m2 * rsa..], b, &mut c[m2 * rsc..], m - m2, k, n, rsa, csa, rsb, csb, rsc, csc);
    } else if n >= k {
        let n2 = n / 2;
        // Left half
        matmul_recursive(a, &b[0..], &mut c[0..], m, k, n2, rsa, csa, rsb, csb, rsc, csc);
        // Right half
        matmul_recursive(a, &b[n2 * csb..], &mut c[n2 * csc..], m, k, n - n2, rsa, csa, rsb, csb, rsc, csc);
    } else {
        let k2 = k / 2;
        // A_left * B_top
        matmul_recursive(&a[0..], &b[0..], c, m, k2, n, rsa, csa, rsb, csb, rsc, csc);
        // A_right * B_bottom
        matmul_recursive(&a[k2 * csa..], &b[k2 * rsb..], c, m, k - k2, n, rsa, csa, rsb, csb, rsc, csc);
    }
}
