use std::f32::consts::PI;

pub struct Xoshiro256pp {
    state: [u64; 4],
}

impl Xoshiro256pp {
    #[inline(always)]
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn new(seed: u64) -> Self {
        let mut sm_state = seed;
        let s0 = Self::splitmix64(&mut sm_state);
        let s1 = Self::splitmix64(&mut sm_state);
        let s2 = Self::splitmix64(&mut sm_state);
        let s3 = Self::splitmix64(&mut sm_state);

        // Ensure state is not all zeros
        if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
            Self {
                state: [1, 2, 3, 4],
            }
        } else {
            Self {
                state: [s0, s1, s2, s3],
            }
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Returns a uniform f32 in range [0.0, 1.0)
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        // Generate uniform 24-bit mantissa
        (self.next_u64() >> 40) as f32 * (1.0 / 16777216.0)
    }

    /// Fast, slightly approximate natural log for values in (0, 1]
    #[inline(always)]
    fn fast_ln(x: f32) -> f32 {
        let bx = x.to_bits();
        let e = (((bx >> 23) & 0xFF) as i32) - 127;
        let m = f32::from_bits((bx & 0x007FFFFF) | 0x3F800000); // 1.m
        // Polynomial approximation for ln(m) on [1, 2]
        let z = m - 1.0;
        let ln_m = z * (1.0 - z * (0.5 - (0.33333333 * z)));
        (e as f32) * 0.69314718 + ln_m
    }

    /// Fast sin/cos using quadratic approximations
    #[inline(always)]
    fn fast_sincos(x: f32) -> (f32, f32) {
        // Range reduction to [-PI, PI] handled by assuming input x is 2*PI*u2
        // where u2 is [0, 1), so x is [0, 2PI)
        let mut x = x;
        if x > PI {
            x -= 2.0 * PI;
        }
        
        // Bhaskara I sine approximation / proper quadratic
        let x2 = x * x;
        let sin_x = x - (x * x2) / 6.0 * (1.0 - x2 / 20.0);
        
        let mut cx = x + PI/2.0;
        if cx > PI { cx -= 2.0 * PI; }
        let cx2 = cx * cx;
        let cos_x = cx - (cx * cx2) / 6.0 * (1.0 - cx2 / 20.0);
        
        (sin_x, cos_x)
    }

    /// Box-Muller transform: consumes 2 uniforms to produce 2 standard normals
    #[inline(always)]
    pub fn next_randn_pair(&mut self) -> (f32, f32) {
        let mut u1 = self.next_f32();
        if u1 < 1e-7 {
            u1 = 1e-7; // Prevent log(0)
        }
        let u2 = self.next_f32();

        let r = (-2.0 * Self::fast_ln(u1)).sqrt();
        let theta = 2.0 * PI * u2;
        let (st, ct) = Self::fast_sincos(theta);

        (r * ct, r * st)
    }
}
