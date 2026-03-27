pub mod ops;
pub mod conversions;
pub mod tiling_cpu;
pub mod dispatch;

// Bridge to cpu_old for incremental migration. 
// Symbols in this module (src/cpu) will shadow those in cpu_old.
pub use crate::cpu_old::*;

// New granular CPU backend re-exports
pub use ops::binary::add::add_bf16;
pub use ops::binary::sub::sub_bf16;
pub use ops::binary::mul::mul_bf16;
pub use ops::binary::div::div_bf16;
