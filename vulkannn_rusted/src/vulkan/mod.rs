pub mod context;
pub mod memory;
pub mod pipeline;
pub mod ops;

pub use context::{AshBackend, BACKEND, init_backend, AsyncOp, poll_async_ops, poll_async_ops_until, wait_for_all};
pub use memory::{CachedBuffer, get_buffer, get_buffer_readback, recycle_buffer, clear_all_caches, prune_buffer_cache};

pub use ops::elementwise::execute_elementwise_into;
pub use ops::index_select::execute_index_select_into;
pub use ops::activation::{execute_activation_into, submit_activation_into};
pub use ops::reduce::execute_reduce;
pub use ops::softmax::execute_softmax_into;
pub use ops::matmul::{execute_matmul_into, execute_linear_into};
pub use ops::bit_linear::execute_bit_linear_into;
pub use ops::norm::{execute_layer_norm_into, execute_rms_norm_into};
