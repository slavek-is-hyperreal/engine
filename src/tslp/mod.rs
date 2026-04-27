// src/tslp/mod.rs
//
// TSLP (Tree Straight-Line Program) scheduler for EML DAGs.
//
// Implements the parallel evaluation schedule from Theorem C3 (PAPER.md):
// - Assign depth levels to DAG nodes
// - Group nodes by depth into parallel "waves"
// - Execute each wave in parallel (GPU dispatch)
//
// Theoretical basis:
// - Ganardi, Jeż & Lohrey (JACM 2021): balanced SLP in O(g) time
// - Brent (1974): algebraic expressions evaluable in O(log N) depth
// - Result: EML network inference ∈ NC1

pub mod depth;
pub mod scheduler;
pub mod executor;

pub use depth::{assign_depths, max_depth};
pub use scheduler::{build_schedule, TslpSchedule};
pub use executor::{simulate_execution, measure_transformer_depth_reduction};
