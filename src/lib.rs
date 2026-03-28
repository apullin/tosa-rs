// Include the flatc-generated code
#[allow(
    unused_imports,
    dead_code,
    non_snake_case,
    clippy::all,
    mismatched_lifetime_syntaxes
)]
mod tosa_generated {
    include!(concat!(env!("OUT_DIR"), "/tosa_generated.rs"));
}

pub mod types;
pub mod builder;
pub mod error;

// Re-export generated types
pub use tosa_generated::tosa as generated;
