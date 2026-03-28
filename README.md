# tosa

Rust types and serialization for TOSA (Tensor Operator Set Architecture) FlatBuffers.

## What it does

- Provides generated FlatBuffers bindings from the official TOSA schema
- Defines ergonomic owned Rust types (`TosaGraph`, `TosaRegion`, `TosaBasicBlock`, `TosaOperator`, `TosaTensor`) with serde support
- Serializes and deserializes TOSA graphs to/from FlatBuffers binary format
- Includes a builder API (`TosaGraphBuilder`) for constructing graphs programmatically

## Prerequisites

`flatc` (FlatBuffers compiler) must be installed and available on your `PATH`.

```sh
# macOS
brew install flatbuffers

# Ubuntu/Debian
apt install flatbuffers-compiler
```

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
tosa = { path = "path/to/rust-tosa" }
```

### Deserializing a .tosa file

```rust
use std::path::Path;

let graph = tosa::load_file(Path::new("model.tosa")).unwrap();
println!("Regions: {}", graph.regions.len());
```

### Building a graph with the builder

```rust
use tosa::*;

let graph = TosaGraphBuilder::new(0, 80, 1)
    .add_region("main")
    .add_block("block0")
    .add_operator(
        Op::Add,
        Attribute::Add(AddAttribute::default()),
        &["a", "b"],
        &["c"],
    )
    .finish_block()
    .finish_region()
    .build();

let bytes = tosa::serialize(&graph);
```

### Round-trip serialization

```rust
let bytes = tosa::serialize(&graph);
let deserialized = tosa::deserialize(&bytes).unwrap();
assert_eq!(graph.regions.len(), deserialized.regions.len());
```

## TOSA spec coverage

This crate covers the full TOSA specification:

- 77 operators (all variants of `Op`)
- 78 attribute types (all variants of `Attribute`)
- 15 data types (`DType`: Bool, Int4, Int8, Int16, Int32, Int48, Uint8, Uint16, Fp16, Fp32, Bf16, Fp8E4M3, Fp8E5M2, Shape, Unknown)

## Testing

153 tests covering type conversions, serialization round-trips, and builder correctness.

```sh
cargo test
```

## License

Apache 2.0

## References

- [TOSA Specification](https://www.mlplatform.org/tosa/)
