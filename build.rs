use std::env;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let status = Command::new("flatc")
        .args(["--rust", "-o", &out_dir, "docs/tosa.fbs"])
        .status()
        .expect("failed to run flatc — is it installed?");

    assert!(status.success(), "flatc failed with status: {}", status);

    println!("cargo:rerun-if-changed=docs/tosa.fbs");
}
