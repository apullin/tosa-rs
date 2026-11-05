use tosa_rs::load_file;
fn main() {
    let g = load_file(std::path::Path::new(&std::env::args().nth(1).unwrap())).unwrap();
    for region in &g.regions { for block in &region.blocks {
        for t in &block.tensors {
            let Some(d) = t.data.as_ref().filter(|d| !d.is_empty()) else { continue };
            let dt = format!("{:?}", t.dtype).to_lowercase();
            let v = if dt.contains("shape") { format!("{:?}", d.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>()) }
                else if dt.contains("int32") { format!("{:?}", d.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>()) }
                else if dt.contains("int16") { format!("{:?}", d.chunks(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>()) }
                else if dt.contains("fp32")||dt.contains("float") { format!("{:?}", d.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>()) }
                else { format!("{:?}", d.iter().map(|b| *b as i8).collect::<Vec<_>>()) };
            // strip the trailing per-name suffix so master/tosater names line up
            let key = t.name.clone();
            println!("{:40} {:?} {}", key, t.dtype, v);
        }
    }}
}
