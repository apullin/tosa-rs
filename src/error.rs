use thiserror::Error;

#[derive(Debug, Error)]
pub enum TosaError {
    #[error("invalid flatbuffer: {0}")]
    InvalidFlatbuffer(String),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("unknown attribute type: {0}")]
    UnknownAttribute(u8),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
