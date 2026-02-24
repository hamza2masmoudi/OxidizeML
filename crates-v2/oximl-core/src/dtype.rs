/// Dynamic Data Types for Tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
}

impl DType {
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DType::Int32 | DType::Int64 | DType::UInt8)
    }
}
