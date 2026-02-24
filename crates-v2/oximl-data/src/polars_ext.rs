use std::path::Path;
use std::fs::File;
use oximl_core::{Tensor, TensorResult, TensorError, DType};

/// Helper function to load a numeric CSV directly into a 2D Tensor.
/// Replaces the Polars dependency to ensure compilation across all Rustc versions.
pub fn load_csv_to_tensor<P: AsRef<Path>>(path: P) -> TensorResult<Tensor> {
    let file = File::open(path).map_err(|e| TensorError::InvalidOperation(format!("File Open Error: {}", e)))?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut flattened_data = Vec::new();
    let mut height = 0;
    let mut width = 0;

    for (i, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| TensorError::InvalidOperation(format!("CSV Record Error: {}", e)))?;
        
        if i == 0 {
            width = record.len();
        } else if record.len() != width {
            return Err(TensorError::InvalidOperation("CSV columns are not uniform".into()));
        }

        for field in record.iter() {
            let val = field.parse::<f64>().unwrap_or(0.0);
            flattened_data.push(val);
        }
        height += 1;
    }

    if height == 0 || width == 0 {
        return Err(TensorError::InvalidOperation("CSV is empty".into()));
    }

    // Use Ndarray backend to absorb the flattened vector
    let array = ndarray::Array2::from_shape_vec((height, width), flattened_data)
        .map_err(|e| TensorError::InvalidOperation(format!("Shape error: {}", e)))?;
    
    Ok(Tensor::Float64(array.into_dyn().into_shared()))
}
