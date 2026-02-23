use oxidize_ml_core::Tensor;
use std::error::Error;
use std::path::Path;

/// Read a CSV file into a tensor and column headers.
/// Assumes all values are numeric. Skips the header row.
pub fn read_csv(path: &str) -> Result<(Tensor<f64>, Vec<String>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(Path::new(path))?;
    let headers: Vec<String> = rdr.headers()?.iter().map(|h| h.to_string()).collect();

    let mut data = Vec::new();
    let mut n_rows = 0usize;

    for result in rdr.records() {
        let record = result?;
        for field in record.iter() {
            let val: f64 = field.parse().unwrap_or(0.0);
            data.push(val);
        }
        n_rows += 1;
    }

    let n_cols = if n_rows > 0 { data.len() / n_rows } else { 0 };
    let tensor = Tensor::new(data, vec![n_rows, n_cols])
        .map_err(|e| format!("Failed to create tensor: {:?}", e))?;

    Ok((tensor, headers))
}

/// Write a tensor to a CSV file with optional headers.
pub fn write_csv(path: &str, data: &Tensor<f64>, headers: Option<&[String]>) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(Path::new(path))?;

    if let Some(h) = headers {
        wtr.write_record(h)?;
    }

    let rows = data.shape().dim(0).map_err(|e| format!("{:?}", e))?;
    let cols = data.shape().dim(1).map_err(|e| format!("{:?}", e))?;

    for i in 0..rows {
        let row: Vec<String> = (0..cols)
            .map(|j| format!("{}", data.get(&[i, j]).unwrap_or(0.0)))
            .collect();
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}
