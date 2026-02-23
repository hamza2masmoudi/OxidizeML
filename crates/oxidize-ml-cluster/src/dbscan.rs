use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

/// DBSCAN — Density-Based Spatial Clustering of Applications with Noise.
pub struct DBSCAN<T: Float> {
    pub eps: T,
    pub min_samples: usize,
    pub labels: Option<Tensor<T>>,
}

impl<T: Float> DBSCAN<T>
where
    Standard: Distribution<T>,
{
    pub fn new(eps: T, min_samples: usize) -> Self {
        DBSCAN {
            eps,
            min_samples,
            labels: None,
        }
    }

    fn euclidean_dist(x: &Tensor<T>, i: usize, j: usize, d: usize) -> TensorResult<T> {
        let mut dist = T::ZERO;
        for k in 0..d {
            let diff = x.get(&[i, k])? - x.get(&[j, k])?;
            dist = dist + diff * diff;
        }
        Ok(dist.sqrt())
    }

    fn region_query(&self, x: &Tensor<T>, i: usize, n: usize, d: usize) -> TensorResult<Vec<usize>> {
        let mut neighbors = Vec::new();
        for j in 0..n {
            if Self::euclidean_dist(x, i, j, d)? <= self.eps {
                neighbors.push(j);
            }
        }
        Ok(neighbors)
    }

    pub fn fit(&mut self, x: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let d = x.shape().dim(1)?;

        let noise: i32 = -1;
        let mut labels = vec![-1i32; n]; // -1 = unvisited
        let mut cluster_id: i32 = 0;
        let mut visited = vec![false; n];

        for i in 0..n {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let neighbors = self.region_query(x, i, n, d)?;

            if neighbors.len() < self.min_samples {
                labels[i] = noise;
                continue;
            }

            labels[i] = cluster_id;
            let mut queue = neighbors.clone();
            let mut qi = 0;

            while qi < queue.len() {
                let j = queue[qi];
                qi += 1;

                if !visited[j] {
                    visited[j] = true;
                    let j_neighbors = self.region_query(x, j, n, d)?;
                    if j_neighbors.len() >= self.min_samples {
                        for &nn in &j_neighbors {
                            if !queue.contains(&nn) {
                                queue.push(nn);
                            }
                        }
                    }
                }

                if labels[j] == -1 {
                    labels[j] = cluster_id;
                }
            }

            cluster_id += 1;
        }

        let label_data: Vec<T> = labels.iter().map(|&l| T::from_f64(l as f64)).collect();
        self.labels = Some(Tensor::new(label_data, vec![n])?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0],
            vec![10.0, 10.0], vec![10.1, 10.1], vec![10.2, 10.0],
            vec![100.0, 100.0], // noise point
        ]).unwrap();

        let mut db = DBSCAN::new(1.0, 2);
        db.fit(&x).unwrap();
        let labels = db.labels.as_ref().unwrap();

        // First cluster
        let c1 = labels.data()[0].to_f64() as i32;
        assert!(c1 >= 0);
        assert_eq!(labels.data()[1].to_f64() as i32, c1);

        // Second cluster
        let c2 = labels.data()[3].to_f64() as i32;
        assert!(c2 >= 0);
        assert_ne!(c1, c2);

        // Noise point
        assert_eq!(labels.data()[6].to_f64() as i32, -1);
    }
}
