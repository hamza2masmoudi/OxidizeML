use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Agglomerative (Hierarchical) Clustering.
///
/// Bottom-up approach: starts with each point as its own cluster,
/// then iteratively merges the closest pair of clusters.
pub struct AgglomerativeClustering<T: Float> {
    pub n_clusters: usize,
    pub linkage: Linkage,
    labels: Option<Vec<usize>>,
    _marker: std::marker::PhantomData<T>,
}

#[derive(Clone, Debug)]
pub enum Linkage {
    Single,    // min distance between clusters
    Complete,  // max distance between clusters
    Average,   // mean distance between clusters
}

impl<T: Float> AgglomerativeClustering<T> {
    pub fn new(n_clusters: usize, linkage: Linkage) -> Self {
        AgglomerativeClustering {
            n_clusters,
            linkage,
            labels: None,
            _marker: std::marker::PhantomData,
        }
    }

    fn euclidean_dist(x: &Tensor<T>, i: usize, j: usize, p: usize) -> f64 {
        let mut sum = 0.0;
        for k in 0..p {
            let diff = x.get(&[i, k]).unwrap().to_f64() - x.get(&[j, k]).unwrap().to_f64();
            sum += diff * diff;
        }
        sum.sqrt()
    }

    pub fn fit(&mut self, x: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        // Compute distance matrix
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let d = Self::euclidean_dist(x, i, j, p);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        // Each point starts as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        while clusters.len() > self.n_clusters {
            // Find closest pair of clusters
            let mut min_dist = f64::INFINITY;
            let mut merge_a = 0;
            let mut merge_b = 1;

            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let d = self.cluster_distance(&dist, &clusters[i], &clusters[j]);
                    if d < min_dist {
                        min_dist = d;
                        merge_a = i;
                        merge_b = j;
                    }
                }
            }

            // Merge clusters
            let merged = clusters[merge_b].clone();
            clusters[merge_a].extend(merged);
            clusters.remove(merge_b);
        }

        // Assign labels
        let mut labels = vec![0usize; n];
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &point_id in cluster {
                labels[point_id] = cluster_id;
            }
        }
        self.labels = Some(labels);
        Ok(())
    }

    fn cluster_distance(&self, dist: &[Vec<f64>], a: &[usize], b: &[usize]) -> f64 {
        match self.linkage {
            Linkage::Single => {
                let mut min = f64::INFINITY;
                for &i in a {
                    for &j in b {
                        if dist[i][j] < min { min = dist[i][j]; }
                    }
                }
                min
            }
            Linkage::Complete => {
                let mut max = 0.0;
                for &i in a {
                    for &j in b {
                        if dist[i][j] > max { max = dist[i][j]; }
                    }
                }
                max
            }
            Linkage::Average => {
                let mut sum = 0.0;
                let mut count = 0;
                for &i in a {
                    for &j in b {
                        sum += dist[i][j];
                        count += 1;
                    }
                }
                sum / count as f64
            }
        }
    }

    pub fn labels(&self) -> Option<&[usize]> {
        self.labels.as_deref()
    }

    pub fn fit_predict(&mut self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.fit(x)?;
        let labels = self.labels.as_ref().unwrap();
        let data: Vec<T> = labels.iter().map(|&l| T::from_usize(l)).collect();
        Tensor::new(data, vec![labels.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agglomerative_single() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.1, 0.1],
            vec![5.0, 5.0], vec![5.1, 5.1],
        ]).unwrap();

        let mut model = AgglomerativeClustering::new(2, Linkage::Single);
        let labels = model.fit_predict(&x).unwrap();

        // Points 0,1 should be in same cluster; 2,3 in another
        let l0 = labels.data()[0].to_f64().round() as usize;
        let l1 = labels.data()[1].to_f64().round() as usize;
        let l2 = labels.data()[2].to_f64().round() as usize;
        let l3 = labels.data()[3].to_f64().round() as usize;
        assert_eq!(l0, l1);
        assert_eq!(l2, l3);
        assert_ne!(l0, l2);
    }
}
