use oxidize_ml_core::Tensor;
use oxidize_ml_autodiff::Variable;
use super::layers::Layer;

/// 1D Convolution layer.
///
/// Input shape:  [batch, in_channels, length]
/// Output shape: [batch, out_channels, out_length]
/// where out_length = (length - kernel_size) / stride + 1
pub struct Conv1D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub weight: Variable, // [out_channels, in_channels, kernel_size]
    pub bias: Variable,   // [out_channels]
}

impl Conv1D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        let k = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let w = Tensor::rand(vec![out_channels, in_channels, kernel_size], Some(42))
            .mul_scalar(2.0 * k).add_scalar(-k);
        let b = Tensor::zeros(vec![out_channels]);

        Conv1D {
            in_channels, out_channels, kernel_size, stride,
            weight: Variable::param(w),
            bias: Variable::param(b),
        }
    }

    /// Forward pass using im2col-style unrolling.
    pub fn forward_tensor(&self, input: &Tensor<f64>) -> Tensor<f64> {
        let shape = input.shape_vec();
        let (batch, _in_ch, length) = (shape[0], shape[1], shape[2]);
        let out_len = (length - self.kernel_size) / self.stride + 1;

        let mut output = vec![0.0f64; batch * self.out_channels * out_len];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for ol in 0..out_len {
                    let mut sum = self.bias.data.data()[oc];
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let input_idx = ol * self.stride + k;
                            let w_val = self.weight.data.get(&[oc, ic, k]).unwrap();
                            let x_val = input.get(&[b, ic, input_idx]).unwrap();
                            sum += w_val * x_val;
                        }
                    }
                    output[b * self.out_channels * out_len + oc * out_len + ol] = sum;
                }
            }
        }

        Tensor::new(output, vec![batch, self.out_channels, out_len]).unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// 2D Convolution layer.
///
/// Input shape:  [batch, in_channels, height, width]
/// Output shape: [batch, out_channels, out_h, out_w]
pub struct Conv2D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Variable, // [out_channels, in_channels, kH, kW]
    pub bias: Variable,   // [out_channels]
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let k = (2.0 / fan_in as f64).sqrt();
        let w = Tensor::rand(
            vec![out_channels, in_channels, kernel_size, kernel_size],
            Some(42),
        ).mul_scalar(2.0 * k).add_scalar(-k);
        let b = Tensor::zeros(vec![out_channels]);

        Conv2D {
            in_channels, out_channels, kernel_size, stride, padding,
            weight: Variable::param(w),
            bias: Variable::param(b),
        }
    }

    /// Compute output spatial dimension.
    fn out_dim(input_dim: usize, kernel: usize, stride: usize, padding: usize) -> usize {
        (input_dim + 2 * padding - kernel) / stride + 1
    }

    /// Forward pass.
    pub fn forward_tensor(&self, input: &Tensor<f64>) -> Tensor<f64> {
        let shape = input.shape_vec();
        let (batch, _ic, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = Self::out_dim(h, self.kernel_size, self.stride, self.padding);
        let out_w = Self::out_dim(w, self.kernel_size, self.stride, self.padding);

        let mut output = vec![0.0f64; batch * self.out_channels * out_h * out_w];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = self.bias.data.data()[oc];
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                                    let iw = (ow * self.stride + kw) as isize - self.padding as isize;

                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        let w_val = self.weight.data.get(&[oc, ic, kh, kw]).unwrap();
                                        let x_val = input.get(&[b, ic, ih as usize, iw as usize]).unwrap();
                                        sum += w_val * x_val;
                                    }
                                }
                            }
                        }
                        let idx = b * self.out_channels * out_h * out_w
                            + oc * out_h * out_w + oh * out_w + ow;
                        output[idx] = sum;
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, self.out_channels, out_h, out_w]).unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Max Pooling 1D.
pub struct MaxPool1D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool1D {
    pub fn new(kernel_size: usize) -> Self {
        MaxPool1D { kernel_size, stride: kernel_size }
    }

    pub fn forward_tensor(&self, input: &Tensor<f64>) -> Tensor<f64> {
        let shape = input.shape_vec();
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        let out_len = (length - self.kernel_size) / self.stride + 1;

        let mut output = vec![f64::NEG_INFINITY; batch * channels * out_len];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_len {
                    let mut max_val = f64::NEG_INFINITY;
                    for k in 0..self.kernel_size {
                        let val = input.get(&[b, c, ol * self.stride + k]).unwrap();
                        if val > max_val { max_val = val; }
                    }
                    output[b * channels * out_len + c * out_len + ol] = max_val;
                }
            }
        }

        Tensor::new(output, vec![batch, channels, out_len]).unwrap()
    }
}

/// Max Pooling 2D.
pub struct MaxPool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2D {
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2D { kernel_size, stride: kernel_size }
    }

    pub fn forward_tensor(&self, input: &Tensor<f64>) -> Tensor<f64> {
        let shape = input.shape_vec();
        let (batch, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = (h - self.kernel_size) / self.stride + 1;
        let out_w = (w - self.kernel_size) / self.stride + 1;

        let mut output = vec![f64::NEG_INFINITY; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let val = input.get(&[b, c, oh * self.stride + kh, ow * self.stride + kw]).unwrap();
                                if val > max_val { max_val = val; }
                            }
                        }
                        let idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[idx] = max_val;
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, channels, out_h, out_w]).unwrap()
    }
}

/// Average Pooling 2D.
pub struct AvgPool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2D {
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2D { kernel_size, stride: kernel_size }
    }

    pub fn forward_tensor(&self, input: &Tensor<f64>) -> Tensor<f64> {
        let shape = input.shape_vec();
        let (batch, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = (h - self.kernel_size) / self.stride + 1;
        let out_w = (w - self.kernel_size) / self.stride + 1;
        let pool_area = (self.kernel_size * self.kernel_size) as f64;

        let mut output = vec![0.0f64; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                sum += input.get(&[b, c, oh * self.stride + kh, ow * self.stride + kw]).unwrap();
                            }
                        }
                        let idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[idx] = sum / pool_area;
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, channels, out_h, out_w]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d() {
        // [batch=1, channels=1, length=5]
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5]).unwrap();
        let conv = Conv1D::new(1, 2, 3, 1);
        let out = conv.forward_tensor(&input);
        assert_eq!(out.shape_vec(), vec![1, 2, 3]); // (5-3)/1+1 = 3
    }

    #[test]
    fn test_conv2d() {
        // [batch=1, channels=1, H=4, W=4]
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]).unwrap();
        let conv = Conv2D::new(1, 2, 3, 1, 0);
        let out = conv.forward_tensor(&input);
        assert_eq!(out.shape_vec(), vec![1, 2, 2, 2]); // (4-3)/1+1 = 2
    }

    #[test]
    fn test_conv2d_padding() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]).unwrap();
        let conv = Conv2D::new(1, 1, 3, 1, 1); // same padding
        let out = conv.forward_tensor(&input);
        assert_eq!(out.shape_vec(), vec![1, 1, 4, 4]); // same size!
    }

    #[test]
    fn test_maxpool2d() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]).unwrap();
        let pool = MaxPool2D::new(2);
        let out = pool.forward_tensor(&input);
        assert_eq!(out.shape_vec(), vec![1, 1, 2, 2]);
        // Max of [0,1,4,5]=5, [2,3,6,7]=7, [8,9,12,13]=13, [10,11,14,15]=15
        assert_eq!(out.data()[0], 5.0);
        assert_eq!(out.data()[1], 7.0);
        assert_eq!(out.data()[2], 13.0);
        assert_eq!(out.data()[3], 15.0);
    }
}
