pub mod modules;
pub mod layers;
pub mod nlp {
    pub use crate::modules::nlp::*;
}
pub mod cv {
    pub use crate::modules::cv::*;
}

pub use cv::*;
pub use nlp::*;
