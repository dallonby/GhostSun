//! GhostSun core: high-fidelity solar disk reconstruction from
//! spectroheliograph SER scans. See README.md at the workspace root.

pub mod deconv;
pub mod denoise;
pub mod ellipse;
pub mod extract;
pub mod flatfield;
pub mod gpu;
pub mod gpu_extract;
pub mod image2d;
pub mod jitter;
pub mod limb;
pub mod linefit;
pub mod lines;
pub mod mathutil;
pub mod metrics;
pub mod output;
pub mod pipeline;
pub mod profile;
pub mod quality;
pub mod render;
pub mod ser;
pub mod stack;
pub mod synth;
pub mod warp;
