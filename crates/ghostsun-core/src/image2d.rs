//! Row-major f32 image container used throughout the pipeline.
//! All processing stays in f32; quantization happens once at output.

#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Self {
        Image { w, h, data: vec![0.0; w * h] }
    }

    #[inline(always)]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.w + x] = v;
    }

    #[inline(always)]
    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.w..(y + 1) * self.w]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        &mut self.data[y * self.w..(y + 1) * self.w]
    }

    /// Clamped pixel access (replicate border).
    #[inline(always)]
    pub fn at_clamped(&self, x: isize, y: isize) -> f32 {
        let xc = x.clamp(0, self.w as isize - 1) as usize;
        let yc = y.clamp(0, self.h as isize - 1) as usize;
        self.at(xc, yc)
    }

    pub fn column(&self, x: usize) -> Vec<f32> {
        (0..self.h).map(|y| self.at(x, y)).collect()
    }

    pub fn set_column(&mut self, x: usize, col: &[f32]) {
        for (y, &v) in col.iter().enumerate() {
            self.set(x, y, v);
        }
    }

    #[allow(dead_code)]
    pub fn mean(&self) -> f64 {
        self.data.iter().map(|&v| v as f64).sum::<f64>() / self.data.len() as f64
    }

    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::MIN, f32::max)
    }

    pub fn transpose(&self) -> Image {
        let mut out = Image::new(self.h, self.w);
        for y in 0..self.h {
            for x in 0..self.w {
                out.set(y, x, self.at(x, y));
            }
        }
        out
    }
}
