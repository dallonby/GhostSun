//! M2.5: GPU profile-model extraction (the dominant pipeline stage).
//!
//! One thread per (frame, slit row): unpack the raw spectrum from the packed
//! SER bytes, run the cubic B-spline IIR prefilter in-thread, scan mu
//! candidates with the linear (C,D) solve, parabola-refine, apply the
//! off-disk fallback taper, and add the PCA residual projection inline
//! (basis computed CPU-side on a frame subsample). Workgroup-shared atomics
//! accumulate the de-smiled per-frame spectra for the telluric anchors.
//!
//! Numerics mirror profile::fit_frame; `ghostsun gpucheck` gates equivalence.

use crate::image2d::Image;
use crate::linefit::LineGeometry;
use crate::mathutil::{pca_topk, polyval};
use crate::profile::{ProfileMaps, ProfileTune};
use crate::ser::SerReader;

const MAX_SPEC_W: usize = 256;
const MAX_SPEC_OFF: usize = 192;
const MAX_WF: usize = 12;

const EXTRACT_WGSL: &str = r#"
struct P {
    sw: u32, sh: u32, spec_w: u32, slit_h: u32,
    transpose: u32, bit16: u32, frames: u32, wf: i32,
    n_mu: u32, pca_k: u32, n_spec: u32, pad0: u32,
    shift: f32, mu_range: f32, depth_gate: f32, pad1: f32,
}
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> smile: array<f32>;
@group(0) @binding(2) var<storage, read> sigma_row: array<f32>;
@group(0) @binding(3) var<storage, read> spec_off: array<f32>;
@group(0) @binding(4) var<storage, read> pca: array<f32>; // mean[nwin] then comps[k][nwin]
@group(0) @binding(5) var<storage, read_write> out3: array<f32>;
@group(0) @binding(6) var<storage, read_write> spec_out: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p: P;
@group(0) @binding(8) var<storage, read> spatial_offset: array<f32>;

var<workgroup> wg_spec: array<atomic<u32>, 200>;
var<workgroup> wg_cnt: atomic<u32>;

fn raw_val_at(f: u32, spec_i: u32, y: u32) -> f32 {
    var sidx: u32;
    if (p.transpose == 1u) {
        sidx = spec_i * p.sw + y;
    } else {
        sidx = y * p.sw + spec_i;
    }
    if (p.bit16 == 1u) {
        let byte = (f * p.sw * p.sh + sidx) * 2u;
        let word = raw[byte >> 2u];
        let sh = (byte & 2u) * 8u;
        return f32((word >> sh) & 0xffffu);
    } else {
        let byte = f * p.sw * p.sh + sidx;
        let word = raw[byte >> 2u];
        let sh = (byte & 3u) * 8u;
        return f32((word >> sh) & 0xffu) * 257.0;
    }
}

fn raw_val(f: u32, spec_i: u32, y: f32) -> f32 {
    let yc = clamp(y, 0.0, f32(p.slit_h - 1u));
    let yi = i32(floor(yc));
    let t = yc - f32(yi);
    let h1 = i32(p.slit_h - 1u);
    let p0 = raw_val_at(f, spec_i, u32(clamp(yi - 1, 0, h1)));
    let p1 = raw_val_at(f, spec_i, u32(clamp(yi, 0, h1)));
    let p2 = raw_val_at(f, spec_i, u32(clamp(yi + 1, 0, h1)));
    let p3 = raw_val_at(f, spec_i, u32(clamp(yi + 2, 0, h1)));
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    return ((a * t + b) * t + c) * t + p1;
}

var<private> cbuf: array<f32, 256>;

fn prefilter(n: i32) {
    let z = -0.2679491924311227;
    let lambda = (1.0 - z) * (1.0 - 1.0 / z);
    for (var i = 0; i < n; i++) { cbuf[i] = cbuf[i] * lambda; }
    let horizon = min(n, 30);
    var sum = cbuf[0];
    var zn = z;
    for (var i = 1; i < horizon; i++) {
        sum = sum + zn * cbuf[i];
        zn = zn * z;
    }
    cbuf[0] = sum;
    for (var i = 1; i < n; i++) { cbuf[i] = cbuf[i] + z * cbuf[i - 1]; }
    cbuf[n - 1] = (z / (z * z - 1.0)) * (z * cbuf[n - 2] + cbuf[n - 1]);
    for (var i = n - 2; i >= 0; i--) { cbuf[i] = z * (cbuf[i + 1] - cbuf[i]); }
}

fn beval(n: i32, x: f32) -> f32 {
    let xf = floor(x);
    let t = x - xf;
    let i = i32(xf);
    let t2 = t * t;
    let t3 = t2 * t;
    let w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    let w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    let w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    let w3 = t3 / 6.0;
    var acc = 0.0;
    var js = array<i32, 4>(i - 1, i, i + 1, i + 2);
    var ws = array<f32, 4>(w0, w1, w2, w3);
    for (var k = 0; k < 4; k++) {
        var j = js[k];
        if (j < 0) { j = -j; }
        if (j >= n) { j = 2 * (n - 1) - j; }
        j = clamp(j, 0, n - 1);
        acc = acc + ws[k] * cbuf[j];
    }
    return acc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // shared spectrum reset
    if (lid.x == 0u) { atomicStore(&wg_cnt, 0u); }
    for (var k = lid.x; k < p.n_spec; k = k + 64u) {
        atomicStore(&wg_spec[k], 0u);
    }
    workgroupBarrier();

    let y = gid.x;
    let f = gid.y;
    let is_on = (y < p.slit_h && f < p.frames);
    var c = 0.0;
    var d = 0.0;
    var mu = 0.0;
    var core = 0.0;
    var depth_out = 0.0;
    var mu_out = -3.0e30;

    if (is_on) {
        // Only the stratified rows used for the per-frame wide spectrum need
        // the entire detector row. Profile fitting touches a narrow interval
        // around the line; the cubic B-spline IIR pole decays as 0.268^d, so
        // 16 guard samples make a local prefilter numerically indistinguishable
        // at the fit window while avoiding ~3/4 of the serial prefilter work.
        let wide_spectrum_row = (y & 7u) == 0u;
        let y_src = clamp(f32(y) + spatial_offset[f], 0.0, f32(p.slit_h - 1u));
        let y0 = u32(floor(y_src));
        let y1 = min(y0 + 1u, p.slit_h - 1u);
        let fy = y_src - f32(y0);
        let center_abs = mix(smile[y0], smile[y1], fy) + p.shift;
        var base: i32 = 0;
        var n = i32(p.spec_w);
        if (!wide_spectrum_row) {
            let reach = p.wf + i32(ceil(p.mu_range)) + 16;
            base = max(i32(floor(center_abs)) - reach, 0);
            let end = min(i32(ceil(center_abs)) + reach + 1, i32(p.spec_w));
            n = end - base;
        }
        for (var i = 0; i < n; i++) { cbuf[i] = raw_val(f, u32(base + i), y_src); }
        prefilter(n);
        let center = center_abs - f32(base);
        let sig = mix(sigma_row[y0], sigma_row[y1], fy);
        let nwin = 2 * p.wf + 1;

        // samples at fixed positions around the smile center
        var xs: array<f32, 25>;
        var ss: array<f32, 25>;
        for (var i = -p.wf; i <= p.wf; i++) {
            let x = clamp(center + f32(i), 1.0, f32(n - 2));
            xs[i + p.wf] = x;
            ss[i + p.wf] = beval(n, x);
        }

        // mu candidate scan
        var best_sse = 3.0e30;
        var best_k = 0;
        var sses: array<f32, 24>;
        var mus: array<f32, 24>;
        for (var k = 0u; k < p.n_mu; k++) {
            let m = center - p.mu_range + 2.0 * p.mu_range * f32(k) / f32(p.n_mu - 1u);
            var sn = 0.0; var sg = 0.0; var sgg = 0.0; var sv = 0.0; var svg = 0.0;
            var svv = 0.0;
            for (var i = 0; i < nwin; i++) {
                let g = exp(-((xs[i] - m) * (xs[i] - m)) / (2.0 * sig * sig));
                sn += 1.0; sg += g; sgg += g * g; sv += ss[i]; svg += ss[i] * g;
                svv += ss[i] * ss[i];
            }
            let det = sn * sgg - sg * sg;
            var sse = 3.0e30;
            if (abs(det) >= 1e-9) {
                let dd = (sv * sg - sn * svg) / det;
                let cc = (sv + dd * sg) / sn;
                // Expanded sum((s - cc + dd*g)^2), using the moments
                // already accumulated above. This removes a duplicate loop
                // and a second set of expensive exponentials per candidate.
                sse = svv - 2.0 * cc * sv + 2.0 * dd * svg
                    + sn * cc * cc - 2.0 * cc * dd * sg + dd * dd * sgg;
            }
            sses[k] = sse;
            mus[k] = m;
            if (sse < best_sse) { best_sse = sse; best_k = i32(k); }
        }
        mu = mus[best_k];
        if (best_k > 0 && best_k + 1 < i32(p.n_mu)) {
            let vm = sses[best_k - 1];
            let v0 = sses[best_k];
            let vp = sses[best_k + 1];
            let den = vm - 2.0 * v0 + vp;
            if (den > 1e-12) {
                let step = mus[1] - mus[0];
                mu = mu + step * clamp(0.5 * (vm - vp) / den, -0.6, 0.6);
            }
        }
        // final (C, D)
        var sn = 0.0; var sg = 0.0; var sgg = 0.0; var sv = 0.0; var svg = 0.0;
        for (var i = 0; i < nwin; i++) {
            let g = exp(-((xs[i] - mu) * (xs[i] - mu)) / (2.0 * sig * sig));
            sn += 1.0; sg += g; sgg += g * g; sv += ss[i]; svg += ss[i] * g;
        }
        let det = sn * sgg - sg * sg;
        if (abs(det) > 1e-9) {
            d = (sv * sg - sn * svg) / det;
            c = (sv + d * sg) / sn;
        } else {
            d = 0.0;
            c = sv / sn;
        }
        var depth = 0.0;
        if (c > 1e-6) { depth = clamp(d / c, -1.0, 1.0); }
        var core_model = c - d;
        let bspl = beval(n, clamp(center, 1.0, f32(n - 2)));
        let t = clamp((depth - p.depth_gate + 0.03) / 0.06, 0.0, 1.0);

        // inline PCA residual projection (matches CPU add-back)
        if (p.pca_k > 0u && t > 0.5 && c > 1.0) {
            var add = pca[u32(p.wf)]; // mean at the center index
            for (var k = 0u; k < p.pca_k; k++) {
                var a = 0.0;
                for (var i = 0; i < nwin; i++) {
                    let x = clamp(mu + f32(i - p.wf), 1.0, f32(n - 2));
                    let s = beval(n, x);
                    let g = exp(-(f32(i - p.wf) * f32(i - p.wf)) / (2.0 * sig * sig));
                    let model = c - d * g;
                    let resid = (s - model) / c;
                    a += (resid - pca[u32(i)]) * pca[(k + 1u) * u32(nwin) + u32(i)];
                }
                add += a * pca[(k + 1u) * u32(nwin) + u32(p.wf)];
            }
            core_model = core_model + add * max(c, 1.0);
        }
        core = max(t * core_model + (1.0 - t) * bspl, 0.0);
        depth_out = max(depth, 0.0) * t;
        if (t > 0.5) { mu_out = mu + f32(base); }

        // De-smiled spectrum contribution.  A full spectrum at every slit
        // row dominated this kernel (hundreds of B-spline evaluations and
        // atomics per output pixel).  A 1-in-8 stratified sample still gives
        // hundreds of illuminated rows per frame, far more than flexure and
        // transparency estimation need, while cutting this work by ~8x.
        // The line-depth gate excludes sky so the continuum bin is also a
        // direct per-frame transparency statistic.
        if (depth > p.depth_gate && wide_spectrum_row) {
            atomicAdd(&wg_cnt, 1u);
            for (var k = 0u; k < p.n_spec; k++) {
                let x = clamp(smile[y] + spec_off[k], 1.0, f32(n - 2));
                let v = u32(clamp(round(beval(n, x)), 0.0, 100000.0));
                atomicAdd(&wg_spec[k], v);
            }
        }

        let o = (f * p.slit_h + y) * 3u;
        out3[o] = core;
        out3[o + 1u] = mu_out;
        out3[o + 2u] = depth_out;
    }

    workgroupBarrier();
    // flush shared spectrum to global (per frame): slot layout
    // frame * (n_spec + 1): [count, spec...]
    if (is_on) {
        for (var k = lid.x; k < p.n_spec; k = k + 64u) {
            let v = atomicLoad(&wg_spec[k]);
            if (v > 0u) {
                atomicAdd(&spec_out[f * (p.n_spec + 1u) + 1u + k], v);
            }
        }
        if (lid.x == 0u) {
            let cval = atomicLoad(&wg_cnt);
            if (cval > 0u) { atomicAdd(&spec_out[f * (p.n_spec + 1u)], cval); }
        }
    }
}
"#;

/// GPU profile extraction mirroring profile::extract_profile.
/// Returns None when the GPU is unavailable or dimensions exceed kernel caps.
#[allow(clippy::too_many_arguments)]
pub fn extract_profile_gpu(
    reader: &SerReader,
    geom: &LineGeometry,
    mean_img: &Image,
    transpose: bool,
    shift: f64,
    tune: &ProfileTune,
    spatial_offsets: Option<&[f64]>,
) -> Option<ProfileMaps> {
    let gpu = crate::gpu::Gpu::get()?;
    let hdr = &reader.header;
    let n = hdr.frame_count;
    let slit_h = if transpose { hdr.width } else { hdr.height };
    let spec_w = if transpose { hdr.height } else { hdr.width };
    if spec_w > MAX_SPEC_W || tune.w_fit > MAX_WF {
        return None;
    }

    // CPU-side prep identical to the CPU path
    let smile: Vec<f64> = (0..slit_h).map(|y| polyval(&geom.coeffs, y as f64)).collect();
    let sigma_row = crate::profile::fit_sigma_rows(mean_img, geom);
    let nwin = 2 * tune.w_fit + 1;
    let iw = mean_img.w as f64;
    let (mut cmin, mut cmax) = (f64::MAX, f64::MIN);
    for y in geom.y1..=geom.y2.min(slit_h - 1) {
        let c = polyval(&geom.coeffs, y as f64);
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    let off_lo = (4.0 - cmin).ceil();
    let off_hi = (iw - 5.0 - cmax).floor();
    let spec_offsets: Vec<f64> = {
        let mut v = Vec::new();
        let mut o = off_lo;
        while o <= off_hi {
            v.push(o);
            o += 1.0;
        }
        v
    };
    if spec_offsets.len() > MAX_SPEC_OFF {
        return None;
    }

    // PCA basis from a CPU subsample (reuses the reference implementation)
    let mut pca_flat: Vec<f32> = vec![0.0; nwin]; // mean (zeros if k=0)
    let mut pca_k = 0usize;
    if tune.pca_k > 0 {
        let mut samples: Vec<Vec<f64>> = Vec::new();
        let mut t = 0;
        while t < n {
            let mut frame = reader.frame(t);
            if transpose {
                frame = frame.transpose();
            }
            let offset = spatial_offsets
                .and_then(|v| v.get(t))
                .copied()
                .unwrap_or(0.0);
            let fit = if offset.abs() >= 1e-6 {
                frame = crate::profile::shift_spatial_cubic(&frame, offset);
                let shifted_smile = crate::profile::shift_series_linear(&smile, offset);
                let shifted_sigma = crate::profile::shift_series_linear(&sigma_row, offset);
                crate::profile::fit_frame(
                    &frame,
                    &shifted_smile,
                    &shifted_sigma,
                    shift,
                    tune,
                    &[],
                )
            } else {
                crate::profile::fit_frame(
                    &frame,
                    &smile,
                    &sigma_row,
                    shift,
                    tune,
                    &[],
                )
            };
            for y in (0..slit_h).step_by(4) {
                if fit.depth[y] > 0.05 && fit.cscale[y] > 1.0 {
                    let v: Vec<f64> =
                        (0..nwin).map(|i| fit.resid[y * nwin + i] as f64).collect();
                    if v.iter().any(|x| x.abs() > 1e-9) {
                        samples.push(v);
                    }
                }
            }
            t += 16;
        }
        if samples.len() > 500 {
            let (comps, mean) = pca_topk(&samples, tune.pca_k, 60);
            pca_k = comps.len();
            pca_flat = mean.iter().map(|&v| v as f32).collect();
            for comp in &comps {
                pca_flat.extend(comp.iter().map(|&v| v as f32));
            }
        }
    }

    // pipeline (compiled once)
    let pipeline = gpu.extract.get_or_init(|| {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("extract"),
            source: wgpu::ShaderSource::Wgsl(EXTRACT_WGSL.into()),
        });
        gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("extract"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
    });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        sw: u32,
        sh: u32,
        spec_w: u32,
        slit_h: u32,
        transpose: u32,
        bit16: u32,
        frames: u32,
        wf: i32,
        n_mu: u32,
        pca_k: u32,
        n_spec: u32,
        pad0: u32,
        shift: f32,
        mu_range: f32,
        depth_gate: f32,
        pad1: f32,
    }

    use wgpu::util::DeviceExt;
    let d = &gpu.device;
    let mk_f32 = |v: &[f32]| {
        d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let smile_f: Vec<f32> = smile.iter().map(|&v| v as f32).collect();
    let sigma_f: Vec<f32> = sigma_row.iter().map(|&v| v as f32).collect();
    let off_f: Vec<f32> = spec_offsets.iter().map(|&v| v as f32).collect();
    let smile_buf = mk_f32(&smile_f);
    let sigma_buf = mk_f32(&sigma_f);
    let off_buf = mk_f32(&off_f);
    let pca_buf = mk_f32(&pca_flat);

    let bpp = reader.bytes_per_px();
    let frame_bytes = hdr.width * hdr.height * bpp;
    let chunk_frames = ((160 << 20) / frame_bytes).clamp(16, 512);
    let n_spec = spec_offsets.len();

    let mut core = Image::new(n, slit_h);
    let mut mu_img = Image::new(n, slit_h);
    let mut depth = Image::new(n, slit_h);
    let mut frame_spec: Vec<Vec<f32>> = vec![vec![0.0; n_spec]; n];

    let mut f0 = 0usize;
    while f0 < n {
        let fc = chunk_frames.min(n - f0);
        let raw = reader.raw_frames(f0, fc);
        // pad to u32 boundary
        let mut raw_padded = raw.to_vec();
        while raw_padded.len() % 4 != 0 {
            raw_padded.push(0);
        }
        let raw_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &raw_padded,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let motion: Vec<f32> = (0..fc)
            .map(|fl| {
                spatial_offsets
                    .and_then(|v| v.get(f0 + fl))
                    .copied()
                    .unwrap_or(0.0) as f32
            })
            .collect();
        let motion_buf = mk_f32(&motion);
        let out_len = fc * slit_h * 3;
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spec_len = fc * (n_spec + 1);
        let spec_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (spec_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let p = P {
            sw: hdr.width as u32,
            sh: hdr.height as u32,
            spec_w: spec_w as u32,
            slit_h: slit_h as u32,
            transpose: transpose as u32,
            bit16: (bpp == 2) as u32,
            frames: fc as u32,
            wf: tune.w_fit as i32,
            n_mu: 21,
            pca_k: pca_k as u32,
            n_spec: n_spec as u32,
            pad0: 0,
            shift: shift as f32,
            mu_range: tune.mu_range as f32,
            depth_gate: tune.depth_gate as f32,
            pad1: 0.0,
        };
        let par_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let read_out = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_spec = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (spec_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: raw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: smile_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sigma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: off_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pca_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: spec_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: par_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: motion_buf.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(((slit_h + 63) / 64) as u32, fc as u32, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_out, 0, (out_len * 4) as u64);
        enc.copy_buffer_to_buffer(&spec_buf, 0, &read_spec, 0, (spec_len * 4) as u64);
        gpu.queue.submit([enc.finish()]);

        let map = |buf: &wgpu::Buffer| -> Option<Vec<u8>> {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            d.poll(wgpu::Maintain::Wait);
            rx.recv().ok()?.ok()?;
            Some(slice.get_mapped_range().to_vec())
        };
        let out_bytes = map(&read_out)?;
        let spec_bytes = map(&read_spec)?;
        let out_f: &[f32] = bytemuck::cast_slice(&out_bytes);
        let spec_u: &[u32] = bytemuck::cast_slice(&spec_bytes);

        for fl in 0..fc {
            let t = f0 + fl;
            for y in 0..slit_h {
                let o = (fl * slit_h + y) * 3;
                core.set(t, y, out_f[o]);
                let m = out_f[o + 1];
                mu_img.set(t, y, if m < -1.0e30 { f32::NAN } else { m });
                depth.set(t, y, out_f[o + 2]);
            }
            let base = fl * (n_spec + 1);
            let cnt = spec_u[base].max(1) as f32;
            for k in 0..n_spec {
                frame_spec[t][k] = spec_u[base + 1 + k] as f32 / cnt;
            }
        }
        f0 += fc;
    }

    Some(ProfileMaps { core, mu: mu_img, depth, frame_spec, spec_offsets })
}
