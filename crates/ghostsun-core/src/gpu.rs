//! M2: GPU compute kernels via wgpu — Metal on macOS and Direct3D 12/Vulkan
//! on Windows.
//!
//! Kernels with CPU counterparts are compared by `ghostsun gpucheck` (max
//! relative difference plus timing). CPU paths remain the fallback for profile
//! extraction, temporal NLM, and warping. The residual column-state model is a
//! GPU-only inverse stage with synthetic artifact and clean-scan regressions.

use crate::image2d::Image;
use std::sync::OnceLock;

pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    nlm: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
    demix: wgpu::ComputePipeline,
    pub(crate) extract: OnceLock<wgpu::ComputePipeline>,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

impl Gpu {
    pub fn get() -> Option<&'static Gpu> {
        GPU.get_or_init(|| Gpu::init()).as_ref()
    }

    fn init() -> Option<Gpu> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        // large storage buffers: a full-scan disk is ~140 MB
        let mut limits = wgpu::Limits::downlevel_defaults();
        let al = adapter.limits();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_compute_workgroups_per_dimension = al.max_compute_workgroups_per_dimension;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage;
        limits.max_bindings_per_bind_group = al.max_bindings_per_bind_group;
        limits.max_compute_invocations_per_workgroup = al.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = al.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ghostsun"),
                required_limits: limits,
                ..Default::default()
            },
            None,
        ))
        .ok()?;
        let make = |src: &str, entry: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let nlm = make(NLM_WGSL, "nlm");
        let warp = make(WARP_WGSL, "warp");
        let demix = make(COLUMN_DEMIX_WGSL, "column-demix");
        Some(Gpu { device, queue, nlm, warp, demix, extract: OnceLock::new() })
    }

    /// Run a compute pipeline: src f32 buffer + uniform params -> dst f32.
    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        src: &[f32],
        dst_len: usize,
        params: &[u8],
        groups: (u32, u32),
    ) -> Option<Vec<f32>> {
        use wgpu::util::DeviceExt;
        let d = &self.device;
        let src_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(src),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dst_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (dst_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let par_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let read_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (dst_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: par_buf.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        enc.copy_buffer_to_buffer(&dst_buf, 0, &read_buf, 0, (dst_len * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Joint residual column-state demixing
// ---------------------------------------------------------------------------

// One workgroup owns one acquisition column. The latent clean column is
// predicted by a symmetric, multi-baseline temporal consensus.
// Thousands of slit samples then robustly constrain five small nuisance modes:
// multiplicative gain, additive bias, scan displacement, slit displacement,
// and isotropic blur.  The normal equations are reduced and solved inside the
// workgroup.  Photometric terms are inverted on the original column (rather
// than replacing it with the prediction), so its fine detail is retained.
const COLUMN_DEMIX_WGSL: &str = r#"
struct P {
    w: u32, h: u32, pad0: u32, pad1: u32,
    level: f32, thresh: f32, tukey_cut: f32, strength: f32,
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

const WG: u32 = 256u;
var<workgroup> s00: array<f32, 256>;
var<workgroup> s01: array<f32, 256>;
var<workgroup> s02: array<f32, 256>;
var<workgroup> s03: array<f32, 256>;
var<workgroup> s04: array<f32, 256>;
var<workgroup> s11: array<f32, 256>;
var<workgroup> s12: array<f32, 256>;
var<workgroup> s13: array<f32, 256>;
var<workgroup> s14: array<f32, 256>;
var<workgroup> s22: array<f32, 256>;
var<workgroup> s23: array<f32, 256>;
var<workgroup> s24: array<f32, 256>;
var<workgroup> s33: array<f32, 256>;
var<workgroup> s34: array<f32, 256>;
var<workgroup> s44: array<f32, 256>;
var<workgroup> sb0: array<f32, 256>;
var<workgroup> sb1: array<f32, 256>;
var<workgroup> sb2: array<f32, 256>;
var<workgroup> sb3: array<f32, 256>;
var<workgroup> sb4: array<f32, 256>;
var<workgroup> scount: array<f32, 256>;
var<workgroup> coeff: array<f32, 5>;

fn at(x: i32, y: i32) -> f32 {
    let xx = u32(clamp(x, 0, i32(p.w) - 1));
    let yy = u32(clamp(y, 0, i32(p.h) - 1));
    return src[yy * p.w + xx];
}

fn prediction(x: i32, y: i32) -> f32 {
    // Two curvature-preserving cubic predictions. The close stencil resolves
    // one-column teeth; the wider stencil supplies an anchor through short
    // runs. Disagreement is used only to estimate the state shared across y.
    let close = (2.0 / 3.0) * (at(x - 1, y) + at(x + 1, y))
              - (1.0 / 6.0) * (at(x - 2, y) + at(x + 2, y));
    let wide = (2.0 / 3.0) * (at(x - 4, y) + at(x + 4, y))
             - (1.0 / 6.0) * (at(x - 8, y) + at(x + 8, y));
    return 0.35 * close + 0.65 * wide;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let x = i32(wid.x);
    let lid = lid3.x;
    let edge = x < 8 || x + 8 >= i32(p.w);
    let inv_level = 1.0 / max(p.level, 1.0e-6);

    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0; var a04 = 0.0;
    var a11 = 0.0; var a12 = 0.0; var a13 = 0.0; var a14 = 0.0;
    var a22 = 0.0; var a23 = 0.0; var a24 = 0.0;
    var a33 = 0.0; var a34 = 0.0; var a44 = 0.0;
    var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0; var b4 = 0.0;
    var count = 0.0;

    if (!edge) {
        for (var yu = lid; yu < p.h; yu = yu + WG) {
            let y = i32(yu);
            let pred = prediction(x, y);
            let obs = at(x, y);
            if (pred > 0.02 * p.thresh || obs > 0.02 * p.thresh) {
                let xm = at(x - 1, y);
                let xp = at(x + 1, y);
                let ref_m = prediction(x, y - 1);
                let ref_p = prediction(x, y + 1);
                let dx = 0.5 * (xp - xm);
                let dy = 0.5 * (ref_p - ref_m);
                let lap = xm + xp + ref_m + ref_p - 4.0 * pred;
                let residual = obs - pred;

                // Normalize all columns of the tiny design matrix to the same
                // physical intensity scale. Tukey weighting rejects active
                // regions and genuine structure that the temporal predictor
                // cannot reproduce.
                let illuminated = pred > p.thresh && obs > 0.25 * p.thresh;
                let photometric_gate = select(0.0, 1.0, illuminated);
                let f0 = photometric_gate * clamp(pred * inv_level, 0.0, 8.0);
                let f1 = photometric_gate * clamp(dx * inv_level, -4.0, 4.0);
                let f2 = photometric_gate * clamp(dy * inv_level, -4.0, 4.0);
                let f3 = photometric_gate * clamp(lap * inv_level, -4.0, 4.0);
                // A constant detector/extraction bias is visible even outside
                // the disk.  Low-signal rows constrain only this mode.
                let f4 = 1.0;
                let rr = residual * inv_level;
                // The two high-gradient limb intersections are a physical
                // round-disk constraint shared by the whole column. Give
                // them extra leverage and a wider robust basin: matched
                // top/bottom brightness loads the gain mode, common motion
                // loads dy, and opposite chord motion loads dx.
                let limb = clamp(abs(dy) * inv_level, 0.0, 1.0);
                let robust_scale = (p.tukey_cut + 0.25 * limb) * p.level;
                let u = abs(residual) / max(robust_scale, 1.0e-6);
                var wt = 0.0;
                if (u < 1.0) {
                    let z = 1.0 - u * u;
                    let signal_weight = select(0.20, 1.0 + 2.0 * limb, illuminated);
                    wt = z * z * signal_weight;
                }
                a00 += wt*f0*f0; a01 += wt*f0*f1; a02 += wt*f0*f2; a03 += wt*f0*f3; a04 += wt*f0*f4;
                a11 += wt*f1*f1; a12 += wt*f1*f2; a13 += wt*f1*f3; a14 += wt*f1*f4;
                a22 += wt*f2*f2; a23 += wt*f2*f3; a24 += wt*f2*f4;
                a33 += wt*f3*f3; a34 += wt*f3*f4; a44 += wt*f4*f4;
                b0 += wt*f0*rr; b1 += wt*f1*rr; b2 += wt*f2*rr; b3 += wt*f3*rr; b4 += wt*f4*rr;
                if (wt > 0.05) { count += 1.0; }
            }
        }
    }

    s00[lid]=a00; s01[lid]=a01; s02[lid]=a02; s03[lid]=a03; s04[lid]=a04;
    s11[lid]=a11; s12[lid]=a12; s13[lid]=a13; s14[lid]=a14;
    s22[lid]=a22; s23[lid]=a23; s24[lid]=a24;
    s33[lid]=a33; s34[lid]=a34; s44[lid]=a44;
    sb0[lid]=b0; sb1[lid]=b1; sb2[lid]=b2; sb3[lid]=b3; sb4[lid]=b4; scount[lid]=count;
    workgroupBarrier();

    var stride = WG / 2u;
    while (stride > 0u) {
        if (lid < stride) {
            s00[lid]+=s00[lid+stride]; s01[lid]+=s01[lid+stride];
            s02[lid]+=s02[lid+stride]; s03[lid]+=s03[lid+stride]; s04[lid]+=s04[lid+stride];
            s11[lid]+=s11[lid+stride]; s12[lid]+=s12[lid+stride];
            s13[lid]+=s13[lid+stride]; s14[lid]+=s14[lid+stride];
            s22[lid]+=s22[lid+stride]; s23[lid]+=s23[lid+stride]; s24[lid]+=s24[lid+stride];
            s33[lid]+=s33[lid+stride]; s34[lid]+=s34[lid+stride]; s44[lid]+=s44[lid+stride];
            sb0[lid]+=sb0[lid+stride]; sb1[lid]+=sb1[lid+stride];
            sb2[lid]+=sb2[lid+stride]; sb3[lid]+=sb3[lid+stride]; sb4[lid]+=sb4[lid+stride];
            scount[lid]+=scount[lid+stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid == 0u) {
        coeff[0]=0.0; coeff[1]=0.0; coeff[2]=0.0; coeff[3]=0.0; coeff[4]=0.0;
        if (!edge && scount[0] >= 128.0) {
            // Ridge strengths encode conservative priors: residual gain is
            // easiest to identify; geometry is bounded below; blur requires
            // substantially more evidence before it is changed.
            var m = array<array<f32, 6>, 5>(
                array<f32, 6>(s00[0]+2.0, s01[0], s02[0], s03[0], s04[0], sb0[0]),
                array<f32, 6>(s01[0], s11[0]+12.0, s12[0], s13[0], s14[0], sb1[0]),
                array<f32, 6>(s02[0], s12[0], s22[0]+12.0, s23[0], s24[0], sb2[0]),
                array<f32, 6>(s03[0], s13[0], s23[0], s33[0]+40.0, s34[0], sb3[0]),
                array<f32, 6>(s04[0], s14[0], s24[0], s34[0], s44[0]+4.0, sb4[0])
            );
            // Five-variable Gauss-Jordan solve with partial pivoting.
            for (var k = 0u; k < 5u; k++) {
                var pivot = k;
                var pv = abs(m[k][k]);
                for (var r = k + 1u; r < 5u; r++) {
                    if (abs(m[r][k]) > pv) { pivot = r; pv = abs(m[r][k]); }
                }
                if (pivot != k) {
                    let tmp = m[k]; m[k] = m[pivot]; m[pivot] = tmp;
                }
                if (abs(m[k][k]) > 1.0e-8) {
                    let d = m[k][k];
                    for (var j = k; j < 6u; j++) { m[k][j] /= d; }
                    for (var r = 0u; r < 5u; r++) {
                        if (r != k) {
                            let q = m[r][k];
                            for (var j = k; j < 6u; j++) { m[r][j] -= q * m[k][j]; }
                        }
                    }
                }
            }
            coeff[0] = clamp(m[0][5], -0.10, 0.10);
            coeff[1] = clamp(m[1][5], -0.40, 0.40);
            coeff[2] = clamp(m[2][5], -0.40, 0.40);
            coeff[3] = clamp(m[3][5], -0.20, 0.20);
            coeff[4] = clamp(m[4][5], -0.10, 0.10);
        }
        for (var k = 0u; k < 5u; k++) { state[wid.x * 5u + k] = coeff[k]; }
    }
    workgroupBarrier();

    for (var yu = lid; yu < p.h; yu = yu + WG) {
        let y = i32(yu);
        let idx = yu * p.w + wid.x;
        if (edge) { dst[idx] = src[idx]; continue; }
        let pred = prediction(x, y);
        let xm = at(x - 1, y);
        let xp = at(x + 1, y);
        let ref_m = prediction(x, y - 1);
        let ref_p = prediction(x, y + 1);
        let dx = 0.5 * (xp - xm);
        let dy = 0.5 * (ref_p - ref_m);
        let lap = xm + xp + ref_m + ref_p - 4.0 * pred;
        // Invert the shared photometric state on the original sample: this
        // normalizes the whole column without substituting neighbour pixels.
        var corrected = src[idx] - p.strength * coeff[4] * p.level;
        if (pred > 0.35 * p.thresh) {
            corrected = corrected / max(1.0 + p.strength * coeff[0], 0.80);
            corrected -= p.strength * (coeff[1]*dx + coeff[2]*dy + coeff[3]*lap);
        }
        dst[idx] = max(corrected, 0.0);
    }
}
"#;

pub struct ColumnState {
    pub gain: Vec<f64>,
    pub x_shift: Vec<f64>,
    pub y_shift: Vec<f64>,
    pub blur: Vec<f64>,
    /// Additive bias as a fraction of the robust intensity level.
    pub offset: Vec<f64>,
}

/// Jointly estimate and remove column-coherent acquisition modes on the GPU.
pub fn demix_columns(disk: &Image, strength: f32) -> Option<(Image, ColumnState)> {
    if disk.w < 9 || disk.h < 256 {
        return None;
    }
    let gpu = Gpu::get()?;
    let level = crate::mathutil::percentile_f32(&disk.data, 80.0).max(1.0);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        w: u32,
        h: u32,
        pad0: u32,
        pad1: u32,
        level: f32,
        thresh: f32,
        tukey_cut: f32,
        strength: f32,
    }
    let p = P {
        w: disk.w as u32,
        h: disk.h as u32,
        pad0: 0,
        pad1: 0,
        level,
        thresh: level * 0.18,
        tukey_cut: 0.12,
        strength: strength.clamp(0.0, 1.0),
    };

    use wgpu::util::DeviceExt;
    let d = &gpu.device;
    let src_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("column-demix-src"),
        contents: bytemuck::cast_slice(&disk.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let image_bytes = (disk.data.len() * 4) as u64;
    let state_bytes = (disk.w * 5 * 4) as u64;
    let dst_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("column-demix-dst"), size: image_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let state_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("column-demix-state"), size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let par_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("column-demix-params"), contents: bytemuck::bytes_of(&p),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let read_image = d.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: image_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read_state = d.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: state_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = gpu.demix.get_bind_group_layout(0);
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("column-demix-bind"), layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: par_buf.as_entire_binding() },
        ],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&gpu.demix);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(disk.w as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&dst_buf, 0, &read_image, 0, image_bytes);
    enc.copy_buffer_to_buffer(&state_buf, 0, &read_state, 0, state_bytes);
    gpu.queue.submit([enc.finish()]);

    let map = |buf: &wgpu::Buffer| -> Option<Vec<u8>> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        d.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;
        Some(slice.get_mapped_range().to_vec())
    };
    let image_data: Vec<f32> = bytemuck::cast_slice(&map(&read_image)?).to_vec();
    let state_data: Vec<f32> = bytemuck::cast_slice(&map(&read_state)?).to_vec();
    let unpack = |k: usize| -> Vec<f64> {
        (0..disk.w).map(|x| state_data[x * 5 + k] as f64).collect()
    };
    let mut state = ColumnState {
        gain: unpack(0), x_shift: unpack(1), y_shift: unpack(2), blur: unpack(3),
        offset: unpack(4),
    };
    // Scan-level evidence gate. A clean scan produces tiny isolated fit
    // coefficients; a real column pattern appears coherently in many frames.
    // Preserve all subtle coefficients once that global evidence exists, but
    // return pristine inputs bit-for-bit when it does not.
    let need = (disk.w / 120).max(12);
    let count_over = |v: &[f64], t: f64| v.iter().filter(|x| x.abs() >= t).count();
    // Strength is also the user's sensitivity control: aggressive settings
    // admit fainter coherent patterns; conservative settings demand stronger
    // evidence. The gate is recomputed independently for every SER scan.
    let sensitivity = (strength as f64).clamp(0.25, 1.0);
    let enabled = strength > 0.0
        && (count_over(&state.gain, 0.003 / sensitivity) >= need
            || count_over(&state.offset, 0.002 / sensitivity) >= need
            || count_over(&state.x_shift, 0.020 / sensitivity) >= need
            || count_over(&state.y_shift, 0.020 / sensitivity) >= need
            || count_over(&state.blur, 0.015 / sensitivity) >= need);
    let mut out = Image::new(disk.w, disk.h);
    if enabled {
        out.data = image_data;
    } else {
        out.data.clone_from(&disk.data);
        state.gain.fill(0.0);
        state.x_shift.fill(0.0);
        state.y_shift.fill(0.0);
        state.blur.fill(0.0);
        state.offset.fill(0.0);
    }
    Some((out, state))
}

// ---------------------------------------------------------------------------
// Temporal NLM (port of quality::temporal_nlm inner loop)
// ---------------------------------------------------------------------------

const NLM_WGSL: &str = r#"
struct P { w: u32, h: u32, radius: i32, pad0: u32, h2: f32, sig2: f32, thresh: f32, pad1: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= i32(p.w) || y >= i32(p.h)) { return; }
    let idx = gid.y * p.w + gid.x;
    let c = src[idx];
    if (c <= p.thresh) { dst[idx] = c; return; }
    var acc: f32 = c;
    var wsum: f32 = 1.0;
    for (var dt: i32 = -p.radius; dt <= p.radius; dt = dt + 1) {
        if (dt == 0) { continue; }
        let xx = x + dt;
        if (xx < 0 || xx >= i32(p.w)) { continue; }
        var d2: f32 = 0.0;
        for (var pp: i32 = -3; pp <= 3; pp = pp + 1) {
            let yy = clamp(y + pp, 0, i32(p.h) - 1);
            let d = src[u32(yy) * p.w + u32(x)] - src[u32(yy) * p.w + u32(xx)];
            d2 = d2 + d * d;
        }
        d2 = d2 / 7.0;
        let excess = max(d2 - 2.0 * p.sig2, 0.0);
        let wgt = exp(-excess / p.h2);
        acc = acc + wgt * src[u32(y) * p.w + u32(xx)];
        wsum = wsum + wgt;
    }
    dst[idx] = acc / wsum;
}
"#;

/// GPU temporal NLM. Parameters are the exact values the CPU path derives.
pub fn temporal_nlm(disk: &Image, radius: usize, h2: f64, sigma: f64, thresh: f32) -> Option<Image> {
    let gpu = Gpu::get()?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        w: u32,
        h: u32,
        radius: i32,
        pad0: u32,
        h2: f32,
        sig2: f32,
        thresh: f32,
        pad1: f32,
    }
    let p = P {
        w: disk.w as u32,
        h: disk.h as u32,
        radius: radius as i32,
        pad0: 0,
        h2: h2 as f32,
        sig2: (sigma * sigma) as f32,
        thresh,
        pad1: 0.0,
    };
    let groups = ((disk.w as u32 + 15) / 16, (disk.h as u32 + 15) / 16);
    let out = gpu.run(&gpu.nlm, &disk.data, disk.w * disk.h, bytemuck::bytes_of(&p), groups)?;
    let mut img = Image::new(disk.w, disk.h);
    img.data = out;
    Some(img)
}

// ---------------------------------------------------------------------------
// Single composed warp (port of warp::warp_single + sample_lanczos3_aniso)
// ---------------------------------------------------------------------------

const WARP_WGSL: &str = r#"
struct P {
    in_w: u32, in_h: u32, out_size: u32, allow_neg: u32,
    sx: f32, shear: f32, xc: f32, yc: f32,
    ct: f32, st: f32, fx: f32, fy: f32,
    oc: f32, kx: f32, pad0: f32, pad1: f32,
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

const PI: f32 = 3.14159265358979;

fn at_clamped(x: i32, y: i32) -> f32 {
    let xc = clamp(x, 0, i32(p.in_w) - 1);
    let yc = clamp(y, 0, i32(p.in_h) - 1);
    return src[u32(yc) * p.in_w + u32(xc)];
}

fn lanczos3(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-9) { return 1.0; }
    if (ax >= 3.0) { return 0.0; }
    let pix = PI * x;
    return 3.0 * (sin(pix) * sin(pix / 3.0)) / (pix * pix);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.out_size || gid.y >= p.out_size) { return; }
    let u0 = f32(gid.x) - p.oc;
    let v0 = f32(gid.y) - p.oc;
    let u1 = u0 * p.fx;
    let v1 = v0 * p.fy;
    let u = u1 * p.ct - v1 * p.st;
    let v = u1 * p.st + v1 * p.ct;
    let x = p.sx * (u + p.shear * v) + p.xc;
    let y = v + p.yc;

    let sxk = max(p.kx, 1.0);
    let rx = i32(ceil(3.0 * sxk));
    let xf = i32(floor(x));
    let yf = i32(floor(y));
    var acc: f32 = 0.0;
    var wsum: f32 = 0.0;
    if (xf >= -3 - rx && yf >= -3 && xf <= i32(p.in_w) + 2 + rx && yf <= i32(p.in_h) + 2) {
        for (var j: i32 = 0; j < 6; j = j + 1) {
            let yy = yf - 2 + j;
            let wy = lanczos3(y - f32(yy));
            if (wy == 0.0) { continue; }
            for (var i: i32 = -rx; i <= rx; i = i + 1) {
                let xx = xf + i;
                let w = lanczos3((x - f32(xx)) / sxk) * wy;
                if (w == 0.0) { continue; }
                acc = acc + w * at_clamped(xx, yy);
                wsum = wsum + w;
            }
        }
    }
    var out: f32 = 0.0;
    if (abs(wsum) >= 1e-12) { out = acc / wsum; }
    if (p.allow_neg == 0u) { out = max(out, 0.0); }
    dst[gid.y * p.out_size + gid.x] = out;
}
"#;

/// GPU single composed warp mirroring warp::warp_single.
pub fn warp_single(
    disk: &Image,
    geom: &crate::ellipse::EllipseGeom,
    wp: &crate::warp::WarpParams,
) -> Option<crate::warp::WarpOutput> {
    let gpu = Gpu::get()?;
    let r = geom.radius;
    let margin = (wp.margin_frac * r).max(40.0);
    let size = (2.0 * (r + margin)).ceil() as usize;
    let oc = size as f64 / 2.0;
    let th = wp.rotation_deg.to_radians();
    let kx = if wp.filtered_downscale { geom.sx.max(1.0) } else { 1.0 };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        in_w: u32,
        in_h: u32,
        out_size: u32,
        allow_neg: u32,
        sx: f32,
        shear: f32,
        xc: f32,
        yc: f32,
        ct: f32,
        st: f32,
        fx: f32,
        fy: f32,
        oc: f32,
        kx: f32,
        pad0: f32,
        pad1: f32,
    }
    let p = P {
        in_w: disk.w as u32,
        in_h: disk.h as u32,
        out_size: size as u32,
        allow_neg: wp.allow_negative as u32,
        sx: geom.sx as f32,
        shear: geom.shear as f32,
        xc: geom.xc as f32,
        yc: geom.yc as f32,
        ct: th.cos() as f32,
        st: th.sin() as f32,
        fx: if wp.flip_x { -1.0 } else { 1.0 },
        fy: if wp.flip_y { -1.0 } else { 1.0 },
        oc: oc as f32,
        kx: kx as f32,
        pad0: 0.0,
        pad1: 0.0,
    };
    let groups = ((size as u32 + 15) / 16, (size as u32 + 15) / 16);
    let out = gpu.run(&gpu.warp, &disk.data, size * size, bytemuck::bytes_of(&p), groups)?;
    let mut image = Image::new(size, size);
    image.data = out;
    Some(crate::warp::WarpOutput { image, xc: oc, yc: oc, radius: r })
}
