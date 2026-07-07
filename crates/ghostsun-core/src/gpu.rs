//! M2: GPU compute kernels via wgpu — Metal on Apple Silicon.
//!
//! Every kernel is a direct port of its CPU counterpart and must pass the
//! `ghostsun gpucheck` equivalence gate (max relative difference against
//! the CPU implementation). CPU paths remain the reference and fallback:
//! all entry points return Option and the pipeline silently falls back.

use crate::image2d::Image;
use std::sync::OnceLock;

pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    nlm: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
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
        Some(Gpu { device, queue, nlm, warp, extract: OnceLock::new() })
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
