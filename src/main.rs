use shortcuts::{
    launch,
    memory::{self, ManagedBuffer},
    shader,
    starter_kit::{self, StarterKit},
    FrameDataUbo, MultiPlatformCamera,
};
use watertender::*;

use anyhow::{ensure, Result};
use realsense_rust::{
    base::Rs2Intrinsics,
    config::Config,
    context::Context,
    frame::{CompositeFrame, DepthFrame},
    kind::{Rs2CameraInfo, Rs2DistortionModel, Rs2Format, Rs2ProductLine, Rs2StreamKind},
    pipeline::{ActivePipeline, FrameWaitError, InactivePipeline},
};
use std::{collections::HashSet, convert::TryFrom, time::Duration};

fn mkfloat(x: usize, y: usize, depth: u16) -> [f32; 3] {
    let x = x as f32;
    let y = y as f32;
    let depth = depth as f32;
    [x, y, depth]
}

struct App {
    pointcloud_gpu: Vec<ManagedBuffer>,
    depth_camera_pipeline: ActivePipeline,
    intrinsics: Rs2Intrinsics,
    last_frames: Option<CompositeFrame>,

    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    scene_ubo: FrameDataUbo<SceneData>,
    camera: MultiPlatformCamera,
    anim: f32,
    starter_kit: StarterKit,
}

fn main() -> Result<()> {
    let info = AppInfo::default().validation(true);
    let vr = std::env::args().count() > 1;
    launch::<App>(info, vr)
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct SceneData {
    cameras: [f32; 4 * 4 * 2],
    anim: f32,
    ppx: f32,
    ppy: f32,
    fx: f32,
    fy: f32,
    coeffs: [f32; 5],
}

impl SceneData {
    pub fn new(cameras: [f32; 4 * 4 * 2], anim: f32, intrin: &Rs2Intrinsics) -> Self {
        assert!(intrin.distortion().model == Rs2DistortionModel::BrownConradyInverse);
        SceneData {
            anim,
            cameras,
            ppx: intrin.ppx(),
            ppy: intrin.ppy(),
            fx: intrin.fx(),
            fy: intrin.fy(),
            coeffs: intrin.distortion().coeffs,
        }
    }
}

unsafe impl bytemuck::Zeroable for SceneData {}
unsafe impl bytemuck::Pod for SceneData {}

const DEPTH_WIDTH: usize = 640;
const DEPTH_HEIGHT: usize = 480;
const RS_FRAMERATE: usize = 60;
const N_POINTS: usize = DEPTH_WIDTH * DEPTH_HEIGHT;
const POINTCLOUD_SIZE: usize = N_POINTS * 3 * 2 * std::mem::size_of::<f32>();

impl MainLoop for App {
    fn new(core: &SharedCore, mut platform: Platform<'_>) -> Result<Self> {
        // ############################## Realsense stuff ##############################
        let mut queried_devices = HashSet::new();
        queried_devices.insert(Rs2ProductLine::Sr300);
        let context = Context::new()?;
        let devices = context.query_devices(queried_devices);
        ensure!(!devices.is_empty(), "No devices found");

        // create pipeline
        let pipeline = InactivePipeline::try_from(&context)?;
        let mut config = Config::new();
        config
            .enable_device_from_serial(devices[0].info(Rs2CameraInfo::SerialNumber).unwrap())?
            .disable_all_streams()?
            .enable_stream(
                Rs2StreamKind::Depth,
                None,
                DEPTH_WIDTH,
                DEPTH_HEIGHT,
                Rs2Format::Z16,
                RS_FRAMERATE,
            )?;
        let depth_camera_pipeline = pipeline.start(Some(config))?;

        let intrinsics = depth_camera_pipeline
            .profile()
            .streams()
            .get(0)
            .unwrap()
            .intrinsics()?;

        // ############################## Vulkan graphics stuff ##############################
        let starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        const SCENE_DATA_BINDING: u32 = 0;
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(SCENE_DATA_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

        let descriptor_set_layout_ci =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None, None)
        }
        .result()?;

        // Scene data
        let scene_ubo = FrameDataUbo::new(
            core.clone(),
            starter_kit::FRAMES_IN_FLIGHT,
            descriptor_set_layout,
            SCENE_DATA_BINDING,
        )?;

        let descriptor_set_layouts = [descriptor_set_layout];

        // Pipeline layout
        let push_constant_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<[f32; 4 * 4]>() as u32)];

        let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        // Pipeline
        let pipeline = shader(
            core,
            include_bytes!("unlit.vert.spv"),
            include_bytes!("unlit.frag.spv"),
            vk::PrimitiveTopology::POINT_LIST,
            starter_kit.render_pass,
            pipeline_layout,
        )?;

        // Pointcloud buffers
        let ci = vk::BufferCreateInfoBuilder::new()
            .size(POINTCLOUD_SIZE as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER);

        let pointcloud_gpu = (0..starter_kit::FRAMES_IN_FLIGHT)
            .map(|_| ManagedBuffer::new(core.clone(), ci, memory::UsageFlags::UPLOAD))
            .collect::<Result<_>>()?;

        Ok(Self {
            last_frames: None,
            intrinsics,
            depth_camera_pipeline,
            pointcloud_gpu,
            camera,
            anim: 0.0,
            pipeline_layout,
            scene_ubo,
            pipeline,
            starter_kit,
        })
    }

    fn frame(
        &mut self,
        frame: Frame,
        core: &SharedCore,
        platform: Platform<'_>,
    ) -> Result<PlatformReturn> {
        // ############################## Realsense to Vulkan ##############################
        let timeout = Duration::from_millis(0);
        let last_frames = match &self.last_frames {
            Some(l) => l,
            None => {
                self.last_frames = Some(self.depth_camera_pipeline.wait(None)?);
                self.last_frames.as_ref().unwrap()
            }
        };

        let frames = match self.depth_camera_pipeline.wait(Some(timeout)) {
            Ok(f) => {
                self.last_frames = Some(f);
                self.last_frames.as_ref().unwrap()
            }
            Err(FrameWaitError::DidTimeoutBeforeFrameArrival) => last_frames,
            Err(e) => Err(e)?,
        };

        let mut depth_frames = frames.frames_of_type::<DepthFrame>();

        if !depth_frames.is_empty() {
            let depth_frame = depth_frames.pop().unwrap();
            let size = depth_frame.get_data_size() / std::mem::size_of::<u16>();
            //let stride = depth_frame.stride() / std::mem::size_of::<u16>();
            let mut pts = Vec::with_capacity(POINTCLOUD_SIZE);
            unsafe {
                let data = depth_frame.get_data() as *const std::os::raw::c_void;
                let slice = std::slice::from_raw_parts(data.cast::<u16>(), size);
                for (row_idx, row) in slice.chunks_exact(DEPTH_WIDTH).enumerate() {
                    for (col_idx, depth) in row.iter().enumerate() {
                        let [x, y, z] = mkfloat(col_idx, row_idx, *depth);
                        pts.extend_from_slice(&[x, y, z]);
                        pts.extend_from_slice(&[
                            col_idx as f32 / DEPTH_WIDTH as f32,
                            row_idx as f32 / DEPTH_HEIGHT as f32,
                            0.,
                        ]);
                    }
                }
            }
            self.pointcloud_gpu[self.starter_kit.frame]
                .write_bytes(0, bytemuck::cast_slice(&pts))?;
        }

        // ############################## Command buffer ##############################
        let cmd = self.starter_kit.begin_command_buffer(frame)?;
        let command_buffer = cmd.command_buffer;
        unsafe {
            core.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.scene_ubo.descriptor_set(self.starter_kit.frame)],
                &[],
            );

            // Draw cmds
            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            /*
            let matrix = nalgebra::Matrix4::from_euler_angles(0., self.anim, 0.);

            core.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::mem::size_of::<[f32; 4 * 4]>() as u32,
                matrix.as_ptr() as _,
            );
            */

            core.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.pointcloud_gpu[self.starter_kit.frame].instance()],
                &[0],
            );

            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            core.device
                .cmd_draw(command_buffer, N_POINTS as u32, 1, 0, 0);
        }

        let (ret, cameras) = self.camera.get_matrices(platform)?;

        self.scene_ubo.upload(
            self.starter_kit.frame,
            &SceneData::new(cameras, self.anim, &self.intrinsics),
        )?;

        self.anim += 0.001;

        // End draw cmds
        self.starter_kit.end_command_buffer(cmd)?;

        Ok(ret)
    }

    fn swapchain_resize(&mut self, images: Vec<vk::Image>, extent: vk::Extent2D) -> Result<()> {
        self.starter_kit.swapchain_resize(images, extent)
    }

    fn event(
        &mut self,
        mut event: PlatformEvent<'_, '_>,
        _core: &Core,
        mut platform: Platform<'_>,
    ) -> Result<()> {
        self.camera.handle_event(&mut event, &mut platform);
        starter_kit::close_when_asked(event, platform);
        Ok(())
    }
}

impl SyncMainLoop for App {
    fn winit_sync(&self) -> (vk::Semaphore, vk::Semaphore) {
        self.starter_kit.winit_sync()
    }
}
