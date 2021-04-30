use anyhow::Result;
use shortcuts::{
    launch,
    mesh::*,
    shader,
    starter_kit::{self, StarterKit},
    memory::{self, ManagedBuffer},
    FrameDataUbo, MultiPlatformCamera, StagingBuffer, Vertex,
};
use watertender::*;

struct App {
    pointcloud_gpu: Vec<ManagedBuffer>,

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
}

unsafe impl bytemuck::Zeroable for SceneData {}
unsafe impl bytemuck::Pod for SceneData {}

const N_POINTS: usize = 1024;
const POINTCLOUD_SIZE: usize = 1024 * 3 * 2 * std::mem::size_of::<f32>();

impl MainLoop for App {
    fn new(core: &SharedCore, mut platform: Platform<'_>) -> Result<Self> {
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

        let pointcloud_gpu = (0..starter_kit::FRAMES_IN_FLIGHT).map(|_| {
            ManagedBuffer::new(core.clone(), ci, memory::UsageFlags::UPLOAD)
        }).collect::<Result<_>>()?;

        Ok(Self {
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
        let cmd = self.starter_kit.begin_command_buffer(frame)?;
        let command_buffer = cmd.command_buffer;

        let mut pts = Vec::with_capacity(POINTCLOUD_SIZE);
        for i in 0..N_POINTS {
            let i = i as f32 / 100.;
            let x = (self.anim + i).cos();
            let y = (self.anim * i).sin();
            let z = (self.anim - i).sin();
            pts.extend_from_slice(&[x, y, z]);
            let (r, g, b) = (x.abs(), y.abs(), z.abs());
            pts.extend_from_slice(&[r, g, b]);
        }
        self.pointcloud_gpu[self.starter_kit.frame].write_bytes(0, bytemuck::cast_slice(&pts))?;

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

            core.device.cmd_draw(
                command_buffer,
                N_POINTS as u32,
                1,
                0,
                0,
            );
        }

        let (ret, cameras) = self.camera.get_matrices(platform)?;

        self.scene_ubo.upload(
            self.starter_kit.frame,
            &SceneData {
                cameras,
                anim: self.anim,
            },
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
