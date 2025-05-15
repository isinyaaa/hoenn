package main

import "base:runtime"
import sa "core:container/small_array"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg/glsl"
import "core:reflect"
import "core:slice"
import "core:strings"

import imgui "vendor/imgui"
import imdl "vendor/imgui/impl_sdl3"
import imvk "vendor/imgui/impl_vulkan"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"

when ODIN_OS == .Darwin {
	// NOTE: just a bogus import of the system library,
	// needed so we can add a linker flag to point to /usr/local/lib (where vulkan is installed by default)
	// when trying to load vulkan.
	@(require, extra_linker_flags = "-rpath /usr/local/lib")
	foreign import __ "system:System.framework"
	DEVICE_EXTENSIONS := []cstring {
		vk.KHR_SWAPCHAIN_EXTENSION_NAME,
		// vk.EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
		// vk.EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
		vk.KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
		vk.EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
		vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
	}
} else {
	DEVICE_EXTENSIONS := []cstring {
		// vk.EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
		// vk.EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
		vk.KHR_SWAPCHAIN_EXTENSION_NAME,
		vk.EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
		vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
	}
}

// Enables Vulkan debug logging and validation layers
ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)

GRAD_SHADER :: #load("shaders/gradient.spv")
SKY_SHADER :: #load("shaders/sky.spv")
COLORED_TRIANGLE_SHADER :: #load("shaders/colored_triangle.spv")
FLAT_SHADER :: #load("shaders/flat.spv")

INIT_RES :: [2]i32{1920, 1080}
PROG :: "vk"

N_QFAM :: 8
Q_CAPS := []vk.QueueFlag{.GRAPHICS, .COMPUTE, .TRANSFER}
N_Q :: 32

N_EXTS :: 300
N_FMTS :: 300

SURF_FMT :: vk.SurfaceFormatKHR {
	format     = .B8G8R8A8_UNORM,
	colorSpace = .SRGB_NONLINEAR,
}
PMODE: vk.PresentModeKHR = .FIFO
MAX_FRAMES_IN_FLIGHT :: 2

g_ctx: runtime.Context
// g_framebuffer_resized: bool

VKQ :: struct {
	idx: u32,
	q:   vk.Queue,
}

Re :: struct($N: int) where N >= 0 {
	pool:   vk.CommandPool,
	bufs:   [N]vk.CommandBuffer,
	fences: [N]vk.Fence,
}

Shader_Type :: enum {
	gradient,
	sky,
	colored_triangle,
	flat,
}

Pipe_Type :: enum {
	compute,
	gfx,
}

Compute_Effect :: enum {
	gradient,
	sky,
}

Gfx_Pipe :: enum {
	triangle,
}

Screen_Pipe :: union {
	Compute_Effect,
	Gfx_Pipe,
}

Pipeline :: struct {
	pipeline: vk.Pipeline,
	layout:   vk.PipelineLayout,
}

state := struct {
	// mui_ctx:         mui.Context,
	window:          ^sdl.Window,
	// sdl_ctx:         struct {
	//
	//    },
	// window:          glfw.WindowHandle,
	instance:        vk.Instance,
	physical_device: vk.PhysicalDevice,
	mprop:           vk.PhysicalDeviceMemoryProperties,
	device:          vk.Device,
	gfx:             VKQ,
	imm:             Re(1),
	pushc:           struct {
		tr:   glsl.vec4,
		cam:  glsl.vec4,
		pos:  glsl.vec4,
		post: glsl.vec4,
	},
	swapchain:       struct {
		surf:         vk.SurfaceKHR,
		chain:        vk.SwapchainKHR,
		caps:         vk.SurfaceCapabilitiesKHR,
		using frames: Re(MAX_FRAMES_IN_FLIGHT),
		count:        u32,
		semas:        []vk.Semaphore,
		finis:        []vk.Semaphore,
		images:       []vk.Image,
		views:        []vk.ImageView,
	},
	ext:             vk.Extent2D,
	drawimg:         Image,
	shaders:         [Shader_Type]vk.ShaderModule,
	modes:           [Pipe_Type]#soa[2]Pipeline,
	// cshadem:         vk.ShaderModule,
	// skyshadem:       vk.ShaderModule,
	cds:             vk.DescriptorSet,
}{}

Image :: struct {
	buf:  vk.Image,
	view: vk.ImageView,
	ext:  vk.Extent3D,
	fmt:  vk.Format,
	addr: vk.DeviceMemory,
	mem:  vk.MemoryRequirements,
}


main :: proc() {
	context.logger = log.create_console_logger()
	g_ctx = context

	if !sdl.Init({.VIDEO}) do log.panic("sdl dead")

	state.window = sdl.CreateWindow(PROG, INIT_RES[0], INIT_RES[1], {.RESIZABLE, .VULKAN})
	defer sdl.DestroyWindow(state.window)

	vk.load_proc_addresses_global(rawptr(sdl.Vulkan_GetVkGetInstanceProcAddr()))
	assert(vk.CreateInstance != nil, "vulkan function pointers not loaded")

	create_info := vk.InstanceCreateInfo {
		sType            = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &vk.ApplicationInfo {
			sType = .APPLICATION_INFO,
			pApplicationName = "Howdy",
			applicationVersion = vk.MAKE_VERSION(1, 0, 0),
			engineVersion = vk.MAKE_VERSION(1, 0, 0),
			pEngineName = "None",
			apiVersion = vk.API_VERSION_1_3,
		},
	}
	{
		count: u32
		exts := slice.from_ptr(sdl.Vulkan_GetInstanceExtensions(&count), int(count))
		extensions := slice.clone_to_dynamic(exts, context.temp_allocator)
		when ODIN_OS == .Darwin {
			create_info.flags |= {.ENUMERATE_PORTABILITY_KHR}
			append(&extensions, vk.KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
		}
		when ENABLE_VALIDATION_LAYERS {
			create_info.ppEnabledLayerNames = raw_data([]cstring{"VK_LAYER_KHRONOS_validation"})
			create_info.enabledLayerCount = 1

			append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
			// append(&extensions, vk.EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)

			// Severity based on logger level.
			severity: vk.DebugUtilsMessageSeverityFlagsEXT
			if context.logger.lowest_level <= .Error do severity |= {.ERROR}
			if context.logger.lowest_level <= .Warning do severity |= {.WARNING}
			if context.logger.lowest_level <= .Info do severity |= {.INFO}
			if context.logger.lowest_level <= .Debug do severity |= {.VERBOSE}

			dbg_create_info := vk.DebugUtilsMessengerCreateInfoEXT {
				sType           = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
				messageSeverity = severity,
				messageType     = {
					.GENERAL,
					.VALIDATION,
					.PERFORMANCE,
					// .DEVICE_ADDRESS_BINDING,
				}, // all of them.
				pfnUserCallback = vk_messenger_callback,
			}
			create_info.pNext = &dbg_create_info
		}

		create_info.enabledExtensionCount = u32(len(extensions))
		create_info.ppEnabledExtensionNames = raw_data(extensions)

		must(vk.CreateInstance(&create_info, nil, &state.instance))
	}
	defer vk.DestroyInstance(state.instance, nil)
	vk.load_proc_addresses_instance(state.instance)

	when ENABLE_VALIDATION_LAYERS {
		dbg_messenger: vk.DebugUtilsMessengerEXT
		must(
			vk.CreateDebugUtilsMessengerEXT(
				state.instance,
				transmute(^vk.DebugUtilsMessengerCreateInfoEXT)create_info.pNext,
				nil,
				&dbg_messenger,
			),
		)
		defer vk.DestroyDebugUtilsMessengerEXT(state.instance, dbg_messenger, nil)
	}

	must(pick_physical_device())
	vk.GetPhysicalDeviceMemoryProperties(state.physical_device, &state.mprop)

	// gather all queues that support gfx
	// TODO: gfx and present queues might be different in hybrid GPU systems, lol whatever man
	{
		q_infos: sa.Small_Array(N_QFAM, vk.DeviceQueueCreateInfo)
		count: u32 = N_QFAM
		qfams: sa.Small_Array(N_QFAM, vk.QueueFamilyProperties)
		vk.GetPhysicalDeviceQueueFamilyProperties(
			state.physical_device,
			&count,
			raw_data(qfams.data[:]),
		)
		qfams.len = int(count)
		for &q, i in sa.slice(&qfams) do if .GRAPHICS in q.queueFlags {
			prios := make([]f32, q.queueCount)
			for _, j in prios do prios[j] = 1.0
			sa.append(&q_infos, vk.DeviceQueueCreateInfo {
				sType            = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = u32(i),
				queueCount       = q.queueCount,
				// Scheduling priority between 0 and 1.
				pQueuePriorities = raw_data(prios),
			})
			// limit our gfx queues for now
			break
		}

		log.debugf("%v, %d", DEVICE_EXTENSIONS[:])
		must(
			vk.CreateDevice(
				state.physical_device,
				&vk.DeviceCreateInfo {
					sType = .DEVICE_CREATE_INFO,
					pNext = &vk.PhysicalDeviceSynchronization2Features {
						sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
						synchronization2 = true,
						pNext = &vk.PhysicalDeviceDynamicRenderingFeatures {
							sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
							dynamicRendering = true,
						},
					},
					pQueueCreateInfos = raw_data(sa.slice(&q_infos)),
					queueCreateInfoCount = u32(q_infos.len),
					enabledLayerCount = create_info.enabledLayerCount,
					ppEnabledLayerNames = create_info.ppEnabledLayerNames,
					ppEnabledExtensionNames = raw_data(DEVICE_EXTENSIONS[:]),
					enabledExtensionCount = u32(len(DEVICE_EXTENSIONS)),
				},
				nil,
				&state.device,
			),
		)
		vk.GetDeviceQueue(state.device, q_infos.data[0].queueFamilyIndex, 0, &state.gfx.q)
		// vk.GetDeviceQueue(state.device, indices.present.?, 0, &state.present_queue)
	}
	defer vk.DestroyDevice(state.device, nil)
	vk.load_proc_addresses_device(state.device)

	assert(
		sdl.Vulkan_CreateSurface(state.window, state.instance, nil, &state.swapchain.surf),
		"could not create surface",
	)
	defer vk.DestroySurfaceKHR(state.instance, state.swapchain.surf, nil)

	// NOTE: looks like a wrong binding with the third arg being a multipointer.
	must(
		vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(
			state.physical_device,
			state.swapchain.surf,
			&state.swapchain.caps,
		),
	)
	count := state.swapchain.caps.minImageCount + 1
	if state.swapchain.caps.maxImageCount > 0 {
		count = state.swapchain.caps.maxImageCount
	}
	assert(count > 0, "no swapchain")
	state.swapchain.count = count
	state.swapchain.images = make([]vk.Image, state.swapchain.count)
	state.swapchain.views = make([]vk.ImageView, state.swapchain.count)
	state.swapchain.semas = make([]vk.Semaphore, state.swapchain.count)
	state.swapchain.finis = make([]vk.Semaphore, state.swapchain.count)
	{
		// query supported formats
		// formats: sa.Small_Array(N_FMTS, vk.SurfaceFormatKHR)
		// count: u32 = N_FMTS
		// must(
		// 	vk.GetPhysicalDeviceSurfaceFormatsKHR(
		// 		state.physical_device,
		// 		state.swapchain.surf,
		// 		&count,
		// 		raw_data(formats.data[:]),
		// 	),
		// )
		// formats.len = int(count)

		// try to set mailbox refresh rate
		// pModes: [7]vk.PresentModeKHR
		// count: u32 = 7
		// must(
		// 	vk.GetPhysicalDeviceSurfacePresentModesKHR(
		// 		state.physical_device,
		// 		state.swapchain.surf,
		// 		&count,
		// 		raw_data(pModes[:]),
		// 	),
		// )
		// for &p in pModes[:count] do if p == .MAILBOX do state.pmode = .MAILBOX
	}

	// descriptors
	dpool: vk.DescriptorPool
	dsl: vk.DescriptorSetLayout
	{
		maxSets :: 10
		sizes := []vk.DescriptorPoolSize {
			// {.STORAGE_IMAGE, maxSets},
			{.SAMPLER, maxSets},
			{.COMBINED_IMAGE_SAMPLER, maxSets},
			{.SAMPLED_IMAGE, maxSets},
			{.STORAGE_IMAGE, maxSets},
			{.UNIFORM_TEXEL_BUFFER, maxSets},
			{.STORAGE_TEXEL_BUFFER, maxSets},
			{.UNIFORM_BUFFER, maxSets},
			{.STORAGE_BUFFER, maxSets},
			{.UNIFORM_BUFFER_DYNAMIC, maxSets},
			{.STORAGE_BUFFER_DYNAMIC, maxSets},
			{.INPUT_ATTACHMENT, maxSets},
		}
		must(
			vk.CreateDescriptorPool(
				state.device,
				&vk.DescriptorPoolCreateInfo {
					sType = .DESCRIPTOR_POOL_CREATE_INFO,
					flags = {.FREE_DESCRIPTOR_SET},
					maxSets = maxSets,
					poolSizeCount = u32(len(sizes)),
					pPoolSizes = raw_data(sizes),
				},
				nil,
				&dpool,
			),
		)
		must(
			vk.CreateDescriptorSetLayout(
				state.device,
				&vk.DescriptorSetLayoutCreateInfo {
					sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					bindingCount = 1,
					pBindings = &vk.DescriptorSetLayoutBinding {
						binding = 0,
						descriptorCount = 1,
						stageFlags = {.COMPUTE},
						descriptorType = .STORAGE_IMAGE,
					},
				},
				nil,
				&dsl,
			),
		)

		must(
			vk.AllocateDescriptorSets(
				state.device,
				&vk.DescriptorSetAllocateInfo {
					sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
					descriptorPool = dpool,
					descriptorSetCount = 1,
					pSetLayouts = &dsl,
				},
				&state.cds,
			),
		)
	}
	defer {
		vk.DestroyDescriptorPool(state.device, dpool, nil)
		vk.DestroyDescriptorSetLayout(state.device, dsl, nil)
	}

	create_swapchain()
	defer destroy_swapchain()

	// create cmd pool
	{
		pool_info := vk.CommandPoolCreateInfo {
			sType            = .COMMAND_POOL_CREATE_INFO,
			flags            = {.RESET_COMMAND_BUFFER},
			queueFamilyIndex = state.gfx.idx,
		}
		must(vk.CreateCommandPool(state.device, &pool_info, nil, &state.swapchain.pool))
		must(vk.CreateCommandPool(state.device, &pool_info, nil, &state.imm.pool))

		bufs: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer
		must(
			vk.AllocateCommandBuffers(
				state.device,
				&vk.CommandBufferAllocateInfo {
					sType = .COMMAND_BUFFER_ALLOCATE_INFO,
					commandPool = state.swapchain.pool,
					level = .PRIMARY,
					commandBufferCount = MAX_FRAMES_IN_FLIGHT,
				},
				raw_data(bufs[:]),
			),
		)
		for b, i in bufs {
			state.swapchain.bufs[i] = b
		}
		must(
			vk.AllocateCommandBuffers(
				state.device,
				&vk.CommandBufferAllocateInfo {
					sType = .COMMAND_BUFFER_ALLOCATE_INFO,
					commandPool = state.imm.pool,
					level = .PRIMARY,
					commandBufferCount = 1,
				},
				raw_data(bufs[:]),
			),
		)
		state.imm.bufs[0] = bufs[0]
	}
	defer {
		vk.DestroyCommandPool(state.device, state.swapchain.pool, nil)
		vk.DestroyCommandPool(state.device, state.imm.pool, nil)
	}
	// sdl.ClaimWindowForGPUDevice()

	// Set up sync primitives.
	{
		fence_info := vk.FenceCreateInfo {
			sType = .FENCE_CREATE_INFO,
			flags = {.SIGNALED},
		}
		sem_info := vk.SemaphoreCreateInfo {
			sType = .SEMAPHORE_CREATE_INFO,
		}
		for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
			must(vk.CreateFence(state.device, &fence_info, nil, &state.swapchain.fences[i]))
		}
		for i in 0 ..< state.swapchain.count {
			must(vk.CreateSemaphore(state.device, &sem_info, nil, &state.swapchain.semas[i]))
			must(vk.CreateSemaphore(state.device, &sem_info, nil, &state.swapchain.finis[i]))
		}
		must(vk.CreateFence(state.device, &fence_info, nil, &state.imm.fences[0]))

	}
	defer {
		for f in state.swapchain.fences {
			vk.DestroyFence(state.device, f, nil)
		}
		for i in 0 ..< state.swapchain.count {
			vk.DestroySemaphore(state.device, state.swapchain.semas[i], nil)
			vk.DestroySemaphore(state.device, state.swapchain.finis[i], nil)
		}
		vk.DestroyFence(state.device, state.imm.fences[0], nil)
	}
	// }

	state.shaders = [Shader_Type]vk.ShaderModule {
		.gradient         = create_shader_module(GRAD_SHADER),
		.sky              = create_shader_module(SKY_SHADER),
		.colored_triangle = create_shader_module(COLORED_TRIANGLE_SHADER),
		.flat             = create_shader_module(FLAT_SHADER),
	}
	defer for s in state.shaders do vk.DestroyShaderModule(state.device, s, nil)

	// compute pipeline
	must(
		vk.CreatePipelineLayout(
			state.device,
			&vk.PipelineLayoutCreateInfo {
				sType = .PIPELINE_LAYOUT_CREATE_INFO,
				setLayoutCount = 1,
				pSetLayouts = &dsl,
				pushConstantRangeCount = 1,
				pPushConstantRanges = &vk.PushConstantRange {
					size = size_of(state.pushc),
					stageFlags = {.COMPUTE},
				},
			},
			nil,
			&state.modes[.compute].layout[0],
		),
	)

	pipe_infos := []vk.ComputePipelineCreateInfo {
		{
			sType = .COMPUTE_PIPELINE_CREATE_INFO,
			stage = vk.PipelineShaderStageCreateInfo {
				sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
				stage = {.COMPUTE},
				module = state.shaders[.gradient],
				pName = "main",
			},
			layout = state.modes[.compute].layout[0],
		},
		{
			sType = .COMPUTE_PIPELINE_CREATE_INFO,
			stage = vk.PipelineShaderStageCreateInfo {
				sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
				stage = {.COMPUTE},
				module = state.shaders[.sky],
				pName = "main",
			},
			layout = state.modes[.compute].layout[0],
		},
	}
	// state.modes[.compute].layout[1] = state.modes[.compute].layout[0]
	must(
		vk.CreateComputePipelines(
			state.device,
			{},
			u32(len(pipe_infos)),
			raw_data(pipe_infos),
			nil,
			raw_data(state.modes[.compute].pipeline[:]),
		),
	)
	log.debugf("%v", state.modes[.compute].pipeline[:])
	color_attach_render := []vk.Format{.R16G16B16A16_SFLOAT}
	pipe_render := vk.PipelineRenderingCreateInfo {
		sType                   = .PIPELINE_RENDERING_CREATE_INFO_KHR,
		// viewMask:                u32,
		colorAttachmentCount    = u32(len(color_attach_render)),
		pColorAttachmentFormats = raw_data(color_attach_render),
		// depthAttachmentFormat:   Format,
		// stencilAttachmentFormat: Format,
	}
	{
		dynamic_states := []vk.DynamicState{.VIEWPORT, .SCISSOR}
		must(
			vk.CreatePipelineLayout(
				state.device,
				&vk.PipelineLayoutCreateInfo{sType = .PIPELINE_LAYOUT_CREATE_INFO},
				nil,
				&state.modes[.gfx].layout[0],
			),
		)
		stages := []vk.PipelineShaderStageCreateInfo {
			{
				sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
				stage = {.VERTEX},
				module = state.shaders[.colored_triangle],
				pName = "main",
			},
			{
				sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
				stage = {.FRAGMENT},
				module = state.shaders[.flat],
				pName = "main",
			},
		}

		pipeline := vk.GraphicsPipelineCreateInfo {
			sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
			stageCount          = u32(len(stages)),
			pStages             = raw_data(stages),
			pVertexInputState   = &vk.PipelineVertexInputStateCreateInfo {
				sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			},
			pInputAssemblyState = &vk.PipelineInputAssemblyStateCreateInfo {
				sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				topology = .TRIANGLE_LIST,
			},
			pViewportState      = &vk.PipelineViewportStateCreateInfo {
				sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				viewportCount = 1,
				scissorCount = 1,
			},
			pRasterizationState = &vk.PipelineRasterizationStateCreateInfo {
				sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				polygonMode = .FILL,
				lineWidth   = 1,
				// cullMode  = {.BACK},
				frontFace   = .CLOCKWISE,
			},
			pMultisampleState   = &vk.PipelineMultisampleStateCreateInfo {
				sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				rasterizationSamples = {._1},
				minSampleShading = 1,
			},
			pColorBlendState    = &vk.PipelineColorBlendStateCreateInfo {
				sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				logicOp = .COPY,
				attachmentCount = 1,
				pAttachments = &vk.PipelineColorBlendAttachmentState {
					colorWriteMask = {.R, .G, .B, .A},
				},
			},
			pDynamicState       = &vk.PipelineDynamicStateCreateInfo {
				sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				dynamicStateCount = u32(len(dynamic_states)),
				pDynamicStates = raw_data(dynamic_states),
			},
			layout              = state.modes[.gfx].layout[0],
			pNext               = &pipe_render,
		}
		must(
			vk.CreateGraphicsPipelines(
				state.device,
				0,
				1,
				&pipeline,
				nil,
				&state.modes[.gfx].pipeline[0],
			),
		)
	}
	defer for modes in state.modes do for p in modes {
		vk.DestroyPipelineLayout(state.device, p.layout, nil)
		vk.DestroyPipeline(state.device, p.pipeline, nil)
	}

	imgui.CreateContext()
	imvk.LoadFunctions(
		vk.API_VERSION_1_3,
		proc "c" (function_name: cstring, user_data: rawptr) -> vk.ProcVoidFunction {
			return vk.GetInstanceProcAddr((vk.Instance)(user_data), function_name)
		},
		state.instance,
	)
	imdl.InitForVulkan(state.window)
	color_attach := []vk.Format{SURF_FMT.format}
	imvk.Init(
		&imvk.InitInfo {
			ApiVersion = vk.API_VERSION_1_3,
			Instance = state.instance,
			PhysicalDevice = state.physical_device,
			Device = state.device,
			QueueFamily = state.gfx.idx,
			Queue = state.gfx.q,
			DescriptorPool = dpool, // See requirements in note above; ignored if using DescriptorPoolSize > 0
			MinImageCount = state.swapchain.count,
			ImageCount = state.swapchain.count,
			// MSAASamples: vk.SampleCountFlags, // 0 defaults to VK_SAMPLE_COUNT_1_BIT
			// PipelineCache: vk.PipelineCache,
			// Subpass: u32,
			UseDynamicRendering = true,
			PipelineRenderingCreateInfo = {
				sType                   = .PIPELINE_RENDERING_CREATE_INFO_KHR,
				// viewMask:                u32,
				colorAttachmentCount    = u32(len(color_attach)),
				pColorAttachmentFormats = raw_data(color_attach),
				// depthAttachmentFormat:   Format,
				// stencilAttachmentFormat: Format,
			},
			CheckVkResultFn = imguiCheckVkResult,
			MinAllocationSize = 1024 * 1024, //vk.DeviceSize,          // Minimum allocation size. Set to 1024*1024 to satisfy zealous best practices validation layer and waste a little memory.
		},
	)
	defer imvk.Shutdown()
	imvk.CreateFontsTexture()
	defer imvk.DestroyFontsTexture()

	defer vk.DeviceWaitIdle(state.device)

	frame_num: u64 = 0
	do_render: bool = true
	current: Compute_Effect
	compute_bound := i32(len(Compute_Effect))
	// max_effects := compute_bound + i32(len(Gfx_Pipe)) - 1
	// sel: Screen_Pipe
	for {
		e: sdl.Event
		resize: bool
		for sdl.PollEvent(&e) {
			free_all(context.temp_allocator)

			#partial switch e.type {
			case .QUIT:
				return
			case .WINDOW_MINIMIZED:
				do_render = false
			case .WINDOW_RESTORED:
				do_render = true
			case .WINDOW_RESIZED:
				resize = true
			}
			imdl.ProcessEvent(&e)
		}

		if resize {
			recreate_swapchain()
		} else if do_render {
			imvk.NewFrame()
			imdl.NewFrame()
			imgui.NewFrame()

			if imgui.Begin("background") {
				name, _ := reflect.enum_name_from_value(current)
				// if current < compute_bound {
				// 	e := Compute_Effect(current)
				// 	name, _ = reflect.enum_name_from_value(e)
				// 	sel = e
				// } else {
				// 	e := Gfx_Pipe(current - compute_bound)
				// 	sel = e
				// 	name, _ = 
				// }
				imgui.Text(strings.clone_to_cstring(fmt.tprintf("Effect: %v", name)))
				imgui.SliderInt("IDX", cast(^i32)&current, 0, compute_bound - 1)
				imgui.InputFloat4("transform", &state.pushc.tr)
				imgui.InputFloat4("camera", &state.pushc.cam)
				imgui.InputFloat4("position", &state.pushc.pos)
				imgui.InputFloat4("post-transform", &state.pushc.post)
			}
			imgui.End()
			// imgui.ShowDemoWindow()
			imgui.Render()
			r := draw(current, frame_num)
			switch {
			case r == .SUCCESS || r == .SUBOPTIMAL_KHR:
			case r == .ERROR_OUT_OF_DATE_KHR:
				recreate_swapchain()
			case:
				log.panicf("vulkan: present failure: %v", r)
			}

			frame_num += 1
		}
	}
}

create_shader_module :: proc(code: []byte) -> (module: vk.ShaderModule) {
	as_u32 := slice.reinterpret([]u32, code)

	create_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(code),
		pCode    = raw_data(as_u32),
	}
	must(vk.CreateShaderModule(state.device, &create_info, nil, &module))
	return
}

mtype :: proc(properties: vk.MemoryPropertyFlags, requiredTypeBits: u32) -> u32 {
	for mtype, i in state.mprop.memoryTypes[:state.mprop.memoryTypeCount] {
		if (1 << u32(i)) & requiredTypeBits == 0 do continue

		if (properties & mtype.propertyFlags) != properties do continue

		return u32(i)
	}

	return 0
}

create_image :: proc(info: ^vk.ImageCreateInfo, img: ^vk.Image) {
	must(vk.CreateImage(state.device, info, nil, img))

	memReq := vk.MemoryRequirements2 {
		sType = .MEMORY_REQUIREMENTS_2,
	}
	vk.GetImageMemoryRequirements2(
		state.device,
		&vk.ImageMemoryRequirementsInfo2{sType = .IMAGE_MEMORY_REQUIREMENTS_INFO_2, image = img^},
		&memReq,
	)
	addr: vk.DeviceMemory
	must(
		vk.AllocateMemory(
			state.device,
			&vk.MemoryAllocateInfo {
				sType = .MEMORY_ALLOCATE_INFO,
				allocationSize = memReq.memoryRequirements.size,
				memoryTypeIndex = mtype(
					{.DEVICE_LOCAL, .HOST_COHERENT, .HOST_VISIBLE},
					memReq.memoryRequirements.memoryTypeBits,
				),
			},
			nil,
			&addr,
		),
	)

	must(
		vk.BindImageMemory2(
			state.device,
			1,
			&vk.BindImageMemoryInfo{sType = .BIND_IMAGE_MEMORY_INFO, image = img^, memory = addr},
		),
	)
}

draw_background :: proc(buf: vk.CommandBuffer, img: vk.Image, effect: Compute_Effect, n: u64) {
	// cr := sub_range({.COLOR})
	// vk.CmdClearColorImage(
	// 	buf,
	// 	img,
	// 	.GENERAL,
	// 	&vk.ClearColorValue{float32 = },
	// 	1,
	// 	&cr,
	// )

	ef := &state.modes[.compute][effect]
	vk.CmdBindPipeline(buf, .COMPUTE, ef.pipeline)
	vk.CmdBindDescriptorSets(
		buf,
		.COMPUTE,
		state.modes[.compute].layout[0],
		0,
		1,
		&state.cds,
		0,
		nil,
	)
	state.pushc.tr = {0.0, 0.0, math.abs(math.sin(f32(n) / 120.0)), 1.0}
	state.pushc.cam = {0.0, math.abs(math.cos(f32(n) / 120.0)), 0.0, 1.0}
	vk.CmdPushConstants(
		buf,
		state.modes[.compute].layout[0],
		{.COMPUTE},
		0,
		size_of(state.pushc),
		&state.pushc,
	)
	vk.CmdDispatch(
		buf,
		u32(math.ceil(f32(state.drawimg.ext.width) / 16.0)),
		u32(math.ceil(f32(state.drawimg.ext.height) / 16.0)),
		1,
	)
}

draw_imgui :: proc(buf: vk.CommandBuffer, view: vk.ImageView) {
	vk.CmdBeginRendering(
		buf,
		&vk.RenderingInfo {
			sType = .RENDERING_INFO,
			// flags = vk.RenderingFlags,
			renderArea = {extent = state.ext},
			layerCount = 1,
			// viewMask:             u32,
			colorAttachmentCount = 1,
			pColorAttachments = &vk.RenderingAttachmentInfo {
				sType = .RENDERING_ATTACHMENT_INFO,
				imageView = view,
				imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
				loadOp = .LOAD,
				storeOp = .STORE,
			},
			// pDepthAttachment = ^vk.RenderingAttachmentInfo,
			// pStencilAttachment:   ^RenderingAttachmentInfo,
		},
	)
	imvk.RenderDrawData(imgui.GetDrawData(), buf)
	vk.CmdEndRendering(buf)
}

imm_exec :: proc() {
	must(vk.ResetFences(state.device, 1, &state.imm.fences[0]))

	cmd := state.imm.bufs[0]
	must(vk.ResetCommandBuffer(cmd, {}))
	must(
		vk.BeginCommandBuffer(
			cmd,
			&vk.CommandBufferBeginInfo {
				sType = .COMMAND_BUFFER_BEGIN_INFO,
				flags = {.ONE_TIME_SUBMIT},
			},
		),
	)
	must(vk.EndCommandBuffer(cmd))

	submit_info := vk.SubmitInfo2 {
		sType                  = .SUBMIT_INFO_2,
		commandBufferInfoCount = 1,
		pCommandBufferInfos    = &vk.CommandBufferSubmitInfo {
			sType = .COMMAND_BUFFER_SUBMIT_INFO,
			commandBuffer = cmd,
		},
	}
	must(vk.QueueSubmit2(state.gfx.q, 1, &submit_info, state.imm.fences[0]))
	must(vk.WaitForFences(state.device, 1, &state.imm.fences[0], true, max(u64)))
}

draw_geometry :: proc(buf: vk.CommandBuffer) {
	mode := state.modes[.gfx][0]
	vk.CmdBeginRendering(
		buf,
		&vk.RenderingInfo {
			sType = .RENDERING_INFO,
			renderArea = {extent = state.ext},
			layerCount = 1,
			// viewMask:             u32,
			colorAttachmentCount = 1,
			pColorAttachments = &vk.RenderingAttachmentInfo {
				sType = .RENDERING_ATTACHMENT_INFO,
				imageView = state.drawimg.view,
				imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
				loadOp = .LOAD,
				storeOp = .STORE,
			},
		},
	)
	vk.CmdBindPipeline(buf, .GRAPHICS, mode.pipeline)

	vk.CmdSetViewport(
		buf,
		0,
		1,
		&vk.Viewport{width = f32(state.ext.width), height = f32(state.ext.height), maxDepth = 1},
	)
	vk.CmdSetScissor(buf, 0, 1, &vk.Rect2D{extent = state.ext})
	vk.CmdDraw(buf, 3, 1, 0, 0)
	vk.CmdEndRendering(buf)
}

draw :: proc(effect: Compute_Effect, n: u64) -> vk.Result {
	fid := n % MAX_FRAMES_IN_FLIGHT
	must(vk.WaitForFences(state.device, 1, &state.swapchain.fences[fid], true, max(u64)))
	// must(vk.WaitForFences(state.device, 1, &f.fence, true, max(u64)))

	// Acquire an image from the swapchain.
	image_index: u32
	if r := vk.AcquireNextImageKHR(
		state.device,
		state.swapchain.chain,
		max(u64),
		state.swapchain.semas[fid],
		0,
		&image_index,
	); r != .SUCCESS && r != .SUBOPTIMAL_KHR {
		return r
	}
	must(vk.ResetFences(state.device, 1, &state.swapchain.fences[fid]))
	cmd := state.swapchain.bufs[fid]

	must(vk.ResetCommandBuffer(cmd, {}))
	must(
		vk.BeginCommandBuffer(
			cmd,
			&vk.CommandBufferBeginInfo {
				sType = .COMMAND_BUFFER_BEGIN_INFO,
				flags = {.ONE_TIME_SUBMIT},
			},
		),
	)

	img := state.swapchain.images[image_index]
	record(cmd, state.drawimg.buf, .UNDEFINED, .GENERAL)
	draw_background(cmd, img, effect, n)
	// switch pipe in effect {
	// case Compute_Effect:
	// case Gfx_Pipe:
	// // draw_background(cmd, img, pipe, n)
	// }

	record(cmd, state.drawimg.buf, .GENERAL, .COLOR_ATTACHMENT_OPTIMAL)
	draw_geometry(cmd)
	record(cmd, state.drawimg.buf, .COLOR_ATTACHMENT_OPTIMAL, .GENERAL)
	// record(cmd, state.drawimg.buf, .GENERAL, .TRANSFER_SRC_OPTIMAL)
	record(cmd, img, .UNDEFINED, .TRANSFER_DST_OPTIMAL)
	cpimg(
		cmd,
		state.drawimg.buf,
		img,
		vk.Extent2D{width = state.drawimg.ext.width, height = state.drawimg.ext.height},
		state.ext,
	)

	record(cmd, img, .TRANSFER_DST_OPTIMAL, .COLOR_ATTACHMENT_OPTIMAL)
	draw_imgui(cmd, state.swapchain.views[image_index])

	record(cmd, img, .COLOR_ATTACHMENT_OPTIMAL, .PRESENT_SRC_KHR)

	must(vk.EndCommandBuffer(cmd))

	ws := sub_info(state.swapchain.semas[fid], {.COLOR_ATTACHMENT_OUTPUT_KHR})
	ps := sub_info(state.swapchain.finis[fid], {.ALL_GRAPHICS_KHR})
	submit_info := vk.SubmitInfo2 {
		sType                    = .SUBMIT_INFO_2,
		commandBufferInfoCount   = 1,
		pCommandBufferInfos      = &vk.CommandBufferSubmitInfo {
			sType = .COMMAND_BUFFER_SUBMIT_INFO,
			commandBuffer = cmd,
		},
		waitSemaphoreInfoCount   = 1,
		pWaitSemaphoreInfos      = &ws,
		signalSemaphoreInfoCount = 1,
		pSignalSemaphoreInfos    = &ps,
	}
	must(vk.QueueSubmit2(state.gfx.q, 1, &submit_info, state.swapchain.fences[fid]))

	// Present.
	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		// pNext              = &vk.SwapchainPresentFenceInfoEXT {
		// 	sType = .SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
		// 	swapchainCount = 1,
		// 	pFences = &state.swapchain.frames[(image_index + 1) % MAX_FRAMES_IN_FLIGHT].fence,
		// },
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &state.swapchain.finis[fid],
		swapchainCount     = 1,
		pSwapchains        = &state.swapchain.chain,
		pImageIndices      = &image_index,
	}
	return vk.QueuePresentKHR(state.gfx.q, &present_info)
}

cpimg :: proc(buf: vk.CommandBuffer, src, dest: vk.Image, sext, dext: vk.Extent2D) {
	vk.CmdBlitImage2(
		buf,
		&vk.BlitImageInfo2 {
			sType = .BLIT_IMAGE_INFO_2,
			srcImage = src,
			srcImageLayout = .GENERAL,
			dstImage = dest,
			dstImageLayout = .TRANSFER_DST_OPTIMAL,
			regionCount = 1,
			pRegions = &vk.ImageBlit2 {
				sType = .IMAGE_BLIT_2,
				srcSubresource = {aspectMask = {.COLOR}, layerCount = 1},
				srcOffsets = {1 = {x = i32(sext.width), y = i32(sext.height), z = 1}},
				dstSubresource = {aspectMask = {.COLOR}, layerCount = 1},
				dstOffsets = {1 = {x = i32(dext.width), y = i32(dext.height), z = 1}},
			},
			filter = .LINEAR,
		},
	)
}

sub_info :: proc(sem: vk.Semaphore, stage: vk.PipelineStageFlags2) -> vk.SemaphoreSubmitInfo {
	return {sType = .SEMAPHORE_SUBMIT_INFO, semaphore = sem, stageMask = stage}
}

record :: proc(buf: vk.CommandBuffer, img: vk.Image, ol, nl: vk.ImageLayout) {
	vk.CmdPipelineBarrier2(
		buf,
		&vk.DependencyInfo {
			sType = .DEPENDENCY_INFO,
			imageMemoryBarrierCount = 1,
			pImageMemoryBarriers = &vk.ImageMemoryBarrier2 {
				sType = .IMAGE_MEMORY_BARRIER_2,
				srcStageMask = {.ALL_COMMANDS},
				srcAccessMask = {.MEMORY_WRITE},
				dstStageMask = {.ALL_COMMANDS},
				dstAccessMask = {.MEMORY_WRITE, .MEMORY_READ},
				oldLayout = ol,
				newLayout = nl,
				image = img,
				subresourceRange = sub_range(
					{.DEPTH if nl == .DEPTH_ATTACHMENT_OPTIMAL else .COLOR},
				),
			},
		},
	)
}

sub_range :: proc(aspect: vk.ImageAspectFlags) -> vk.ImageSubresourceRange {
	return {
		aspectMask = aspect,
		levelCount = 1,
		layerCount = 1,
		// baseMipLevel = 0,
		// baseArrayLayer = 0,
	}
}

recreate_swapchain :: proc() {
	// Don't do anything when minimized.
	// for w, h := glfw.GetFramebufferSize(state.window);
	//     w == 0 || h == 0;
	//     w, h = glfw.GetFramebufferSize(state.window) {
	// 	glfw.WaitEvents()
	//
	// 	// Handle closing while minimized.
	// 	if glfw.WindowShouldClose(state.window) {break}
	// }

	vk.DeviceWaitIdle(state.device)
	destroy_swapchain()
	create_swapchain()
}

create_swapchain :: proc() {
	// Setup swapchain.
	// if state.gfxq != state.pq {
	// 	create_info.imageSharingMode = .CONCURRENT
	// 	create_info.queueFamilyIndexCount = 2
	// 	create_info.pQueueFamilyIndices = raw_data(
	// 		[]u32{state.gfxq, state.pq},
	// 	)
	// }

	state.ext = choose_swapchain_extent()
	// pModes := []vk.PresentModeKHR{.FIFO}
	must(
		vk.CreateSwapchainKHR(
			state.device,
			&vk.SwapchainCreateInfoKHR {
				sType            = .SWAPCHAIN_CREATE_INFO_KHR,
				surface          = state.swapchain.surf,
				minImageCount    = state.swapchain.count,
				imageFormat      = SURF_FMT.format,
				imageColorSpace  = SURF_FMT.colorSpace,
				imageExtent      = state.ext,
				imageArrayLayers = 1,
				imageUsage       = {.COLOR_ATTACHMENT, .TRANSFER_DST},
				preTransform     = state.swapchain.caps.currentTransform,
				compositeAlpha   = {.OPAQUE},
				presentMode      = PMODE,
				clipped          = true,
				// pNext = &vk.SwapchainPresentModesCreateInfoEXT {
				// 	sType = .SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT,
				// 	presentModeCount = u32(len(pModes)),
				// 	pPresentModes = raw_data(pModes),
				// },
			},
			nil,
			&state.swapchain.chain,
		),
	)

	// Setup swapchain images.
	count: u32 = state.swapchain.count
	must(
		vk.GetSwapchainImagesKHR(
			state.device,
			state.swapchain.chain,
			&count,
			raw_data(state.swapchain.images[:]),
		),
	)

	for image, i in state.swapchain.images {
		create_info := vk.ImageViewCreateInfo {
			sType            = .IMAGE_VIEW_CREATE_INFO,
			image            = image,
			viewType         = .D2,
			format           = SURF_FMT.format,
			subresourceRange = sub_range({.COLOR}),
		}
		must(vk.CreateImageView(state.device, &create_info, nil, &state.swapchain.views[i]))
	}

	state.drawimg.ext = {
		width  = state.ext.width,
		height = state.ext.height,
		depth  = 1,
	}
	state.drawimg.fmt = .R16G16B16A16_SFLOAT

	create_image(
		&vk.ImageCreateInfo {
			sType       = .IMAGE_CREATE_INFO,
			flags       = {},
			imageType   = .D2,
			format      = state.drawimg.fmt,
			extent      = state.drawimg.ext,
			mipLevels   = 1,
			arrayLayers = 1,
			samples     = {._1},
			tiling      = .LINEAR,
			usage       = {.TRANSFER_SRC, .TRANSFER_DST, .STORAGE, .COLOR_ATTACHMENT},
			// sharingMode:           SharingMode,
			// queueFamilyIndexCount: u32,
			// pQueueFamilyIndices:   [^]u32,
			// initialLayout = .,
		},
		&state.drawimg.buf,
	)
	must(
		vk.CreateImageView(
			state.device,
			&vk.ImageViewCreateInfo {
				sType = .IMAGE_VIEW_CREATE_INFO,
				image = state.drawimg.buf,
				viewType = .D2,
				format = state.drawimg.fmt,
				subresourceRange = sub_range({.COLOR}),
			},
			nil,
			&state.drawimg.view,
		),
	)

	vk.UpdateDescriptorSets(
		state.device,
		1,
		&vk.WriteDescriptorSet {
			sType = .WRITE_DESCRIPTOR_SET,
			dstSet = state.cds,
			descriptorCount = 1,
			descriptorType = .STORAGE_IMAGE,
			pImageInfo = &vk.DescriptorImageInfo {
				imageView = state.drawimg.view,
				imageLayout = .GENERAL,
			},
		},
		0,
		nil,
	)
}

choose_swapchain_extent :: proc() -> vk.Extent2D {
	// if state.swapchain.caps.currentExtent.width != max(u32) do return state.swapchain.caps.currentExtent

	w, h: i32
	sdl.GetWindowSizeInPixels(state.window, &w, &h)
	return vk.Extent2D {
		width = clamp(
			u32(w),
			state.swapchain.caps.minImageExtent.width,
			state.swapchain.caps.maxImageExtent.width,
		),
		height = clamp(
			u32(h),
			state.swapchain.caps.minImageExtent.height,
			state.swapchain.caps.maxImageExtent.height,
		),
	}
}

destroy_swapchain :: proc() {
	for view in state.swapchain.views {
		vk.DestroyImageView(state.device, view, nil)
	}
	vk.DestroySwapchainKHR(state.device, state.swapchain.chain, nil)
	vk.DestroyImage(state.device, state.drawimg.buf, nil)
	vk.FreeMemory(state.device, state.drawimg.addr, nil)
	vk.DestroyImageView(state.device, state.drawimg.view, nil)

	// cleanup
	// // All steps succeeded.
	// (*pAllocation)->InitImageUsage(*pImageCreateInfo);
	// if(pAllocationInfo != VMA_NULL)
	// {
	//     allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
	// }
	// allocator->FreeMemory(
	//     1, // allocationCount
	//     pAllocation);
	// *pAllocation = VK_NULL_HANDLE;
	// (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
}

N_PHYS :: 3

@(require_results)
pick_physical_device :: proc() -> vk.Result {
	count: u32
	phys: sa.Small_Array(N_PHYS, vk.PhysicalDevice)
	count = N_PHYS
	vk.EnumeratePhysicalDevices(state.instance, &count, raw_data(phys.data[:])) or_return
	assert(count != 0, "no GPU found")
	phys.len = int(count)

	props: vk.PhysicalDeviceProperties
	features: vk.PhysicalDeviceFeatures
	exts: sa.Small_Array(N_EXTS, vk.ExtensionProperties)

	qfams: sa.Small_Array(N_QFAM, vk.QueueFamilyProperties)

	best_device_score: u64 = 0
	devloop: for device in sa.slice(&phys) {
		vk.GetPhysicalDeviceProperties(device, &props)

		name := byte_arr_str(&props.deviceName)
		score: u64
		switch props.deviceType {
		case .DISCRETE_GPU:
			score += 300_000
		case .INTEGRATED_GPU:
			score += 200_000
		case .VIRTUAL_GPU:
			score += 100_000
		case .CPU, .OTHER:
			log.infof("vulkan: discarding %q %v device type", name, props.deviceType)
			continue devloop
		}
		// vk.GetPhysicalDeviceFeatures(device, &features)
		// if !features.geometryShader {
		// 	log.info("vulkan: device does not support geometry shaders")
		// 	continue
		// }
		log.infof("vulkan: evaluating device %q: %v", name, props)

		// check extension support
		defer sa.clear(&exts)
		count = N_EXTS
		vk.EnumerateDeviceExtensionProperties(
			device,
			nil,
			&count,
			raw_data(exts.data[:]),
		) or_continue
		exts.len = int(count)

		rextloop: for rext in DEVICE_EXTENSIONS {
			for &extension in sa.slice(&exts) do if byte_arr_str(&extension.extensionName) == string(rext) do continue rextloop
			log.infof("vulkan: device does not support required extension %q", rext)
			continue devloop
		}

		// check queue support
		sa.clear(&qfams)
		count = N_QFAM
		vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, raw_data(qfams.data[:]))
		qfams.len = int(count)
		qloop: for rq in Q_CAPS {
			for &q in sa.slice(&qfams) do if rq in q.queueFlags do continue qloop
			log.infof("vulkan: device does not support required queue %q", rq)
			continue devloop
		}

		// Maximum texture size.
		score += u64(props.limits.maxImageDimension2D)
		log.infof(
			"vulkan: added the max 2D image dimensions (texture size) of %v to the score",
			props.limits.maxImageDimension2D,
		)
		if score > best_device_score {
			state.physical_device = device
			best_device_score = score
		}
		log.infof("vulkan: device %q scored %v", name, score)
	}

	if best_device_score == 0 do log.panic("vulkan: no suitable GPU found")
	return .SUCCESS
}

vk_messenger_callback :: proc "system" (
	messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
	messageTypes: vk.DebugUtilsMessageTypeFlagsEXT,
	pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
	pUserData: rawptr,
) -> b32 {
	context = g_ctx

	level: log.Level
	switch {
	case .ERROR in messageSeverity:
		level = .Error
	case .WARNING in messageSeverity:
		level = .Warning
	case .INFO in messageSeverity:
		level = .Info
	case:
		level = .Debug
	}

	log.logf(level, "vulkan[%v]: %s", messageTypes, pCallbackData.pMessage)
	return false
}

imguiCheckVkResult :: proc "c" (err: vk.Result) {
	context = g_ctx
	if int(err) == 0 {return}
	if int(err) < 0 {
		log.fatalf("imvk: %i", err)
	}
	log.errorf("imvk: %i", err)
}

byte_arr_str :: proc(arr: ^[$N]byte) -> string {
	return strings.truncate_to_byte(string(arr[:]), 0)
}

must :: proc(result: vk.Result, loc := #caller_location) {
	if result != .SUCCESS {
		log.panicf("vulkan failure %v", result, location = loc)
	}
}
