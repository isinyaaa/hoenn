package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:reflect"
import "core:slice"
import "core:strings"

import imgui "vendor/imgui"
import imdl "vendor/imgui/impl_sdl3"
import imvk "vendor/imgui/impl_vulkan"
import gltf "vendor:cgltf"
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
		vk.KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
		// maybe use these if vulkan < 1.3
		// vk.EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		// vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
		// vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
		// vk.EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
		// vk.EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME,
	}
} else {
	DEVICE_EXTENSIONS := []cstring {
		// vk.EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
		// vk.EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
		vk.KHR_SWAPCHAIN_EXTENSION_NAME,
		// vk.EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		// vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
		// vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
		// vk.EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
		// vk.EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME,
	}
}

BLEND :: 1

// Enables Vulkan debug logging and validation layers
ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)

GRAD_SHADER :: #load("shaders/gradient.spv")
SKY_SHADER :: #load("shaders/sky.spv")
COLORED_TRIANGLE_SHADER :: #load("shaders/colored_triangle.spv")
FLAT_SHADER :: #load("shaders/flat.spv")
GLB_PATH :: "assets/basicmesh.glb"

INIT_RES :: [2]i32{1920, 1080}
// max supported resolution
DRAW_RES :: [2]i32{3840, 2160}
PROG :: "vk"

Smaller_Array :: struct($N: int, $T: typeid) where N >= 0 {
	data: [N]T,
	len:  u32,
}

// queue family limit
N_QFAM :: 8
Q_CAPS := []vk.QueueFlag{.GRAPHICS, .COMPUTE, .TRANSFER}
// maximum supported queues
N_Q :: 32

// maximum device extensions to query
N_EXTS :: 300
// maximum surface formats to query
N_FMTS :: 300

// default surface format, should be widely supported
SURF_FMT :: vk.SurfaceFormatKHR {
	format     = .B8G8R8A8_UNORM,
	colorSpace = .SRGB_NONLINEAR,
}
// default present mode
PMODE: vk.PresentModeKHR = .FIFO
// max frames to handle in app, most devices want at least 3 swapchain images
MAX_FRAMES_IN_FLIGHT :: 3

g_ctx: runtime.Context

XFrame :: struct {
	cmd:   vk.CommandBuffer,
	fence: vk.Fence,
}

Sync :: struct {
	swap, render: vk.Semaphore,
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

MeshBuffer :: struct {
	indices:  vk.Buffer,
	vertices: vk.Buffer,
	vtx_addr: vk.DeviceAddress,
	// kept for cleanup purposes
	idx_mem:  vk.DeviceMemory,
	vtx_mem:  vk.DeviceMemory,
}

GeoSurface :: struct {
	start: u32,
	count: u32,
}

MeshAsset :: struct {
	name:  string,
	surfs: []GeoSurface,
	buf:   MeshBuffer,
}

vec2 :: [2]f32
vec3 :: [3]f32
vec4 :: [4]f32

mat4 :: matrix[4, 4]f32

// alignment to match glsl uniform
Vertex :: struct #min_field_align (16) {
	pos:    vec3,
	color:  vec4,
	normal: vec3,
	uv:     vec2,
}

Geometry :: enum {
	triangle,
	cube,
	sphere,
	monkey,
}

Image :: struct {
	img:  vk.Image,
	view: vk.ImageView,
	fmt:  vk.Format,
	mem:  vk.DeviceMemory,
}

Descriptor :: struct {
	set:    vk.DescriptorSet,
	layout: vk.DescriptorSetLayout,
}

Bound_Resource :: enum {
	draw,
	image,
	scene,
}

Queue :: struct {
	q:   vk.Queue,
	idx: u32,
}

GPU :: struct {
	api:    vk.Instance,
	phy:    struct {
		dev: vk.PhysicalDevice,
		mem: []vk.MemoryPropertyFlags,
	},
	dev:    vk.Device,
	submit: Queue,
}

Window :: struct {
	handle: ^sdl.Window,
	surf:   vk.SurfaceKHR,
	caps:   vk.SurfaceCapabilitiesKHR,
}

Swapchain :: struct {
	chain:       vk.SwapchainKHR,
	ext:         vk.Extent2D,
	draw, depth: Image,
	frames:      [MAX_FRAMES_IN_FLIGHT]struct {
		using base: XFrame,
		sync:       Sync,
	},
	images:      [MAX_FRAMES_IN_FLIGHT]vk.Image,
	views:       [MAX_FRAMES_IN_FLIGHT]vk.ImageView,
}

Memory :: struct {
	pool:       vk.DescriptorPool,
	descriptor: vk.DescriptorSet,
}

state := struct {
	using gpu:    GPU,
	using window: Window,
	using mem:    Memory,
	swap:         Swapchain,
	geometry:     [Geometry]MeshAsset,
	shaders:      [Shader_Type]vk.ShaderModule,
	// immediate mode
	imm:          struct {
		pool:  vk.CommandPool,
		frame: XFrame,
	},
	// dynamic render mode
	render:       struct {
		ext:     vk.Extent2D,
		pool:    vk.CommandPool,
		compute: struct {
			layout:  vk.PipelineLayout,
			effects: [2]vk.Pipeline,
			pushc:   struct #min_field_align (8) {
				tr, cam, pos, post: vec4,
			},
		},
		gfx:     struct {
			layout: vk.PipelineLayout,
			pipe:   vk.Pipeline,
			pushc:  struct #min_field_align (8) {
				tr:        mat4,
				vert_addr: vk.DeviceAddress,
			},
		},
	},
}{}

main :: proc() {
	context.logger = log.create_console_logger()
	g_ctx = context

	if !sdl.Init({.VIDEO}) do log.panic("sdl dead")

	state.handle = sdl.CreateWindow(PROG, INIT_RES[0], INIT_RES[1], {.RESIZABLE, .VULKAN})
	defer sdl.DestroyWindow(state.handle)

	vk.load_proc_addresses_global(rawptr(sdl.Vulkan_GetVkGetInstanceProcAddr()))
	assert(vk.CreateInstance != nil, "vulkan function pointers not loaded")

	create_info := vk.InstanceCreateInfo {
		sType            = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &vk.ApplicationInfo {
			sType = .APPLICATION_INFO,
			pApplicationName = PROG,
			applicationVersion = vk.MAKE_VERSION(1, 0, 0),
			engineVersion = vk.MAKE_VERSION(1, 0, 0),
			pEngineName = "None",
			apiVersion = vk.API_VERSION_1_3,
		},
	}
	{
		count: u32
		extensions := slice.clone_to_dynamic(
			slice.from_ptr(sdl.Vulkan_GetInstanceExtensions(&count), int(count)),
			context.temp_allocator,
		)
		for ext in extensions do if ext == vk.KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME do create_info.flags |= {.ENUMERATE_PORTABILITY_KHR}
		when ENABLE_VALIDATION_LAYERS {
			create_info.ppEnabledLayerNames = raw_data([]cstring{"VK_LAYER_KHRONOS_validation"})
			create_info.enabledLayerCount = 1

			append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)

			// Severity based on logger level.
			severity: vk.DebugUtilsMessageSeverityFlagsEXT
			if context.logger.lowest_level <= .Error do severity |= {.ERROR}
			if context.logger.lowest_level <= .Warning do severity |= {.WARNING}
			if context.logger.lowest_level <= .Info do severity |= {.INFO}
			if context.logger.lowest_level <= .Debug do severity |= {.VERBOSE}

			feats := []vk.ValidationFeatureEnableEXT{.GPU_ASSISTED, .BEST_PRACTICES}
			create_info.pNext = &vk.DebugUtilsMessengerCreateInfoEXT {
				sType           = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
				messageSeverity = severity,
				messageType     = {
					.GENERAL,
					.VALIDATION,
					.PERFORMANCE,
					// NOTE: unsupported on moltenvk
					//.DEVICE_ADDRESS_BINDING
				},
				pfnUserCallback = vk_messenger_callback,
				pNext           = &vk.ValidationFeaturesEXT {
					sType = .VALIDATION_FEATURES_EXT,
					enabledValidationFeatureCount = u32(len(feats)),
					pEnabledValidationFeatures = raw_data(feats),
				},
			}
		}

		create_info.enabledExtensionCount = u32(len(extensions))
		create_info.ppEnabledExtensionNames = raw_data(extensions)

		must(vk.CreateInstance(&create_info, nil, &state.api))
	}
	defer vk.DestroyInstance(state.api, nil)
	vk.load_proc_addresses_instance(state.api)

	when ENABLE_VALIDATION_LAYERS {
		dbg_messenger: vk.DebugUtilsMessengerEXT
		must(
			vk.CreateDebugUtilsMessengerEXT(
				state.api,
				transmute(^vk.DebugUtilsMessengerCreateInfoEXT)create_info.pNext,
				nil,
				&dbg_messenger,
			),
		)
		defer vk.DestroyDebugUtilsMessengerEXT(state.api, dbg_messenger, nil)
	}

	must(pick_physical_device())
	mem_prop: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(state.phy.dev, &mem_prop)
	state.phy.mem = make([]vk.MemoryPropertyFlags, int(mem_prop.memoryTypeCount))
	for mtype, i in mem_prop.memoryTypes[:mem_prop.memoryTypeCount] do state.phy.mem[i] = mtype.propertyFlags

	// gather all queues that support gfx
	// TODO: gfx and present queues might be different in hybrid GPU systems, lol whatever man
	{
		q_infos: [1]vk.DeviceQueueCreateInfo
		qfams: Smaller_Array(N_QFAM, vk.QueueFamilyProperties)
		qfams.len = N_QFAM
		vk.GetPhysicalDeviceQueueFamilyProperties(
			state.phy.dev,
			&qfams.len,
			raw_data(qfams.data[:]),
		)
		for &q, i in qfams.data[:qfams.len] do if .GRAPHICS in q.queueFlags {
			prios := make([]f32, q.queueCount)
			for _, j in prios do prios[j] = 1.0
			q_infos[0] = vk.DeviceQueueCreateInfo {
				sType            = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = u32(i),
				queueCount       = q.queueCount,
				// Scheduling priority between 0 and 1.
				pQueuePriorities = raw_data(prios),
			}
			// limit our gfx queues for now
			break
		}

		must(
			vk.CreateDevice(
				state.phy.dev,
				&vk.DeviceCreateInfo {
					sType = .DEVICE_CREATE_INFO,
					pQueueCreateInfos = raw_data(q_infos[:]),
					queueCreateInfoCount = u32(len(q_infos)),
					enabledLayerCount = create_info.enabledLayerCount,
					ppEnabledLayerNames = create_info.ppEnabledLayerNames,
					ppEnabledExtensionNames = raw_data(DEVICE_EXTENSIONS[:]),
					enabledExtensionCount = u32(len(DEVICE_EXTENSIONS)),
					pNext = &vk.PhysicalDeviceSynchronization2Features {
						sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
						synchronization2 = true,
						pNext = &vk.PhysicalDeviceDynamicRenderingFeatures {
							sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
							dynamicRendering = true,
							pNext = &vk.PhysicalDeviceVulkan12Features {
								sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
								bufferDeviceAddress = true,
								descriptorIndexing = true,
							},
						},
					},
				},
				nil,
				&state.dev,
			),
		)
		vk.GetDeviceQueue(state.dev, q_infos[0].queueFamilyIndex, 0, &state.submit.q)
	}
	defer vk.DestroyDevice(state.dev, nil)
	vk.load_proc_addresses_device(state.dev)

	assert(
		sdl.Vulkan_CreateSurface(state.handle, state.api, nil, &state.surf),
		"could not create surface",
	)
	defer vk.DestroySurfaceKHR(state.api, state.surf, nil)

	// NOTE: looks like a wrong binding with the third arg being a multipointer.
	must(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(state.phy.dev, state.surf, &state.caps))
	assert(state.caps.maxImageCount >= 3 && state.caps.minImageCount <= 3, "no swapchain")

	dslayout: vk.DescriptorSetLayout
	{
		maxSets :: 10
		state.pool = create_pool(
		10,
		{
			{.STORAGE_IMAGE, 1},
			{.UNIFORM_BUFFER, 1},
			// {.STORAGE_BUFFER, 3},
			// {.COMBINED_IMAGE_SAMPLER, 4},
		},
		)
		dslayout = descriptor_layout({.COMPUTE}, {.STORAGE_IMAGE})
		// image := descriptor_layout({.FRAGMENT}, {.COMBINED_IMAGE_SAMPLER})
		// scene := descriptor_layout({.FRAGMENT, .VERTEX}, {.STORAGE_IMAGE})
		state.descriptor = descriptor_set(state.pool, dslayout)
		// layout = draw,
		// }
		// .image = {set = descriptor_set()},
		// }
	}
	defer {
		vk.DestroyDescriptorPool(state.dev, state.pool, nil)
		vk.DestroyDescriptorSetLayout(state.dev, dslayout, nil)
	}

	create_swapchain()
	create_drawsurf(vk.Extent2D{width = u32(DRAW_RES[0]), height = u32(DRAW_RES[1])})
	defer {
		destroy_swapchain()
		destroy_image(state.swap.draw)
		destroy_image(state.swap.depth)
	}

	// create cmd pool
	{
		pool_info := vk.CommandPoolCreateInfo {
			sType            = .COMMAND_POOL_CREATE_INFO,
			flags            = {.RESET_COMMAND_BUFFER},
			queueFamilyIndex = state.submit.idx,
		}
		must(vk.CreateCommandPool(state.dev, &pool_info, nil, &state.render.pool))
		must(vk.CreateCommandPool(state.dev, &pool_info, nil, &state.imm.pool))

		bufs: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer
		must(
			vk.AllocateCommandBuffers(
				state.dev,
				&vk.CommandBufferAllocateInfo {
					sType = .COMMAND_BUFFER_ALLOCATE_INFO,
					commandPool = state.render.pool,
					level = .PRIMARY,
					commandBufferCount = MAX_FRAMES_IN_FLIGHT,
				},
				raw_data(bufs[:]),
			),
		)
		for b, i in bufs {
			state.swap.frames[i].cmd = b
		}
		must(
			vk.AllocateCommandBuffers(
				state.dev,
				&vk.CommandBufferAllocateInfo {
					sType = .COMMAND_BUFFER_ALLOCATE_INFO,
					commandPool = state.imm.pool,
					level = .PRIMARY,
					commandBufferCount = 1,
				},
				&state.imm.frame.cmd,
			),
		)
	}
	defer {
		vk.DestroyCommandPool(state.dev, state.render.pool, nil)
		vk.DestroyCommandPool(state.dev, state.imm.pool, nil)
	}

	// Set up sync primitives.
	{
		fence_info := vk.FenceCreateInfo {
			sType = .FENCE_CREATE_INFO,
			flags = {.SIGNALED},
		}
		sem_info := vk.SemaphoreCreateInfo {
			sType = .SEMAPHORE_CREATE_INFO,
		}
		for &f in state.swap.frames {
			must(vk.CreateFence(state.dev, &fence_info, nil, &f.fence))
			must(vk.CreateSemaphore(state.dev, &sem_info, nil, &f.sync.swap))
			must(vk.CreateSemaphore(state.dev, &sem_info, nil, &f.sync.render))
		}
		must(vk.CreateFence(state.dev, &fence_info, nil, &state.imm.frame.fence))

	}
	defer {
		for &f in state.swap.frames {
			vk.DestroyFence(state.dev, f.fence, nil)
			vk.DestroySemaphore(state.dev, f.sync.swap, nil)
			vk.DestroySemaphore(state.dev, f.sync.render, nil)
		}
		vk.DestroyFence(state.dev, state.imm.frame.fence, nil)
	}
	// }

	state.shaders = [Shader_Type]vk.ShaderModule {
		.gradient         = create_shader_module(GRAD_SHADER),
		.sky              = create_shader_module(SKY_SHADER),
		.colored_triangle = create_shader_module(COLORED_TRIANGLE_SHADER),
		.flat             = create_shader_module(FLAT_SHADER),
	}
	defer for s in state.shaders do vk.DestroyShaderModule(state.dev, s, nil)

	// compute pipeline
	must(
		vk.CreatePipelineLayout(
			state.dev,
			&vk.PipelineLayoutCreateInfo {
				sType = .PIPELINE_LAYOUT_CREATE_INFO,
				setLayoutCount = 1,
				pSetLayouts = &dslayout,
				pushConstantRangeCount = 1,
				pPushConstantRanges = &vk.PushConstantRange {
					size = size_of(state.render.compute.pushc),
					stageFlags = {.COMPUTE},
				},
			},
			nil,
			&state.render.compute.layout,
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
			layout = state.render.compute.layout,
		},
		{
			sType = .COMPUTE_PIPELINE_CREATE_INFO,
			stage = vk.PipelineShaderStageCreateInfo {
				sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
				stage = {.COMPUTE},
				module = state.shaders[.sky],
				pName = "main",
			},
			layout = state.render.compute.layout,
		},
	}
	// state.modes[.compute].layout[1] = state.modes[.compute].layout[0]
	must(
		vk.CreateComputePipelines(
			state.dev,
			{},
			u32(len(pipe_infos)),
			raw_data(pipe_infos),
			nil,
			raw_data(state.render.compute.effects[:]),
		),
	)
	defer {
		vk.DestroyPipelineLayout(state.dev, state.render.compute.layout, nil)
		for fx in state.render.compute.effects do vk.DestroyPipeline(state.dev, fx, nil)
	}

	// gfx pipeline
	{
		color_attach_render := []vk.Format{state.swap.draw.fmt}
		dynamic_states := []vk.DynamicState{.VIEWPORT, .SCISSOR}
		must(
			vk.CreatePipelineLayout(
				state.dev,
				&vk.PipelineLayoutCreateInfo {
					sType = .PIPELINE_LAYOUT_CREATE_INFO,
					setLayoutCount = 1,
					pSetLayouts = &dslayout,
					pushConstantRangeCount = 1,
					pPushConstantRanges = &vk.PushConstantRange {
						size = size_of(state.render.gfx.pushc),
						stageFlags = {.VERTEX},
					},
				},
				nil,
				&state.render.gfx.layout,
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
		blends := []vk.PipelineColorBlendAttachmentState {
			{colorWriteMask = {.R, .G, .B, .A}},
			{
				colorWriteMask = {.R, .G, .B, .A},
				blendEnable = true,
				srcColorBlendFactor = .SRC_ALPHA,
				dstColorBlendFactor = .ONE,
				colorBlendOp = .ADD,
				srcAlphaBlendFactor = .ONE,
				dstAlphaBlendFactor = .ZERO,
				alphaBlendOp = .ADD,
			},
			{
				colorWriteMask = {.R, .G, .B, .A},
				blendEnable = true,
				srcColorBlendFactor = .SRC_ALPHA,
				dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
				colorBlendOp = .ADD,
				srcAlphaBlendFactor = .ONE,
				dstAlphaBlendFactor = .ZERO,
				alphaBlendOp = .ADD,
			},
		}

		must(
			vk.CreateGraphicsPipelines(
				state.dev,
				0,
				1,
				&vk.GraphicsPipelineCreateInfo {
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
						pAttachments = &blends[BLEND],
					},
					pDepthStencilState  = &vk.PipelineDepthStencilStateCreateInfo {
						sType = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
						depthTestEnable = true,
						depthWriteEnable = true,
						depthCompareOp = .GREATER_OR_EQUAL,
						maxDepthBounds = 1,
					},
					pDynamicState       = &vk.PipelineDynamicStateCreateInfo {
						sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
						dynamicStateCount = u32(len(dynamic_states)),
						pDynamicStates = raw_data(dynamic_states),
					},
					layout              = state.render.gfx.layout,
					pNext               = &vk.PipelineRenderingCreateInfo {
						sType                   = .PIPELINE_RENDERING_CREATE_INFO_KHR,
						// viewMask:                u32,
						colorAttachmentCount    = u32(len(color_attach_render)),
						pColorAttachmentFormats = raw_data(color_attach_render),
						depthAttachmentFormat   = state.swap.depth.fmt,
						// stencilAttachmentFormat: Format,
					},
				},
				nil,
				&state.render.gfx.pipe,
			),
		)
	}
	defer {
		vk.DestroyPipelineLayout(state.dev, state.render.gfx.layout, nil)
		vk.DestroyPipeline(state.dev, state.render.gfx.pipe, nil)
	}

	vertices := []Vertex {
		{pos = {0.5, -0.5, 0}, color = {0, 0, 0, 1}},
		{pos = {0.5, 0.5, 0}, color = {0.5, 0.5, 0.5, 1}},
		{pos = {-0.5, -0.5, 0}, color = {1, 0, 0, 1}},
		{pos = {-0.5, 0.5, 0}, color = {0, 1, 0, 1}},
	}
	// log.debugf(
	// 	"%v %v %v %v",
	// 	offset_of(Vertex, pos),
	// 	offset_of(Vertex, color),
	// 	offset_of(Vertex, normal),
	// 	offset_of(Vertex, uv),
	// )
	rect_indices := []u32{0, 1, 2, 2, 1, 3}
	triangle := upload_mesh(rect_indices, vertices)
	defer destroy_mesh(triangle)

	meshes := load_gltf_meshes(GLB_PATH, .glb, color_override = true)
	state.geometry = {
		.triangle = MeshAsset {
			name = "triangle",
			surfs = []GeoSurface{{count = 6}},
			buf = triangle,
		},
		.cube = meshes[0],
		.sphere = meshes[1],
		.monkey = meshes[2],
	}
	defer for m in meshes do destroy_mesh(m.buf)

	// imgui init
	impool := create_pool(
		10,
		{
			{.STORAGE_IMAGE, 1},
			{.STORAGE_BUFFER, 1},
			{.UNIFORM_BUFFER, 1},
			{.COMBINED_IMAGE_SAMPLER, 1},
			{.SAMPLER, 1},
			{.SAMPLED_IMAGE, 1},
			{.UNIFORM_TEXEL_BUFFER, 1},
			{.STORAGE_TEXEL_BUFFER, 1},
			{.UNIFORM_BUFFER_DYNAMIC, 1},
			{.STORAGE_BUFFER_DYNAMIC, 1},
			{.INPUT_ATTACHMENT, 1},
		},
	)
	defer vk.DestroyDescriptorPool(state.dev, impool, nil)
	imgui.CreateContext()
	defer imgui.DestroyContext()
	imvk.LoadFunctions(
		vk.API_VERSION_1_3,
		proc "c" (function_name: cstring, user_data: rawptr) -> vk.ProcVoidFunction {
			return vk.GetInstanceProcAddr((vk.Instance)(user_data), function_name)
		},
		state.api,
	)
	imdl.InitForVulkan(state.handle)
	defer imdl.Shutdown()
	color_attach := []vk.Format{SURF_FMT.format}
	imvk.Init(
		&imvk.InitInfo {
			ApiVersion = vk.API_VERSION_1_3,
			Instance = state.api,
			PhysicalDevice = state.phy.dev,
			Device = state.dev,
			QueueFamily = state.submit.idx,
			Queue = state.submit.q,
			DescriptorPool = impool,
			MinImageCount = MAX_FRAMES_IN_FLIGHT,
			ImageCount = MAX_FRAMES_IN_FLIGHT,
			// MSAASamples = ._1,
			UseDynamicRendering = true,
			PipelineRenderingCreateInfo = {
				sType = .PIPELINE_RENDERING_CREATE_INFO_KHR,
				colorAttachmentCount = u32(len(color_attach)),
				pColorAttachmentFormats = raw_data(color_attach),
				depthAttachmentFormat = state.swap.depth.fmt,
			},
			CheckVkResultFn = imguiCheckVkResult,
			MinAllocationSize = 1024 * 1024,
		},
	)
	defer imvk.Shutdown()
	imvk.CreateFontsTexture()
	defer imvk.DestroyFontsTexture()

	defer vk.DeviceWaitIdle(state.dev)

	frame_num: u64
	do_render: bool = true
	effect: Compute_Effect
	compute_bound := i32(len(Compute_Effect))
	geo_bound := i32(len(Geometry))
	geometry: Geometry
	scale: f32 = 1
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
				efname, _ := reflect.enum_name_from_value(effect)
				geoname, _ := reflect.enum_name_from_value(geometry)
				imgui.Text(
					strings.clone_to_cstring(
						fmt.tprintf("Effect: %v\tGeometry: %v", efname, geoname),
					),
				)
				imgui.SliderInt("FX", cast(^i32)&effect, 0, compute_bound - 1)
				imgui.SliderInt("GEO", cast(^i32)&geometry, 0, geo_bound - 1)
				imgui.SliderFloat("SCALE", cast(^f32)&scale, 0.3, 1)
				imgui.InputFloat4("transform", &state.render.compute.pushc.tr)
				imgui.InputFloat4("camera", &state.render.compute.pushc.cam)
				imgui.InputFloat4("position", &state.render.compute.pushc.pos)
				imgui.InputFloat4("post-transform", &state.render.compute.pushc.post)
			}
			imgui.End()
			imgui.Render()
			extent := vk.Extent2D {
				width  = u32(f32(state.swap.ext.width) * scale),
				height = u32(f32(state.swap.ext.height) * scale),
			}
			r := draw(geometry, effect, extent, frame_num)
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

create_pool :: proc(total: u32, ratios: []vk.DescriptorPoolSize) -> (pool: vk.DescriptorPool) {
	for &p in ratios do p.descriptorCount *= total
	must(
		vk.CreateDescriptorPool(
			state.dev,
			&vk.DescriptorPoolCreateInfo {
				sType = .DESCRIPTOR_POOL_CREATE_INFO,
				flags = {.FREE_DESCRIPTOR_SET},
				maxSets = total,
				poolSizeCount = u32(len(ratios)),
				pPoolSizes = raw_data(ratios),
			},
			nil,
			&pool,
		),
	)
	return
}

descriptor_set :: proc(
	pool: vk.DescriptorPool,
	dsl: vk.DescriptorSetLayout,
) -> (
	ds: vk.DescriptorSet,
) {
	dsl := dsl
	must(
		vk.AllocateDescriptorSets(
			state.dev,
			&vk.DescriptorSetAllocateInfo {
				sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
				descriptorPool = pool,
				descriptorSetCount = 1,
				pSetLayouts = &dsl,
			},
			&ds,
		),
	)
	return
}

descriptor_layout :: proc(
	stages: vk.ShaderStageFlags,
	descriptors: []vk.DescriptorType,
) -> (
	dsl: vk.DescriptorSetLayout,
) {
	binds := make([]vk.DescriptorSetLayoutBinding, len(descriptors))
	for &d, i in descriptors {
		binds[i] = {
			descriptorCount = 1,
			stageFlags      = stages,
			descriptorType  = d,
		}
	}
	// TODO: need same amount as binds
	flags := []vk.DescriptorBindingFlags {
		{
			.UPDATE_AFTER_BIND,
			.UPDATE_UNUSED_WHILE_PENDING,
			.PARTIALLY_BOUND,
			// .VARIABLE_DESCRIPTOR_COUNT       ,
		},
	}
	must(
		vk.CreateDescriptorSetLayout(
			state.dev,
			&vk.DescriptorSetLayoutCreateInfo {
				sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				bindingCount = u32(len(binds)),
				pBindings = raw_data(binds),
				pNext = &vk.DescriptorSetLayoutBindingFlagsCreateInfo {
					sType = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
					bindingCount = u32(len(binds)),
					pBindingFlags = raw_data(flags),
				},
			},
			nil,
			&dsl,
		),
	)
	return
}

load_gltf_meshes :: proc(
	path: string,
	ftype: gltf.file_type,
	color_override: bool = false,
) -> (
	meshes: []MeshAsset,
) {
	opts := gltf.options {
		type = ftype,
	}
	file, r := gltf.parse_file(opts, strings.clone_to_cstring(path))
	defer gltf.free(file)
	assert(r == .success, "gltf: could not parse file")
	r = gltf.load_buffers(opts, file, "assets/basicmesh.glb")
	assert(r == .success, "gltf: coult not load buffers")
	assert(gltf.validate(file) == .success, "gltf: coult not validate")
	copyData :: proc(accessor: ^gltf.accessor, dst: rawptr) {
		bufferView := accessor.buffer_view
		data := bufferView.data
		if data == nil {
			data = bufferView.buffer.data
		}
		src := rawptr(uintptr(data) + uintptr(bufferView.offset))
		size := int(bufferView.size)
		mem.copy(dst, src, size)
	}

	meshes = make([]MeshAsset, len(file.meshes))

	for &mesh, m in file.meshes {
		idxs: [dynamic]u32
		vtxs: #soa[dynamic]Vertex
		voff: u32
		ioff: u32
		surfs: [dynamic]GeoSurface
		for &p in mesh.primitives {
			if p.type != .triangles {
				log.warnf("skipping %v primitive", p.type)
				continue
			}
			surf := GeoSurface {
				start = ioff,
				count = u32(p.indices.count),
			}
			append(&surfs, surf)

			resize(&idxs, int(surf.count))
			idxs := idxs[ioff:]
			ioff += surf.count

			if p.indices.component_type == .r_32u {
				copyData(p.indices, raw_data(idxs))
			} else if p.indices.component_type == .r_16u {
				data := make([]u16, int(p.indices.count))
				defer delete(data)
				copyData(p.indices, raw_data(data))
				for v, i in data do idxs[i] = u32(v)
			} else if p.indices.component_type == .r_8u {
				data := make([]u8, int(p.indices.count))
				defer delete(data)
				copyData(p.indices, raw_data(data))
				for v, i in data do idxs[i] = u32(v)
			}
			resize(&vtxs, p.attributes[0].data.count)
			vtxs := vtxs[voff:]
			voff += u32(p.attributes[0].data.count)

			for &attribute in p.attributes {
				#partial switch attribute.type {
				case .position:
					assert(attribute.data.type == .vec3)
					copyData(attribute.data, raw_data(vtxs.pos[:]))
				case .texcoord:
					assert(attribute.data.type == .vec2)
					copyData(attribute.data, raw_data(vtxs.uv[:]))
				case .normal:
					assert(attribute.data.type == .vec3)
					copyData(attribute.data, raw_data(vtxs.normal[:]))
				case .color:
					assert(attribute.data.type == .vec4)
					copyData(attribute.data, raw_data(vtxs.color[:]))
				}
			}
		}
		verts := make([]Vertex, len(vtxs))
		for v, i in vtxs {
			verts[i] = v
			if color_override do verts[i].color = vec4{v.normal.x, v.normal.y, v.normal.z, 1.0}
		}
		name := strings.clone_from_cstring(mesh.name)
		if len(idxs) == 0 {
			log.debugf("skipping mesh %v with 0 indices and surfs %v", name, surfs)
			continue
		} else if len(vtxs) == 0 {
			log.debugf("skipping mesh %v with 0 vertices and surfs %v", name, surfs)
			continue
		}
		log.infof("allocating mesh %v with %v verts and %v indices", name, len(idxs), len(verts))
		meshes[m] = {
			name  = name,
			surfs = surfs[:],
			buf   = upload_mesh(idxs[:], verts),
		}
	}
	return
}

destroy_mesh :: proc(mesh: MeshBuffer) {
	vk.DestroyBuffer(state.dev, mesh.vertices, nil)
	vk.FreeMemory(state.dev, mesh.vtx_mem, nil)
	vk.DestroyBuffer(state.dev, mesh.indices, nil)
	vk.FreeMemory(state.dev, mesh.idx_mem, nil)
}

upload_mesh :: proc(indices: []u32, vertices: []Vertex) -> (m: MeshBuffer) {
	vtxbuf_size := size_of(Vertex) * len(vertices)
	m.vertices, m.vtx_mem = create_buffer(
		vk.DeviceSize(vtxbuf_size),
		{.TRANSFER_DST, .STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
		.gpu_only,
	)
	addr_info := vk.BufferDeviceAddressInfo {
		sType  = .BUFFER_DEVICE_ADDRESS_INFO,
		buffer = m.vertices,
	}
	m.vtx_addr = vk.GetBufferDeviceAddress(state.dev, &addr_info)

	idbuf_size := size_of(u32) * len(indices)
	m.indices, m.idx_mem = create_buffer(
		vk.DeviceSize(idbuf_size),
		{.TRANSFER_DST, .INDEX_BUFFER},
		.gpu_only,
	)

	total_size := vtxbuf_size + idbuf_size
	staging, devmem := create_buffer(vk.DeviceSize(total_size), {.TRANSFER_SRC}, .cpu_copy)
	addr_info.buffer = staging
	defer vk.DestroyBuffer(state.dev, staging, nil)
	defer vk.FreeMemory(state.dev, devmem, nil)

	dest := map_memory(devmem, vk.DeviceSize(total_size))
	assert(vtxbuf_size == copy(dest[:vtxbuf_size], slice.reinterpret([]byte, vertices)))
	assert(idbuf_size == copy(dest[vtxbuf_size:], slice.reinterpret([]byte, indices)))
	Args :: struct {
		src, vtx, idx: vk.Buffer,
		offset, total: vk.DeviceSize,
	}
	imm_exec(proc(cmd: vk.CommandBuffer, args: Args) {
		vk.CmdCopyBuffer(cmd, args.src, args.vtx, 1, &vk.BufferCopy{size = args.offset})
		vk.CmdCopyBuffer(
			cmd,
			args.src,
			args.idx,
			1,
			&vk.BufferCopy{srcOffset = args.offset, size = args.total - args.offset},
		)
	}, Args{staging, m.vertices, m.indices, vk.DeviceSize(vtxbuf_size), vk.DeviceSize(total_size)})
	return
}

map_memory :: proc(dm: vk.DeviceMemory, size: vk.DeviceSize) -> []byte {
	d: rawptr
	vk.MapMemory(state.dev, dm, 0, size, {}, &d)
	return slice.bytes_from_ptr(d, int(size))
}

create_shader_module :: proc(code: []byte) -> (module: vk.ShaderModule) {
	create_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(code),
		pCode    = raw_data(slice.reinterpret([]u32, code)),
	}
	must(vk.CreateShaderModule(state.dev, &create_info, nil, &module))
	return
}

mtype :: proc(properties: vk.MemoryPropertyFlags, required: u32) -> u32 {
	for mtype, i in state.phy.mem do if (1 << u32(i)) & required != 0 && (properties & mtype) == properties do return u32(i)
	return 0
}

Buffer_Access :: enum {
	// sequential access, no device access
	cpu_copy,
	// cached, device local, may not be host visible
	cpu_random_access,
	// device local
	gpu_only,
}

create_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	access: Buffer_Access,
) -> (
	buf: vk.Buffer,
	addr: vk.DeviceMemory,
) {
	must(
		vk.CreateBuffer(
			state.dev,
			&vk.BufferCreateInfo{sType = .BUFFER_CREATE_INFO, size = size, usage = usage},
			nil,
			&buf,
		),
	)

	required := vk.MemoryRequirements2 {
		sType = .MEMORY_REQUIREMENTS_2,
	}
	vk.GetBufferMemoryRequirements2(
		state.dev,
		&vk.BufferMemoryRequirementsInfo2 {
			sType = .BUFFER_MEMORY_REQUIREMENTS_INFO_2,
			buffer = buf,
		},
		&required,
	)
	mem_props: vk.MemoryPropertyFlags
	switch access {
	case .cpu_copy:
		mem_props |= {.HOST_VISIBLE}
	case .cpu_random_access:
		mem_props |= {.HOST_CACHED, .DEVICE_LOCAL}
	case .gpu_only:
		mem_props |= {.DEVICE_LOCAL}
	}
	// if !cpu_only do mem_props |= {.DEVICE_LOCAL}
	must(
		vk.AllocateMemory(
			state.dev,
			&vk.MemoryAllocateInfo {
				sType = .MEMORY_ALLOCATE_INFO,
				allocationSize = required.memoryRequirements.size,
				memoryTypeIndex = mtype(mem_props, required.memoryRequirements.memoryTypeBits),
				pNext = &vk.MemoryAllocateFlagsInfo {
					sType = .MEMORY_ALLOCATE_FLAGS_INFO,
					flags = {.DEVICE_ADDRESS},
				},
			},
			nil,
			&addr,
		),
	)

	must(
		vk.BindBufferMemory2(
			state.dev,
			1,
			&vk.BindBufferMemoryInfo {
				sType = .BIND_BUFFER_MEMORY_INFO,
				buffer = buf,
				memory = addr,
			},
		),
	)
	return
}

create_image :: proc(info: ^vk.ImageCreateInfo) -> (img: vk.Image, addr: vk.DeviceMemory) {
	must(vk.CreateImage(state.dev, info, nil, &img))

	memReq := vk.MemoryRequirements2 {
		sType = .MEMORY_REQUIREMENTS_2,
	}
	vk.GetImageMemoryRequirements2(
		state.dev,
		&vk.ImageMemoryRequirementsInfo2{sType = .IMAGE_MEMORY_REQUIREMENTS_INFO_2, image = img},
		&memReq,
	)
	must(
		vk.AllocateMemory(
			state.dev,
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
			state.dev,
			1,
			&vk.BindImageMemoryInfo{sType = .BIND_IMAGE_MEMORY_INFO, image = img, memory = addr},
		),
	)
	return
}

draw_background :: proc(
	buf: vk.CommandBuffer,
	img: vk.Image,
	extent: vk.Extent2D,
	effect: Compute_Effect,
	n: u64,
) {
	vk.CmdBindPipeline(buf, .COMPUTE, state.render.compute.effects[effect])
	vk.CmdBindDescriptorSets(
		buf,
		.COMPUTE,
		state.render.compute.layout,
		0,
		1,
		&state.descriptor,
		0,
		nil,
	)
	state.render.compute.pushc.tr = {0.0, 0.0, math.abs(math.sin(f32(n) / 120.0)), 1.0}
	state.render.compute.pushc.cam = {0.0, math.abs(math.cos(f32(n) / 120.0)), 0.0, 1.0}
	vk.CmdPushConstants(
		buf,
		state.render.compute.layout,
		{.COMPUTE},
		0,
		size_of(state.render.compute.pushc),
		&state.render.compute.pushc,
	)
	vk.CmdDispatch(
		buf,
		u32(math.ceil(f32(extent.width) / 16.0)),
		u32(math.ceil(f32(extent.height) / 16.0)),
		1,
	)
}

bind_mem :: proc(
	cmd: vk.CommandBuffer,
	layout: vk.DescriptorSetLayout,
	point: vk.PipelineBindPoint,
	first: u32,
	ds: vk.DescriptorSet,
) {

}

draw_imgui :: proc(buf: vk.CommandBuffer, view: vk.ImageView, extent: vk.Extent2D) {
	vk.CmdBeginRendering(
		buf,
		&vk.RenderingInfo {
			sType = .RENDERING_INFO,
			renderArea = {extent = extent},
			layerCount = 1,
			colorAttachmentCount = 1,
			pColorAttachments = &vk.RenderingAttachmentInfo {
				sType = .RENDERING_ATTACHMENT_INFO,
				imageView = view,
				imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
				loadOp = .LOAD,
				storeOp = .STORE,
			},
		},
	)
	imvk.RenderDrawData(imgui.GetDrawData(), buf)
	vk.CmdEndRendering(buf)
}

imm_exec :: proc(f: proc(_: vk.CommandBuffer, _: $T), args: T) {
	frame := &state.imm.frame
	must(vk.ResetFences(state.dev, 1, &frame.fence))

	must(vk.ResetCommandBuffer(frame.cmd, {}))
	must(
		vk.BeginCommandBuffer(
			frame.cmd,
			&vk.CommandBufferBeginInfo {
				sType = .COMMAND_BUFFER_BEGIN_INFO,
				flags = {.ONE_TIME_SUBMIT},
			},
		),
	)
	f(frame.cmd, args)
	must(vk.EndCommandBuffer(frame.cmd))

	submit_info := vk.SubmitInfo2 {
		sType                  = .SUBMIT_INFO_2,
		commandBufferInfoCount = 1,
		pCommandBufferInfos    = &vk.CommandBufferSubmitInfo {
			sType = .COMMAND_BUFFER_SUBMIT_INFO,
			commandBuffer = frame.cmd,
		},
	}
	must(vk.QueueSubmit2(state.submit.q, 1, &submit_info, frame.fence))
	must(vk.WaitForFences(state.dev, 1, &frame.fence, true, max(u64)))
}

draw_geometry :: proc(ig: Geometry, buf: vk.CommandBuffer, extent: vk.Extent2D) {
	vk.CmdBeginRendering(
		buf,
		&vk.RenderingInfo {
			sType = .RENDERING_INFO,
			renderArea = {extent = extent},
			layerCount = 1,
			colorAttachmentCount = 1,
			pColorAttachments = &vk.RenderingAttachmentInfo {
				sType = .RENDERING_ATTACHMENT_INFO,
				imageView = state.swap.draw.view,
				imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
				loadOp = .LOAD,
				storeOp = .STORE,
			},
			pDepthAttachment = &vk.RenderingAttachmentInfo {
				sType = .RENDERING_ATTACHMENT_INFO,
				imageView = state.swap.depth.view,
				imageLayout = .DEPTH_ATTACHMENT_OPTIMAL,
				loadOp = .CLEAR,
				storeOp = .STORE,
			},
		},
	)
	vk.CmdBindPipeline(buf, .GRAPHICS, state.render.gfx.pipe)
	vk.CmdSetViewport(
		buf,
		0,
		1,
		&vk.Viewport{width = f32(extent.width), height = f32(extent.height), maxDepth = 1},
	)
	vk.CmdSetScissor(buf, 0, 1, &vk.Rect2D{extent = extent})

	perp := linalg.matrix4_perspective(
		math.PI / 2.0,
		f32(extent.width) / f32(extent.height),
		0.1,
		100,
		flip_z_axis = true,
	)
	perp[1][1] *= -1
	state.render.gfx.pushc.tr = perp * linalg.matrix4_translate_f32({0, 0, -3})
	geo := &state.geometry[ig]
	state.render.gfx.pushc.vert_addr = geo.buf.vtx_addr
	vk.CmdPushConstants(
		buf,
		state.render.gfx.layout,
		{.VERTEX},
		0,
		size_of(state.render.gfx.pushc),
		&state.render.gfx.pushc,
	)
	vk.CmdBindIndexBuffer(buf, geo.buf.indices, 0, .UINT32)

	vk.CmdDrawIndexed(buf, geo.surfs[0].count, 1, geo.surfs[0].start, 0, 0)
	vk.CmdEndRendering(buf)
}

draw :: proc(
	geometry: Geometry,
	effect: Compute_Effect,
	extent: vk.Extent2D,
	n: u64,
) -> vk.Result {
	fid := n % MAX_FRAMES_IN_FLIGHT
	frame := &state.swap.frames[fid]
	sync := frame.sync
	must(vk.WaitForFences(state.dev, 1, &frame.fence, true, max(u64)))
	must(vk.ResetFences(state.dev, 1, &frame.fence))

	image_index: u32
	if r := vk.AcquireNextImageKHR(
		state.dev,
		state.swap.chain,
		max(u64),
		sync.swap,
		0,
		&image_index,
	); r != .SUCCESS && r != .SUBOPTIMAL_KHR {
		return r
	}

	must(vk.ResetCommandBuffer(frame.cmd, {}))
	must(
		vk.BeginCommandBuffer(
			frame.cmd,
			&vk.CommandBufferBeginInfo {
				sType = .COMMAND_BUFFER_BEGIN_INFO,
				flags = {.ONE_TIME_SUBMIT},
			},
		),
	)

	img := state.swap.images[image_index]
	record(frame.cmd, state.swap.draw.img, .UNDEFINED, .GENERAL)
	draw_background(frame.cmd, img, extent, effect, n)

	record(frame.cmd, state.swap.draw.img, .GENERAL, .COLOR_ATTACHMENT_OPTIMAL)
	record(frame.cmd, state.swap.depth.img, .UNDEFINED, .DEPTH_ATTACHMENT_OPTIMAL)
	draw_geometry(geometry, frame.cmd, extent)

	record(frame.cmd, state.swap.draw.img, .COLOR_ATTACHMENT_OPTIMAL, .TRANSFER_SRC_OPTIMAL)
	record(frame.cmd, img, .UNDEFINED, .TRANSFER_DST_OPTIMAL)
	cpimg(frame.cmd, state.swap.draw.img, img, extent, state.swap.ext)

	record(frame.cmd, img, .TRANSFER_DST_OPTIMAL, .COLOR_ATTACHMENT_OPTIMAL)
	draw_imgui(frame.cmd, state.swap.views[image_index], extent)

	record(frame.cmd, img, .COLOR_ATTACHMENT_OPTIMAL, .PRESENT_SRC_KHR)

	must(vk.EndCommandBuffer(frame.cmd))

	ws := sub_info(sync.swap, {.COLOR_ATTACHMENT_OUTPUT_KHR})
	ps := sub_info(sync.render, {.ALL_GRAPHICS_KHR})
	submit_info := vk.SubmitInfo2 {
		sType                    = .SUBMIT_INFO_2,
		commandBufferInfoCount   = 1,
		pCommandBufferInfos      = &vk.CommandBufferSubmitInfo {
			sType = .COMMAND_BUFFER_SUBMIT_INFO,
			commandBuffer = frame.cmd,
		},
		waitSemaphoreInfoCount   = 1,
		pWaitSemaphoreInfos      = &ws,
		signalSemaphoreInfoCount = 1,
		pSignalSemaphoreInfos    = &ps,
	}
	must(vk.QueueSubmit2(state.submit.q, 1, &submit_info, frame.fence))

	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		// pNext              = &vk.SwapchainPresentFenceInfoEXT {
		// 	sType = .SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
		// 	swapchainCount = 1,
		// 	pFences = &state.swapchain.frames[(image_index + 1) % MAX_FRAMES_IN_FLIGHT].fence,
		// },
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &sync.render,
		swapchainCount     = 1,
		pSwapchains        = &state.swap.chain,
		pImageIndices      = &image_index,
	}
	return vk.QueuePresentKHR(state.submit.q, &present_info)
}

cpimg :: proc(buf: vk.CommandBuffer, src, dest: vk.Image, sext, dext: vk.Extent2D) {
	vk.CmdBlitImage2(
		buf,
		&vk.BlitImageInfo2 {
			sType = .BLIT_IMAGE_INFO_2,
			srcImage = src,
			srcImageLayout = .TRANSFER_SRC_OPTIMAL,
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
	return {aspectMask = aspect, levelCount = 1, layerCount = 1}
}

recreate_swapchain :: proc() {
	vk.DeviceWaitIdle(state.dev)
	destroy_swapchain()

	create_swapchain()

	if state.swap.ext.width < state.render.ext.width && state.swap.ext.height < state.render.ext.height do return
	log.warnf("recreating draw surface with dim %v", state.swap.ext)

	destroy_image(state.swap.draw)
	destroy_image(state.swap.depth)

	create_drawsurf(state.swap.ext)
}

create_drawsurf :: proc(ext: vk.Extent2D) {
	state.render.ext = ext

	ext3 := vk.Extent3D {
		width  = ext.width,
		height = ext.height,
		depth  = 1,
	}
	state.swap.draw.fmt = .R16G16B16A16_SFLOAT

	state.swap.draw.img, state.swap.draw.mem = create_image(
		&vk.ImageCreateInfo {
			sType = .IMAGE_CREATE_INFO,
			imageType = .D2,
			format = state.swap.draw.fmt,
			extent = ext3,
			mipLevels = 1,
			arrayLayers = 1,
			samples = {._1},
			tiling = .OPTIMAL,
			usage = {.TRANSFER_SRC, .TRANSFER_DST, .STORAGE, .COLOR_ATTACHMENT},
		},
	)
	must(
		vk.CreateImageView(
			state.dev,
			&vk.ImageViewCreateInfo {
				sType = .IMAGE_VIEW_CREATE_INFO,
				image = state.swap.draw.img,
				viewType = .D2,
				format = state.swap.draw.fmt,
				subresourceRange = sub_range({.COLOR}),
			},
			nil,
			&state.swap.draw.view,
		),
	)

	write_set(
		state.descriptor,
		0,
		state.swap.draw.view,
		state.swap.draw.img,
		{},
		.GENERAL,
		.STORAGE_IMAGE,
	)

	state.swap.depth.fmt = .D32_SFLOAT
	state.swap.depth.img, state.swap.depth.mem = create_image(
		&vk.ImageCreateInfo {
			sType = .IMAGE_CREATE_INFO,
			imageType = .D2,
			format = state.swap.depth.fmt,
			extent = ext3,
			mipLevels = 1,
			arrayLayers = 1,
			samples = {._1},
			tiling = .OPTIMAL,
			usage = {.DEPTH_STENCIL_ATTACHMENT},
		},
	)
	must(
		vk.CreateImageView(
			state.dev,
			&vk.ImageViewCreateInfo {
				sType = .IMAGE_VIEW_CREATE_INFO,
				image = state.swap.depth.img,
				viewType = .D2,
				format = state.swap.depth.fmt,
				subresourceRange = sub_range({.DEPTH}),
			},
			nil,
			&state.swap.depth.view,
		),
	)
}

write_set :: proc(
	set: vk.DescriptorSet,
	binding: u32,
	view: vk.ImageView,
	img: vk.Image,
	sampler: vk.Sampler,
	layout: vk.ImageLayout,
	dtype: vk.DescriptorType,
) {
	vk.UpdateDescriptorSets(
		state.dev,
		1,
		&vk.WriteDescriptorSet {
			sType = .WRITE_DESCRIPTOR_SET,
			dstSet = set,
			dstBinding = binding,
			descriptorCount = 1,
			descriptorType = dtype,
			pImageInfo = &vk.DescriptorImageInfo {
				imageView = view,
				imageLayout = layout,
				sampler = sampler,
			},
		},
		0,
		nil,
	)
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

	w, h: i32
	sdl.GetWindowSizeInPixels(state.handle, &w, &h)
	state.swap.ext = vk.Extent2D {
		width  = clamp(u32(w), state.caps.minImageExtent.width, state.caps.maxImageExtent.width),
		height = clamp(u32(h), state.caps.minImageExtent.height, state.caps.maxImageExtent.height),
	}

	must(
		vk.CreateSwapchainKHR(
			state.dev,
			&vk.SwapchainCreateInfoKHR {
				sType            = .SWAPCHAIN_CREATE_INFO_KHR,
				surface          = state.surf,
				minImageCount    = MAX_FRAMES_IN_FLIGHT,
				imageFormat      = SURF_FMT.format,
				imageColorSpace  = SURF_FMT.colorSpace,
				imageExtent      = state.swap.ext,
				imageArrayLayers = 1,
				imageUsage       = {.COLOR_ATTACHMENT, .TRANSFER_DST},
				preTransform     = state.caps.currentTransform,
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
			&state.swap.chain,
		),
	)

	count: u32 = MAX_FRAMES_IN_FLIGHT
	must(
		vk.GetSwapchainImagesKHR(
			state.dev,
			state.swap.chain,
			&count,
			raw_data(state.swap.images[:]),
		),
	)

	create_info := vk.ImageViewCreateInfo {
		sType            = .IMAGE_VIEW_CREATE_INFO,
		viewType         = .D2,
		format           = SURF_FMT.format,
		subresourceRange = sub_range({.COLOR}),
	}
	for image, i in state.swap.images {
		create_info.image = image
		must(vk.CreateImageView(state.dev, &create_info, nil, &state.swap.views[i]))
	}
}

destroy_swapchain :: proc() {
	for view in state.swap.views {
		vk.DestroyImageView(state.dev, view, nil)
	}
	vk.DestroySwapchainKHR(state.dev, state.swap.chain, nil)
}

destroy_image :: proc(im: Image) {
	vk.DestroyImageView(state.dev, im.view, nil)
	vk.DestroyImage(state.dev, im.img, nil)
	vk.FreeMemory(state.dev, im.mem, nil)
}

N_PHYS :: 3

@(require_results)
pick_physical_device :: proc() -> vk.Result {
	phys: Smaller_Array(N_PHYS, vk.PhysicalDevice)
	phys.len = N_PHYS
	vk.EnumeratePhysicalDevices(state.api, &phys.len, raw_data(phys.data[:])) or_return
	assert(phys.len != 0, "no GPU found")

	props: vk.PhysicalDeviceProperties
	features: vk.PhysicalDeviceFeatures
	exts: Smaller_Array(N_EXTS, vk.ExtensionProperties)

	qfams: Smaller_Array(N_QFAM, vk.QueueFamilyProperties)

	best_device_score: u64
	devloop: for device in phys.data[:phys.len] {
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
		exts.len = N_EXTS
		vk.EnumerateDeviceExtensionProperties(
			device,
			nil,
			&exts.len,
			raw_data(exts.data[:]),
		) or_continue

		rextloop: for rext in DEVICE_EXTENSIONS {
			for &extension in exts.data[:exts.len] do if byte_arr_str(&extension.extensionName) == string(rext) do continue rextloop
			log.infof("vulkan: device does not support required extension %q", rext)
			continue devloop
		}

		// check queue support
		qfams.len = N_QFAM
		vk.GetPhysicalDeviceQueueFamilyProperties(device, &qfams.len, raw_data(qfams.data[:]))
		qloop: for rq in Q_CAPS {
			for &q in qfams.data[:qfams.len] do if rq in q.queueFlags do continue qloop
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
			state.phy.dev = device
			best_device_score = score
		}
		log.infof("vulkan: device %q scored %v", name, score)
	}

	if best_device_score == 0 do log.panic("vulkan: no suitable GPU found")
	// fprops: vk.FormatProperties
	// vk.GetPhysicalDeviceFormatProperties(state.physical_device, .R16G16B16A16_SFLOAT, &fprops)
	// log.debugf("%v", fprops)
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
