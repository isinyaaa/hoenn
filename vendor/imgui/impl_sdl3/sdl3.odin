package impl_sdl3

import sdl "vendor:sdl3"

when ODIN_OS == .Windows {
	when ODIN_ARCH == .amd64 {
		foreign import lib "../imgui_windows_x64.lib"
	} else {
		foreign import lib "../imgui_windows_arm64.lib"
	}
} else when ODIN_OS == .Linux {
	when ODIN_ARCH == .amd64 {
		foreign import lib "../imgui_linux_x64.a"
	} else {
		foreign import lib "../imgui_linux_arm64.a"
	}
} else when ODIN_OS == .Darwin {
	when ODIN_ARCH == .amd64 {
		foreign import lib "../imgui_darwin_x64.a"
	} else {
		foreign import lib "../imgui_darwin_arm64.a"
	}
}

@(link_prefix = "ImGui_ImplSDL3_", default_calling_convention = "c")
foreign lib {
	InitForVulkan :: proc(window: ^sdl.Window) -> bool ---
	Shutdown :: proc() ---
	ProcessEvent :: proc(event: ^sdl.Event) -> bool ---
	NewFrame :: proc() ---
}
