run: build
	MOLTEN=$$HOME/MoltenVK/MoltenVK/dynamic/dylib/macOS ; DYLD_LIBRARY_PATH=$$MOLTEN_SDK VK_ICD_FILENAMES=$$MOLTEN/MoltenVK_icd.json ./hoenn

debug: build
	MOLTEN=$$HOME/MoltenVK/MoltenVK/dynamic/dylib/macOS ; DYLD_LIBRARY_PATH=$$MOLTEN_SDK VK_ICD_FILENAMES=$$MOLTEN/MoltenVK_icd.json lldb hoenn

build:
	glslc shaders/gradient.comp -o shaders/gradient.spv
	glslc shaders/sky.comp -o shaders/sky.spv
	glslc shaders/colored_triangle.vert -o shaders/colored_triangle.spv
	glslc shaders/flat.frag -o shaders/flat.spv
	odin build . -debug
