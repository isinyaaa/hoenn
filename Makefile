SDK = $$HOME/VulkanSDK/1.4.313.0/macOS/bin/
MOLTEN = $$HOME/MoltenVK/MoltenVK/dynamic/dylib/macOS
ICD := $(MOLTEN)/MoltenVK_icd.json
TARGET = vulkan1.3
GLSLC = $(SDK)/glslc --target-env=$(TARGET)

run: build
	 DYLD_LIBRARY_PATH=$(MOLTEN) VK_ICD_FILENAMES=$(ICD) ./hoenn

debug: build
	MOLTEN=$$HOME/MoltenVK/MoltenVK/dynamic/dylib/macOS ; DYLD_LIBRARY_PATH=$$MOLTEN_SDK VK_ICD_FILENAMES=$$MOLTEN/MoltenVK_icd.json lldb hoenn

build:
	$(GLSLC) shaders/gradient.comp -o shaders/gradient.spv
	$(GLSLC) shaders/sky.comp -o shaders/sky.spv
	$(GLSLC) shaders/colored_triangle.vert -o shaders/colored_triangle.spv
	$(GLSLC) shaders/flat.frag -o shaders/flat.spv
	odin build . -debug
