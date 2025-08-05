#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

try:
    from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
    print("✅ Successfully imported ComprehensiveNVLabsParser")
    
    parser = ComprehensiveNVLabsParser()
    print(f"✅ Parser created: canvas_size={parser.canvas_size}, base_scaling_factor={parser.base_scaling_factor}")
    
    # Test parsing
    test_commands = ["line_triangle_1.000-0.500", "line_normal_0.600-0.750"]
    image = parser.process_action_commands_to_image(test_commands, "test")
    
    if image is not None:
        print(f"✅ Image generated: {image.shape}, pixels: {image.size}")
    else:
        print("❌ Failed to generate image")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
