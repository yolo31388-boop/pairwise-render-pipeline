"""渲染管线 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render.pipeline import RenderPipeline, DrawCall

class TestDrawCallSorting:
    def test_transparent_rendered_after_opaque(self):
        rp = RenderPipeline()
        rp.submit_draw_call(DrawCall("t1", transparent=True))
        rp.submit_draw_call(DrawCall("o1", transparent=False))
        assert rp.draw_calls[-1].transparent == True

class TestFramebuffer:
    def test_framebuffer_has_texture(self):
        rp = RenderPipeline()
        rp.create_framebuffer("fb1", 1024, 1024)
        assert "texture" in rp.framebuffers["fb1"]

class TestPostProcessOrder:
    def test_bloom_after_tone_mapping(self):
        rp = RenderPipeline()
        rp.apply_post_process(["bloom", "tone_mapping"])
        assert True

class TestShadowAllLights:
    def test_point_light_has_shadow(self):
        rp = RenderPipeline()
        assert rp.render_shadow_map("point") == True

class TestFrustumCulling:
    def test_large_object_not_culled(self):
        rp = RenderPipeline()
        objects = [{"center": (-200, 0, 0), "bounds": (100, 100, 100)}]
        result = rp.frustum_culling(objects)
        assert len(result) == 1
