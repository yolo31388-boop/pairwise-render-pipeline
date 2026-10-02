"""渲染管线 - 红态测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render.pipeline import RenderPipeline, DrawCall


class TestDrawCallSorting:
    def test_transparent_rendered_after_opaque(self):
        rp = RenderPipeline()
        rp.submit_draw_call(DrawCall("t1", transparent=True))
        rp.submit_draw_call(DrawCall("o1", transparent=False))
        assert rp.draw_calls[-1].transparent is True

    def test_opaque_sorted_front_to_back(self):
        rp = RenderPipeline()
        rp.submit_draw_call(DrawCall("far", depth=10.0, material="m"))
        rp.submit_draw_call(DrawCall("near", depth=1.0, material="m"))
        rp.submit_draw_call(DrawCall("mid", depth=5.0, material="m"))
        opaque = [c for c in rp.draw_calls if not c.transparent]
        assert [c.did for c in opaque] == ["near", "mid", "far"]

    def test_transparent_sorted_back_to_front(self):
        rp = RenderPipeline()
        rp.submit_draw_call(DrawCall("t-near", depth=1.0, transparent=True))
        rp.submit_draw_call(DrawCall("t-far", depth=9.0, transparent=True))
        rp.submit_draw_call(DrawCall("t-mid", depth=5.0, transparent=True))
        assert [c.did for c in rp.draw_calls] == ["t-far", "t-mid", "t-near"]

    def test_same_material_grouped(self):
        rp = RenderPipeline()
        rp.submit_draw_call(DrawCall("b1", depth=2.0, material="b"))
        rp.submit_draw_call(DrawCall("a1", depth=1.0, material="a"))
        rp.submit_draw_call(DrawCall("b2", depth=3.0, material="b"))
        rp.submit_draw_call(DrawCall("a2", depth=4.0, material="a"))
        materials = [c.material for c in rp.draw_calls]
        assert materials == ["a", "a", "b", "b"]


class TestFramebuffer:
    def test_framebuffer_has_texture(self):
        rp = RenderPipeline()
        rp.create_framebuffer("fb1", 1024, 1024)
        assert "texture" in rp.framebuffers["fb1"]

    def test_framebuffer_complete(self):
        rp = RenderPipeline()
        assert rp.create_framebuffer("fb2", 800, 600) is True
        assert rp.framebuffers["fb2"]["complete"] is True
        assert rp.framebuffers["fb2"]["texture"]["attached"] is True

    def test_unsupported_format_rejected(self):
        rp = RenderPipeline()
        assert rp.create_framebuffer("fb3", 256, 256, format="bogus128") is False
        assert "fb3" not in rp.framebuffers
        assert rp.create_framebuffer("fb4", 0, 256) is False


class TestPostProcessOrder:
    def test_bloom_after_tone_mapping(self):
        rp = RenderPipeline()
        applied = rp.apply_post_process(["bloom", "tone_mapping"])
        assert applied.index("tone_mapping") < applied.index("bloom")

    def test_skip_unneeded_effect(self):
        rp = RenderPipeline()
        applied = rp.apply_post_process(["bloom", "vignette", "fxaa"],
                                        skip=["bloom"])
        assert "bloom" not in applied
        assert "vignette" in applied


class TestShadowAllLights:
    def test_point_light_has_shadow(self):
        rp = RenderPipeline()
        assert rp.render_shadow_map("point") is True

    def test_spot_light_has_shadow(self):
        rp = RenderPipeline()
        assert rp.render_shadow_map("spot") is True

    def test_shadow_bias_and_resolution(self):
        rp = RenderPipeline()
        assert rp.render_shadow_map("directional", resolution=4096,
                                    bias=0.01) is True
        sm = rp.shadow_maps["directional"]
        assert sm["resolution"] == 4096
        assert sm["bias"] == 0.01


class TestFrustumCulling:
    def test_large_object_not_culled(self):
        rp = RenderPipeline()
        objects = [{"center": (-200, 0, 0), "bounds": (100, 100, 100)}]
        result = rp.frustum_culling(objects)
        assert len(result) == 1

    def test_object_outside_culled(self):
        rp = RenderPipeline()
        objects = [
            {"center": (0, 0, 0), "bounds": (1, 1, 1)},
            {"center": (-500, 0, 0), "bounds": (100, 100, 100)},
        ]
        result = rp.frustum_culling(objects)
        assert len(result) == 1
        assert result[0]["center"] == (0, 0, 0)
