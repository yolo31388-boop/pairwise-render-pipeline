"""渲染测试 - 8个失败"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import RenderPipeline, Object

class TestFrustumCull:
    def test_rotated_object_not_wrongly_culled(self):
        rp = RenderPipeline()
        obj = Object([100, 0], [10, 10], rotation=45)
        cam = {"left": 90, "right": 110}
        visible = rp.frustum_cull(cam, [obj])
        assert len(visible) == 1, "旋转物体被错误剔除"

class TestTransparentSort:
    def test_transparent_sorted_by_depth(self):
        rp = RenderPipeline()
        far = Object([0, 100], [1,1], transparent=True)
        near = Object([0, 10], [1,1], transparent=True)
        result = rp.sort_transparent([near, far], {"pos": [0,0]})
        assert result[0] is far, "透明物体未从远到近排序"

class TestShadow:
    def test_no_shadow_acne(self):
        rp = RenderPipeline()
        rp.shadow_bias = 0.05
        obj = Object([0, 10], [1, 1])
        depth = rp.calculate_shadow((0, 100), obj)
        assert depth > 0, "阴影痤疮"

class TestMipmap:
    def test_mip_level_by_screen_coverage(self):
        rp = RenderPipeline()
        level = rp.select_mip_level(1024, 32)
        assert level > 0, "mipmap层级未按覆盖率计算"

class TestBatching:
    def test_batch_by_material_and_texture(self):
        rp = RenderPipeline()
        o1 = Object([0,0],[1,1], material="m1", texture="t1")
        o2 = Object([0,0],[1,1], material="m1", texture="t2")
        batches = rp.batch_objects([o1, o2])
        assert len(batches) == 2, "不同纹理被错误合批"
