"""渲染管线 - 带6个bug"""
from dataclasses import dataclass, field

@dataclass
class Object:
    pos: list
    size: list
    rotation: float = 0.0
    transparent: bool = False
    material: str = "default"
    texture: str = None

class RenderPipeline:
    def __init__(self):
        self.objects: list[Object] = []
        self.shadow_bias = 0.001  # bug4: 偏差不够

    def get_aabb(self, obj: Object):
        # bug1: 不考虑旋转
        half_w = obj.size[0] / 2
        half_h = obj.size[1] / 2
        return (obj.pos[0]-half_w, obj.pos[1]-half_h, obj.pos[0]+half_w, obj.pos[1]+half_h)

    def frustum_cull(self, camera, objects):
        # bug1: 用AABB不考虑旋转
        visible = []
        for obj in objects:
            aabb = self.get_aabb(obj)
            if self._in_frustum(aabb, camera):
                visible.append(obj)
        return visible

    def _in_frustum(self, aabb, cam):
        return aabb[2] > cam["left"] and aabb[0] < cam["right"]

    def sort_transparent(self, objects, camera):
        # bug2: 透明物体不按深度排序
        return objects

    def render(self, camera):
        opaque = [o for o in self.objects if not o.transparent]
        transparent = [o for o in self.objects if o.transparent]
        # bug3: 透明物体也开深度写入
        visible_opaque = self.frustum_cull(camera, opaque)
        visible_trans = self.frustum_cull(camera, self.sort_transparent(transparent, camera))
        return len(visible_opaque) + len(visible_trans)

    def calculate_shadow(self, light_pos, obj):
        # bug4: 偏差不够导致阴影痤疮
        depth = abs(obj.pos[1] - light_pos[1])
        return depth - self.shadow_bias

    def select_mip_level(self, texture_size, screen_size):
        # bug5: 固定层级
        return 0

    def batch_objects(self, objects):
        # bug6: 只看材质不看纹理
        batches = {}
        for obj in objects:
            key = obj.material
            batches.setdefault(key, []).append(obj)
        return batches
