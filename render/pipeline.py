"""渲染管线 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class DrawCall:
    did: str
    depth: float = 0.0
    transparent: bool = False
    material: str = "default"

class RenderPipeline:
    def __init__(self):
        self.draw_calls: list = []
        self.framebuffers: dict = {}

    def submit_draw_call(self, call: DrawCall) -> None:
        # bug1: 不按深度和材质排序
        self.draw_calls.append(call)

    def create_framebuffer(self, name: str, width: int, height: int) -> bool:
        # bug2: 不检查完整性，不绑定纹理
        self.framebuffers[name] = {"width": width, "height": height}
        return True

    def apply_post_process(self, effects: list) -> None:
        # bug3: 固定顺序，不可跳过
        pass

    def render_shadow_map(self, light_type: str) -> bool:
        # bug4: 只渲染方向光
        return light_type == "directional"

    def frustum_culling(self, objects: list) -> list:
        # bug5: 只检查中心点
        return [obj for obj in objects if obj.get("center", (0,0,0))[0] > -100]
