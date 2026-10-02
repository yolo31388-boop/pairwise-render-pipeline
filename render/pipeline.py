"""渲染管线"""
from dataclasses import dataclass

SUPPORTED_FORMATS = {"rgba8", "rgb8", "rgba16f", "depth24"}
SHADOW_LIGHT_TYPES = {"directional", "point", "spot"}
# 正确的后处理执行顺序：tone mapping 必须在 bloom 之前
POST_PROCESS_ORDER = ["tone_mapping", "bloom", "vignette", "fxaa"]


@dataclass
class DrawCall:
    did: str
    depth: float = 0.0
    transparent: bool = False
    material: str = "default"


class RenderPipeline:
    MAX_FRAMEBUFFER_SIZE = 16384

    def __init__(self):
        self.draw_calls: list = []
        self.framebuffers: dict = {}
        self.shadow_maps: dict = {}
        self.applied_effects: list = []
        self.shadow_map_resolution = 2048
        self.shadow_bias = 0.005

    def submit_draw_call(self, call: DrawCall) -> None:
        self.draw_calls.append(call)
        self._sort_draw_calls()

    def _sort_draw_calls(self) -> None:
        opaque = [c for c in self.draw_calls if not c.transparent]
        transparent = [c for c in self.draw_calls if c.transparent]
        # 不透明物体：按材质分组（减少状态切换），同材质内按深度从前到后
        opaque.sort(key=lambda c: (c.material, c.depth))
        # 透明物体：在不透明之后渲染，按深度从后到前
        transparent.sort(key=lambda c: -c.depth)
        self.draw_calls = opaque + transparent

    def create_framebuffer(self, name: str, width: int, height: int,
                           format: str = "rgba8") -> bool:
        if (width <= 0 or height <= 0
                or width > self.MAX_FRAMEBUFFER_SIZE
                or height > self.MAX_FRAMEBUFFER_SIZE):
            return False
        if format not in SUPPORTED_FORMATS:
            return False
        fb = {
            "width": width,
            "height": height,
            "format": format,
            "texture": {
                "id": f"{name}_tex",
                "width": width,
                "height": height,
                "format": format,
                "attached": True,
            },
        }
        fb["complete"] = self._check_framebuffer_complete(fb)
        if not fb["complete"]:
            return False
        self.framebuffers[name] = fb
        return True

    @staticmethod
    def _check_framebuffer_complete(fb: dict) -> bool:
        tex = fb.get("texture")
        return bool(
            tex
            and tex.get("attached")
            and tex.get("width") == fb.get("width")
            and tex.get("height") == fb.get("height")
            and tex.get("format") in SUPPORTED_FORMATS
        )

    def apply_post_process(self, effects: list, skip: list = None) -> list:
        skip = set(skip or ())
        requested = [e for e in effects if e not in skip]
        # 按管线定义的正确顺序执行，tone_mapping 先于 bloom
        applied = [e for e in POST_PROCESS_ORDER if e in requested]
        applied += [e for e in requested if e not in POST_PROCESS_ORDER]
        self.applied_effects = applied
        return applied

    def render_shadow_map(self, light_type: str, resolution: int = None,
                          bias: float = None) -> bool:
        if light_type not in SHADOW_LIGHT_TYPES:
            return False
        self.shadow_maps[light_type] = {
            "resolution": resolution or self.shadow_map_resolution,
            "bias": bias if bias is not None else self.shadow_bias,
        }
        return True

    def frustum_culling(self, objects: list, frustum: list = None) -> list:
        planes = frustum or self._default_frustum()
        return [
            obj for obj in objects
            if self._aabb_in_frustum(
                obj.get("center", (0, 0, 0)),
                obj.get("bounds", (0, 0, 0)),
                planes,
            )
        ]

    @staticmethod
    def _default_frustum() -> list:
        r = 100.0
        return [
            ((1, 0, 0), r), ((-1, 0, 0), r),
            ((0, 1, 0), r), ((0, -1, 0), r),
            ((0, 0, 1), r), ((0, 0, -1), r),
        ]

    @staticmethod
    def _aabb_in_frustum(center: tuple, bounds: tuple, planes: list) -> bool:
        # AABB 与视锥求交：对每个平面取沿法线最远的正顶点，
        # 正顶点都在平面外侧则整个包围盒在视锥外
        for normal, d in planes:
            px = center[0] + (bounds[0] if normal[0] >= 0 else -bounds[0])
            py = center[1] + (bounds[1] if normal[1] >= 0 else -bounds[1])
            pz = center[2] + (bounds[2] if normal[2] >= 0 else -bounds[2])
            if normal[0] * px + normal[1] * py + normal[2] * pz + d < 0:
                return False
        return True
