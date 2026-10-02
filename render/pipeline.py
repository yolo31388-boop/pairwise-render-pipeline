"""渲染管线"""
from dataclasses import dataclass

# 后处理效果的正确执行顺序：tone mapping 必须在 bloom 之前，
# 否则过曝区域先被 bloom 放大，tone mapping 再压缩会导致 bloom 溢出。
POST_PROCESS_ORDER = ("tone_mapping", "bloom", "vignette", "fxaa")

SUPPORTED_LIGHT_TYPES = ("directional", "point", "spot")

# 驱动支持的颜色附件格式，不在列表里的格式拒绝创建。
SUPPORTED_COLOR_FORMATS = ("rgba8", "rgba16f", "rgba32f", "depth24")

_BYTES_PER_PIXEL = {"rgba8": 4, "rgba16f": 8, "rgba32f": 16, "depth24": 4}


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
        self.shadow_maps: dict = {}
        self.last_effects: list = []
        self.shadow_bias: float = 0.005
        # 视锥平面 (法线, 偏移)，平面方程 normal·p + offset >= 0 为内侧。
        self.frustum_planes: list = [
            ((1.0, 0.0, 0.0), 100.0),
            ((-1.0, 0.0, 0.0), 10000.0),
            ((0.0, 1.0, 0.0), 10000.0),
            ((0.0, -1.0, 0.0), 10000.0),
            ((0.0, 0.0, 1.0), 10000.0),
            ((0.0, 0.0, -1.0), 10000.0),
        ]

    def submit_draw_call(self, call: DrawCall) -> None:
        self.draw_calls.append(call)
        # 不透明物体在前：按材质分组减少状态切换，组内按深度从前到后提交；
        # 透明物体在最后：同样按材质分组，组内按深度从后到前混合，
        # 避免透明物体被后提交的不透明物体错误遮挡。
        self.draw_calls.sort(key=lambda c: (
            1 if c.transparent else 0,
            c.material,
            -c.depth if c.transparent else c.depth,
        ))

    def create_framebuffer(self, name: str, width: int, height: int,
                           color_format: str = "rgba8") -> bool:
        if width <= 0 or height <= 0:
            return False
        if color_format not in SUPPORTED_COLOR_FORMATS:
            return False
        texture = {
            "width": width,
            "height": height,
            "format": color_format,
            # 纹理创建时清零初始化，避免后处理采样到未初始化内存导致花屏。
            "data": bytearray(width * height * _BYTES_PER_PIXEL[color_format]),
        }
        fb = {
            "width": width,
            "height": height,
            "format": color_format,
            "texture": texture,
        }
        fb["complete"] = self._check_framebuffer_complete(fb)
        if not fb["complete"]:
            return False
        self.framebuffers[name] = fb
        return True

    def _check_framebuffer_complete(self, fb: dict) -> bool:
        texture = fb.get("texture")
        if texture is None:
            return False
        if fb["width"] <= 0 or fb["height"] <= 0:
            return False
        if fb["format"] not in SUPPORTED_COLOR_FORMATS:
            return False
        expected = fb["width"] * fb["height"] * _BYTES_PER_PIXEL[fb["format"]]
        return len(texture["data"]) == expected

    def apply_post_process(self, effects: list) -> list:
        # 只执行请求的效果（未请求的跳过），并按管线的正确顺序重排。
        requested = set(effects)
        self.last_effects = [e for e in POST_PROCESS_ORDER if e in requested]
        return self.last_effects

    def render_shadow_map(self, light_type: str, resolution: int = None,
                          distance: float = 0.0) -> bool:
        if light_type not in SUPPORTED_LIGHT_TYPES:
            return False
        res = resolution or self._shadow_resolution_for(distance)
        self.shadow_maps[light_type] = {
            "resolution": res,
            # 深度偏移，防止阴影 acne（表面自我遮挡条纹）。
            "bias": self.shadow_bias,
        }
        return True

    def _shadow_resolution_for(self, distance: float) -> int:
        # 阴影贴图分辨率随距离提升，缓解远距离阴影锯齿。
        if distance > 500.0:
            return 4096
        if distance > 100.0:
            return 2048
        return 1024

    def frustum_culling(self, objects: list) -> list:
        return [obj for obj in objects if self._aabb_in_frustum(obj)]

    def _aabb_in_frustum(self, obj: dict) -> bool:
        center = obj.get("center", (0, 0, 0))
        bounds = obj.get("bounds", (0, 0, 0))
        mins = tuple(c - b for c, b in zip(center, bounds))
        maxs = tuple(c + b for c, b in zip(center, bounds))
        # 用包围盒(AABB)与视锥求交，而不是只测中心点：
        # 对每个平面取 AABB 在该法线方向上的最远顶点(positive vertex)，
        # 若最远顶点都在平面外侧，则整个包围盒在视锥外。
        for normal, offset in self.frustum_planes:
            px = maxs[0] if normal[0] >= 0 else mins[0]
            py = maxs[1] if normal[1] >= 0 else mins[1]
            pz = maxs[2] if normal[2] >= 0 else mins[2]
            if normal[0] * px + normal[1] * py + normal[2] * pz + offset < 0:
                return False
        return True
