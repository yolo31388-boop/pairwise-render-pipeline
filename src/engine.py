"""渲染管线（纯标准库模拟）。"""
import math
from dataclasses import dataclass

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
        # 阴影：斜率缩放偏差 + 最小偏差兜底
        self.shadow_bias = 0.005        # 最小偏差（minBias）
        self.slope_scale_bias = 2.0     # 斜率缩放系数
        # 透明物体状态：只做深度测试，不写深度
        self.transparent_depth_write = False
        # 各向异性过滤最大采样数
        self.max_anisotropy = 16
        self.draw_calls: list = []

    def get_aabb(self, obj: Object):
        """旋转后的轴对齐包围盒（保守 AABB）。"""
        half_w = obj.size[0] / 2
        half_h = obj.size[1] / 2
        angle = math.radians(obj.rotation)
        cos_a = abs(math.cos(angle))
        sin_a = abs(math.sin(angle))
        rotated_half_w = cos_a * half_w + sin_a * half_h
        rotated_half_h = sin_a * half_w + cos_a * half_h
        return (obj.pos[0] - rotated_half_w,
                obj.pos[1] - rotated_half_h,
                obj.pos[0] + rotated_half_w,
                obj.pos[1] + rotated_half_h)

    def get_bounding_radius(self, obj: Object):
        """外接圆最大半径，旋转任意角度都成立。"""
        half_w = obj.size[0] / 2
        half_h = obj.size[1] / 2
        return math.hypot(half_w, half_h)

    def get_obb(self, obj: Object):
        """OBB：中心、半宽高、旋转角（弧度）。"""
        return {
            "center": (obj.pos[0], obj.pos[1]),
            "half_extents": (obj.size[0] / 2, obj.size[1] / 2),
            "rotation": math.radians(obj.rotation),
        }

    def frustum_cull(self, camera, objects):
        # 旋转后的 AABB 与最大半径包围球，任一相交即保留
        visible = []
        for obj in objects:
            aabb = self.get_aabb(obj)
            radius = self.get_bounding_radius(obj)
            if self._in_frustum(aabb, camera) or \
                    self._sphere_in_frustum(obj.pos, radius, camera):
                visible.append(obj)
        return visible

    def _in_frustum(self, aabb, cam):
        if not (aabb[2] > cam["left"] and aabb[0] < cam["right"]):
            return False
        if "top" in cam and "bottom" in cam:
            if not (aabb[3] > cam["bottom"] and aabb[1] < cam["top"]):
                return False
        return True

    def _sphere_in_frustum(self, center, radius, cam):
        """包围球视锥测试，处理包围盒跨在视锥平面上的情况。"""
        x, y = center[0], center[1]
        if not (x + radius > cam["left"] and x - radius < cam["right"]):
            return False
        if "top" in cam and "bottom" in cam:
            if not (y + radius > cam["bottom"] and y - radius < cam["top"]):
                return False
        return True

    def sort_transparent(self, objects, camera):
        """按相机距离从远到近排序（painter's algorithm）。"""
        cam_pos = camera.get("pos", (0, 0))
        def distance_sq(obj):
            dx = obj.pos[0] - cam_pos[0]
            dy = obj.pos[1] - cam_pos[1]
            return dx * dx + dy * dy
        return sorted(objects, key=distance_sq, reverse=True)

    def get_depth_state(self, obj: Object):
        """不透明物体：开深度测试+写深度；透明物体：只测试不写入。"""
        if obj.transparent:
            return {"depth_test": True, "depth_write": False}
        return {"depth_test": True, "depth_write": True}

    def render(self, camera):
        opaque = [o for o in self.objects if not o.transparent]
        transparent = [o for o in self.objects if o.transparent]

        visible_opaque = self.frustum_cull(camera, opaque)
        visible_trans = self.frustum_cull(
            camera, self.sort_transparent(transparent, camera))

        # 不透明先画（写深度），透明后画（远到近，只测试不写）
        self.draw_calls = []
        for obj in visible_opaque:
            self.draw_calls.append({"object": obj,
                                    "depth_test": True, "depth_write": True})
        for obj in visible_trans:
            self.draw_calls.append({"object": obj,
                                    "depth_test": True, "depth_write": False})
        return len(self.draw_calls)

    def calculate_shadow(self, light_pos, obj, normal=None):
        """斜率缩放偏差（slope-scale bias），并以最小偏差兜底。"""
        lx = light_pos[0] - obj.pos[0]
        ly = light_pos[1] - obj.pos[1]
        depth = math.hypot(lx, ly)

        if normal is None:
            # rotation=0 时表面法线朝向 +y
            angle = math.radians(obj.rotation)
            normal = (-math.sin(angle), math.cos(angle))
        light_dir = self._normalize((lx, ly))
        normal = self._normalize(normal)
        cos_angle = abs(normal[0] * light_dir[0] + normal[1] * light_dir[1])
        cos_angle = min(1.0, max(1e-6, cos_angle))
        # 光线与法线夹角的斜率：tan(acos(N·L))
        slope = math.sqrt(max(0.0, 1.0 - cos_angle * cos_angle)) / cos_angle

        bias = max(self.shadow_bias, self.slope_scale_bias * slope)
        return depth - bias

    @staticmethod
    def _normalize(vec):
        length = math.hypot(vec[0], vec[1])
        if length < 1e-12:
            return (0.0, 0.0)
        return (vec[0] / length, vec[1] / length)

    def select_mip_level(self, texture_size, screen_size, anisotropy=1):
        """按纹素在屏幕上的覆盖率选择 mipmap 层级，支持各向异性。

        screen_size 可传标量（各向同性），或 (屏幕x方向尺寸, 屏幕y方向尺寸)。
        """
        if isinstance(screen_size, (tuple, list)):
            screen_x = max(1.0, float(screen_size[0]))
            screen_y = max(1.0, float(screen_size[1]))
        else:
            screen_x = screen_y = max(1.0, float(screen_size))

        texel_footprint_x = texture_size / screen_x
        texel_footprint_y = texture_size / screen_y
        major = max(texel_footprint_x, texel_footprint_y)
        minor = min(texel_footprint_x, texel_footprint_y)

        # 各向异性：沿主轴多次采样，可用更精细的 mip 层级
        if anisotropy and anisotropy > 1 and minor > 0:
            samples = min(float(anisotropy), major / minor)
            samples = min(float(self.max_anisotropy), samples)
            effective_footprint = major / samples
        else:
            effective_footprint = major

        max_level = int(math.floor(math.log2(max(1, texture_size)) + 1e-9))
        if effective_footprint <= 1.0:
            return 0
        level = int(math.floor(math.log2(effective_footprint) + 0.5))
        return max(0, min(max_level, level))

    def batch_objects(self, objects):
        # 材质与纹理都相同才能合批
        batches = {}
        for obj in objects:
            key = (obj.material, obj.texture)
            batches.setdefault(key, []).append(obj)
        return batches
