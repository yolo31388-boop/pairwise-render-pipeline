"""渲染管线 - 纯标准库模拟"""
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


@dataclass
class DrawCall:
    obj: object
    depth_write: bool
    depth_test: bool
    phase: str


class RenderPipeline:
    def __init__(self):
        self.objects: list[Object] = []
        # 最小阴影偏差（兜底），配合斜率缩放偏差使用
        self.shadow_bias = 0.001
        # 斜率缩放偏差系数（类似 polygon offset factor）
        self.slope_scale_bias = 2.0
        # 各向异性过滤最大倍数
        self.max_anisotropy = 16
        self.draw_calls: list[DrawCall] = []
        self.passes = {"opaque": [], "transparent": []}

    # ---------- 包围体 ----------

    def get_obb(self, obj: Object):
        """返回物体旋转后 4 个角点的世界坐标（OBB）。"""
        half_w = obj.size[0] / 2.0
        half_h = obj.size[1] / 2.0
        angle = math.radians(obj.rotation)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = []
        for lx, ly in ((-half_w, -half_h), (half_w, -half_h),
                       (half_w, half_h), (-half_w, half_h)):
            wx = obj.pos[0] + lx * cos_a - ly * sin_a
            wy = obj.pos[1] + lx * sin_a + ly * cos_a
            corners.append((wx, wy))
        return corners

    def get_aabb(self, obj: Object):
        """旋转后 AABB：对 OBB 角点重新取轴对齐包围盒。"""
        corners = self.get_obb(obj)
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), min(ys), max(xs), max(ys))

    def get_max_radius(self, obj: Object):
        """物体外接圆半径（半对角线），旋转不变，用于保守剔除。"""
        half_w = obj.size[0] / 2.0
        half_h = obj.size[1] / 2.0
        return math.hypot(half_w, half_h)

    # ---------- 视锥剔除 ----------

    def frustum_cull(self, camera, objects):
        visible = []
        for obj in objects:
            aabb = self.get_aabb(obj)
            radius = self.get_max_radius(obj)
            if self._in_frustum(aabb, camera) or \
                    self._sphere_in_frustum(obj.pos, radius, camera):
                visible.append(obj)
        return visible

    def _in_frustum(self, aabb, cam):
        x_overlap = aabb[2] > cam["left"] and aabb[0] < cam["right"]
        if "top" in cam and "bottom" in cam:
            y_overlap = aabb[3] > cam["bottom"] and aabb[1] < cam["top"]
            return x_overlap and y_overlap
        return x_overlap

    def _sphere_in_frustum(self, center, radius, cam):
        cx, cy = center[0], center[1]
        closest_x = min(max(cx, cam["left"]), cam["right"])
        dx = cx - closest_x
        if "top" in cam and "bottom" in cam:
            closest_y = min(max(cy, cam["bottom"]), cam["top"])
            dist_sq = dx * dx + (cy - closest_y) ** 2
        else:
            dist_sq = dx * dx
        return dist_sq <= radius * radius

    # ---------- 透明排序 ----------

    def _camera_pos(self, camera):
        if isinstance(camera, dict):
            return camera.get("pos", [0.0, 0.0])
        return list(camera[:2])

    def _distance_to_camera(self, obj: Object, camera):
        cam_pos = self._camera_pos(camera)
        return math.hypot(obj.pos[0] - cam_pos[0], obj.pos[1] - cam_pos[1])

    def sort_transparent(self, objects, camera):
        """透明物体按相机距离从远到近排序（painter's algorithm）。"""
        return sorted(objects,
                      key=lambda o: self._distance_to_camera(o, camera),
                      reverse=True)

    # ---------- 深度状态 ----------

    def should_write_depth(self, obj: Object):
        """不透明物体写深度；透明物体只做深度测试、不写深度。"""
        return not obj.transparent

    def depth_write_for(self, obj: Object):
        return self.should_write_depth(obj)

    def _build_draw_calls(self, phase, objects):
        return [DrawCall(obj=o,
                         depth_write=self.should_write_depth(o),
                         depth_test=True,
                         phase=phase) for o in objects]

    # ---------- 阴影 ----------

    def compute_slope(self, light_pos, obj: Object):
        """表面法线与光线方向夹角的 tan 值（斜率），光线掠射时趋于无穷。"""
        dx = light_pos[0] - obj.pos[0]
        dy = light_pos[1] - obj.pos[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            return 0.0
        lx, ly = dx / length, dy / length
        angle = math.radians(obj.rotation)
        nx, ny = -math.sin(angle), math.cos(angle)
        cos_theta = max(1e-4, abs(nx * lx + ny * ly))
        return math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)) / cos_theta

    def compute_shadow_bias(self, light_pos, obj: Object):
        """斜率缩放偏差，并用最小偏差兜底。"""
        slope_scaled = self.slope_scale_bias * self.compute_slope(light_pos, obj)
        return max(self.shadow_bias, slope_scaled)

    def calculate_shadow(self, light_pos, obj: Object):
        depth = abs(obj.pos[1] - light_pos[1])
        return depth - self.compute_shadow_bias(light_pos, obj)

    # ---------- Mipmap ----------

    @staticmethod
    def _as_pair(value):
        if isinstance(value, (list, tuple)):
            return float(value[0]), float(value[1])
        return float(value), float(value)

    def select_mip_level(self, texture_size, screen_size, anisotropy=1):
        """按纹素在屏幕上的覆盖率选 mip 层级；anisotropy>1 时按短轴选层。"""
        tex_w, tex_h = self._as_pair(texture_size)
        scr_w, scr_h = self._as_pair(screen_size)
        if scr_w <= 0 or scr_h <= 0 or tex_w <= 0 or tex_h <= 0:
            return 0
        rho_u = tex_w / scr_w
        rho_v = tex_h / scr_h
        max_level = int(math.floor(math.log2(max(tex_w, tex_h))))

        def level_for(rho):
            if rho <= 1.0:
                return 0
            return int(math.floor(math.log2(rho) + 0.5))

        if anisotropy and anisotropy > 1:
            # 各向异性：沿短轴密度选 mip，长轴靠多次采样，避免过度模糊/闪烁
            level = level_for(min(rho_u, rho_v))
        else:
            level = level_for(max(rho_u, rho_v))
        return max(0, min(level, max_level))

    # ---------- 合批 ----------

    def batch_objects(self, objects):
        """材质和纹理都相同才能合批，不同纹理拆成不同批次。"""
        batches = {}
        for obj in objects:
            key = (obj.material, obj.texture)
            batches.setdefault(key, []).append(obj)
        return batches

    # ---------- 整帧渲染 ----------

    def render(self, camera):
        opaque = [o for o in self.objects if not o.transparent]
        transparent = [o for o in self.objects if o.transparent]

        visible_opaque = self.frustum_cull(camera, opaque)
        sorted_trans = self.sort_transparent(transparent, camera)
        visible_trans = self.frustum_cull(camera, sorted_trans)

        self.passes = {
            "opaque": self._build_draw_calls("opaque", visible_opaque),
            "transparent": self._build_draw_calls("transparent", visible_trans),
        }
        self.draw_calls = self.passes["opaque"] + self.passes["transparent"]
        return len(visible_opaque) + len(visible_trans)
