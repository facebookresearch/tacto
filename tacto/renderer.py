# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Set backend platform for OpenGL render (pyrender.OffscreenRenderer)
- Pyglet, the same engine that runs the pyrender viewer. This requires an active
  display manager, so you can’t run it on a headless server. This is the default option.
- OSMesa, a software renderer. require extra install OSMesa.
  (https://pyrender.readthedocs.io/en/latest/install/index.html#installing-osmesa)
- EGL, which allows for GPU-accelerated rendering without a display manager.
  Requires NVIDIA’s drivers.

The handle for EGL is egl (preferred, require NVIDIA driver),
The handle for OSMesa is osmesa.
Default is pyglet, which requires active window
"""

# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"

import logging

import cv2
import numpy as np
import pybullet as p
import pyrender
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False):
    r = R.from_euler(xyz, angles, degrees=degrees)

    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r.as_matrix()
    return pose


# def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0]):
#     q = p.getQuaternionFromEuler(angles)
#     r = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)

#     pose = np.eye(4)
#     pose[:3, 3] = translation
#     pose[:3, :3] = r
#     return pose


class Renderer:
    def __init__(self, width, height, background, config_path):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param config_path:
        """
        self._width = width
        self._height = height

        if background is not None:
            self.set_background(background)
        else:
            self._background_real = None

        logger.info("Loading configuration from: %s" % config_path)
        self.conf = OmegaConf.load(config_path)

        self.force_enabled = (
            self.conf.sensor.force is not None and self.conf.sensor.force.enable
        )

        if self.force_enabled:
            self.min_force = self.conf.sensor.force.range_force[0]
            self.max_force = self.conf.sensor.force.range_force[1]
            self.max_deformation = self.conf.sensor.force.max_deformation

        self.shadow_enabled = (
            "shadow" in self.conf.sensor.lights and self.conf.sensor.lights.shadow
        )

        self.spot_light_enabled = (
            "spot" in self.conf.sensor.lights and self.conf.sensor.lights.spot
        )

        self.flags_render = 0

        # enable flags for rendering
        if self.shadow_enabled:
            # Please use spotlight for rendering shadows
            # Reference: https://pyrender.readthedocs.io/en/latest/_modules/pyrender/light.html
            assert self.spot_light_enabled == True

            self.flags_render |= (
                pyrender.constants.RenderFlags.RGBA
                | pyrender.constants.RenderFlags.SHADOWS_SPOT
            )

        self._init_pyrender()

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def background(self):
        return self._background_real

    def _init_pyrender(self):
        """
        Initialize pyrender
        """
        # Create scene for pybullet sync
        self.scene = pyrender.Scene()

        # Create scene for rendering given depth image
        self.scene_depth = pyrender.Scene()

        """
        objects format:
            {obj_name: pyrender node}
        """
        self.object_nodes = {}
        self.current_object_nodes = {}

        self.current_light_nodes = []
        self.cam_light_ids = []

        self._init_gel()
        self._init_camera()
        self._init_light()

        self.r = pyrender.OffscreenRenderer(self.width, self.height)

        colors, depths = self.render(object_poses=None, noise=False, calibration=False)

        self.depth0 = depths
        self._background_sim = colors

    def _init_gel(self):
        """
        Add gel surface in the scene
        """
        # Create gel surface (flat/curve surface based on config file)
        gel_trimesh = self._generate_gel_trimesh()

        mesh_gel = pyrender.Mesh.from_trimesh(gel_trimesh, smooth=False)
        self.gel_pose0 = np.eye(4)
        self.gel_node = pyrender.Node(mesh=mesh_gel, matrix=self.gel_pose0)
        self.scene.add_node(self.gel_node)

        # Add extra gel node into scene_depth
        self.gel_node_depth = pyrender.Node(mesh=mesh_gel, matrix=self.gel_pose0)
        self.scene_depth.add_node(self.gel_node_depth)

    def _generate_gel_trimesh(self):

        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        X0, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        if hasattr(g, "mesh") and g.mesh is not None:
            gel_trimesh = trimesh.load(g.mesh)

            # scale up for clearer indentation
            matrix = np.eye(4)
            matrix[[0, 1, 2], [0, 1, 2]] = 1.02
            gel_trimesh = gel_trimesh.apply_transform(matrix)

        elif not g.curvature:
            # Flat gel surface
            gel_trimesh = trimesh.Trimesh(
                vertices=[
                    [X0, Y0 + W / 2, Z0 + H / 2],
                    [X0, Y0 + W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 + H / 2],
                ],
                faces=[[0, 1, 2], [2, 3, 0]],
            )
        else:
            # Curved gel surface
            N = g.countW
            M = int(N * H / W)
            R = g.R
            zrange = g.curvatureMax

            y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
            z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
            yy, zz = np.meshgrid(y, z)

            h = R - np.maximum(0, R ** 2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
            xx = X0 - zrange * h / h.max()

            gel_trimesh = self._generate_trimesh_from_depth(xx)

        return gel_trimesh

    def _generate_trimesh_from_depth(self, depth):
        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        _, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        N = depth.shape[1]
        M = depth.shape[0]

        # Create grid mesh
        vertices = []
        faces = []

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        # Vertex format: [x, y, z]
        vertices = np.zeros([N * M, 3])

        # Add x, y, z position to vertex
        vertices[:, 0] = depth.reshape([-1])
        vertices[:, 1] = yy.reshape([-1])
        vertices[:, 2] = zz.reshape([-1])

        # Create faces

        faces = np.zeros([(N - 1) * (M - 1) * 6], dtype=np.uint)

        # calculate id for each vertex: (i, j) => i * m + j
        xid = np.arange(N)
        yid = np.arange(M)
        yyid, xxid = np.meshgrid(xid, yid)
        ids = yyid[:-1, :-1].reshape([-1]) + xxid[:-1, :-1].reshape([-1]) * N

        # create upper triangle
        faces[::6] = ids  # (i, j)
        faces[1::6] = ids + N  # (i+1, j)
        faces[2::6] = ids + 1  # (i, j+1)

        # create lower triangle
        faces[3::6] = ids + 1  # (i, j+1)
        faces[4::6] = ids + N  # (i+1, j)
        faces[5::6] = ids + N + 1  # (i+1, j+1)

        faces = faces.reshape([-1, 3])
        gel_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return gel_trimesh

    def _init_camera(self):
        """
        Set up camera
        """

        self.camera_nodes = []
        self.camera_zero_poses = []

        conf_cam = self.conf.sensor.camera
        self.nb_cam = len(conf_cam)

        for i in range(self.nb_cam):
            cami = conf_cam[i]

            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(cami.yfov), znear=cami.znear,
            )
            camera_zero_pose = euler2matrix(
                angles=np.deg2rad(cami.orientation), translation=cami.position,
            )
            self.camera_zero_poses.append(camera_zero_pose)

            # Add camera node into scene
            camera_node = pyrender.Node(camera=camera, matrix=camera_zero_pose)
            self.scene.add_node(camera_node)
            self.camera_nodes.append(camera_node)

            # Add extra camera node into scene_depth
            self.camera_node_depth = pyrender.Node(
                camera=camera, matrix=camera_zero_pose
            )
            self.scene_depth.add_node(self.camera_node_depth)

            # Add corresponding light for rendering the camera
            self.cam_light_ids.append(list(cami.lightIDList))

    def _init_light(self):
        """
        Set up light
        """

        # Load light from config file
        light = self.conf.sensor.lights

        origin = np.array(light.origin)

        xyz = []
        if light.polar:
            # Apply polar coordinates
            thetas = light.xrtheta.thetas
            rs = light.xrtheta.rs
            xs = light.xrtheta.xs
            for i in range(len(thetas)):
                theta = np.pi / 180 * thetas[i]
                xyz.append([xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])
        else:
            # Apply cartesian coordinates
            xyz = np.array(light.xyz.coords)

        colors = np.array(light.colors)
        intensities = light.intensities

        # Save light nodes
        self.light_nodes = []
        self.light_poses0 = []

        for i in range(len(colors)):

            if not self.spot_light_enabled:
                # create pyrender.PointLight
                color = colors[i]
                light_pose_0 = euler2matrix(
                    angles=[0, 0, 0], translation=xyz[i] + origin
                )

                light = pyrender.PointLight(color=color, intensity=intensities[i])

            elif self.spot_light_enabled:
                # create pyrender.SpotLight
                color = colors[i]

                theta = np.pi / 180 * (thetas[i] - 90)
                tuning_angle = -np.pi / 16
                light_pose_0 = euler2matrix(
                    xyz="yzx",
                    angles=[0, tuning_angle, theta],
                    translation=xyz[i] + origin,
                )

                light = pyrender.SpotLight(
                    color=color,
                    intensity=intensities[i],
                    innerConeAngle=0,
                    outerConeAngle=np.pi / 3,
                )

            light_node = pyrender.Node(light=light, matrix=light_pose_0)

            self.scene.add_node(light_node)
            self.light_nodes.append(light_node)
            self.light_poses0.append(light_pose_0)
            self.current_light_nodes.append(light_node)

            # Add extra light node into scene_depth
            light_node_depth = pyrender.Node(light=light, matrix=light_pose_0)
            self.scene_depth.add_node(light_node_depth)

    def add_object(
        self, objTrimesh, obj_name, position=[0, 0, 0], orientation=[0, 0, 0]
    ):
        """
        Add object into the scene
        """

        mesh = pyrender.Mesh.from_trimesh(objTrimesh)
        pose = euler2matrix(angles=orientation, translation=position)
        obj_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(obj_node)

        self.object_nodes[obj_name] = obj_node
        self.current_object_nodes[obj_name] = obj_node

    def update_camera_pose(self, position, orientation):
        """
        Update sensor pose (including camera, lighting, and gel surface)
        """

        pose = euler2matrix(angles=orientation, translation=position)

        # Update camera
        for i in range(self.nb_cam):
            camera_pose = pose.dot(self.camera_zero_poses[i])
            self.camera_nodes[i].matrix = camera_pose

        # Update gel
        gel_pose = pose.dot(self.gel_pose0)
        self.gel_node.matrix = gel_pose

        # Update light
        for i in range(len(self.light_nodes)):
            light_pose = pose.dot(self.light_poses0[i])
            light_node = self.light_nodes[i]
            light_node.matrix = light_pose

    def update_object_pose(self, obj_name, position, orientation):
        """
        orientation: euler angles
        """

        node = self.object_nodes[obj_name]
        pose = euler2matrix(angles=orientation, translation=position)
        self.scene.set_pose(node, pose=pose)

    def update_light(self, lightIDList):
        """
        Update the light node based on lightIDList, remove the previous light
        """
        # Remove previous light nodes
        for node in self.current_light_nodes:
            self.scene.remove_node(node)

        # Add light nodes
        self.current_light_nodes = []
        for i in lightIDList:
            light_node = self.light_nodes[i]
            self.scene.add_node(light_node)
            self.current_light_nodes.append(light_node)

    def _add_noise(self, color):
        """
        Add Gaussian noise to the RGB image
        :param color:
        :return:
        """
        # Add noise to the RGB image
        mean = self.conf.sensor.noise.color.mean
        std = self.conf.sensor.noise.color.std

        if mean != 0 or std != 0:
            noise = np.random.normal(mean, std, color.shape)  # Gaussian noise
            color = np.clip(color + noise, 0, 255).astype(
                np.uint8
            )  # Add noise and clip

        return color

    def _calibrate(self, color, camera_index):
        """
        Calibrate simulation wrt real sensor by adding background
        :param color:
        :return:
        """

        # cv2.imshow("bg", self._background_real)
        # cv2.waitKey(0)
        #
        # cv2.imshow("image", color)
        # cv2.waitKey(0)

        # cv2.imshow("sim_bg", self._background_sim[0])
        # cv2.waitKey(0)

        if self._background_real is not None:
            # Simulated difference image, with scaling factor 0.5
            diff = (color.astype(np.float) - self._background_sim[camera_index]) * 0.5

            # Add low-pass filter to match real readings
            diff = cv2.GaussianBlur(diff, (7, 7), 0)

            # Combine the simulated difference image with real background image
            color = np.clip((diff[:, :, :3] + self._background_real), 0, 255).astype(
                np.uint8
            )

        return color

    def set_background(self, background):
        self._background_real = cv2.resize(background, (self._width, self._height))
        self._background_real = self._background_real[:, :, ::-1]
        return 0

    def adjust_with_force(
        self, camera_pos, camera_ori, normal_forces, object_poses,
    ):
        """
        Adjust object pose with normal force feedback
        The larger the normal force, the larger indentation
        Currently linear adjustment from force to shift distance
        It can be replaced by non-linear adjustment with calibration from real sensor
        """
        existing_obj_names = list(self.current_object_nodes.keys())
        for obj_name in existing_obj_names:
            # Remove object from scene if not in contact
            if obj_name not in normal_forces:
                self.scene.remove_node(self.current_object_nodes[obj_name])
                self.current_object_nodes.pop(obj_name)

        # Add/Update the objects' poses the scene if in contact
        for obj_name in normal_forces:
            if obj_name not in object_poses:
                continue
            obj_pos, objOri = object_poses[obj_name]

            # Add the object node to the scene
            if obj_name not in self.current_object_nodes:
                node = self.object_nodes[obj_name]
                self.scene.add_node(node)
                self.current_object_nodes[obj_name] = node

            if self.force_enabled:
                offset = -1.0
                if obj_name in normal_forces:
                    offset = (
                        min(self.max_force, normal_forces[obj_name]) / self.max_force
                    )

                # Calculate pose changes based on normal force
                camera_pos = np.array(camera_pos)
                obj_pos = np.array(obj_pos)

                direction = camera_pos - obj_pos
                direction = direction / (np.sum(direction ** 2) ** 0.5 + 1e-6)
                obj_pos = obj_pos + offset * self.max_deformation * direction

            self.update_object_pose(obj_name, obj_pos, objOri)

    def _post_process(self, color, depth, camera_index, noise=True, calibration=True):
        if calibration:
            color = self._calibrate(color, camera_index)
        if noise:
            color = self._add_noise(color)
        return color, depth

    def render(
        self, object_poses=None, normal_forces=None, noise=True, calibration=True
    ):
        """

        :param object_poses:
        :param normal_forces:
        :param noise:
        :return:
        """
        colors, depths = [], []

        for i in range(self.nb_cam):
            # Set the main camera node for rendering
            self.scene.main_camera_node = self.camera_nodes[i]

            # Set up corresponding lights (max: 8)
            self.update_light(self.cam_light_ids[i])

            # Adjust contact based on force
            if object_poses is not None and normal_forces is not None:
                # Get camera pose for adjusting object pose
                camera_pose = self.camera_nodes[i].matrix
                camera_pos = camera_pose[:3, 3].T
                camera_ori = R.from_matrix(camera_pose[:3, :3]).as_quat()

                self.adjust_with_force(
                    camera_pos, camera_ori, normal_forces, object_poses,
                )

            color, depth = self.r.render(self.scene, flags=self.flags_render)
            color, depth = self._post_process(color, depth, i, noise, calibration)

            colors.append(color)
            depths.append(depth)

        return colors, depths

    def render_from_depth(self, depth, noise=True, calibration=True, scale=1.0):
        """
        :param depth:
        :param scale:
        :return:
        """
        # Load config
        X0 = self.conf.sensor.gel.origin[0]
        width = self.conf.sensor.gel.width
        height = self.conf.sensor.gel.height

        # scale depth map
        depth = depth * scale

        # Auto-fit the canvas size
        h_resize = depth.shape[0]
        w_resize = int(h_resize * width / height)
        depth_resize = np.zeros([w_resize, h_resize])

        if w_resize <= depth.shape[1]:
            w_left = int((depth.shape[1] - w_resize) / 2)
            depth_resize = depth[:, w_left : w_left + w_resize]
        else:
            w_left = int((w_resize - depth.shape[1]) / 2)
            print("w_left", w_left, "depth shape", depth.shape)
            depth_resize[:, w_left : w_left + depth.shape[1]] = depth

        surf_trimesh = self._generate_trimesh_from_depth(X0 - depth_resize)
        mesh = pyrender.Mesh.from_trimesh(surf_trimesh, smooth=True)

        # Update depth node
        self.gel_node_depth.mesh = mesh

        color, depth = self.r.render(self.scene_depth)
        color, depth = self._post_process(color, depth, 0, noise, calibration)

        return color, depth
