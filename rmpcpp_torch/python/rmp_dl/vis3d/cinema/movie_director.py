from time import sleep
from typing import List, Optional
import open3d as o3d
from rmp_dl.vis3d.cinema.actor import ActorBase
from rmp_dl.vis3d.cinema.camera import CameraBase

class MovieDirector:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True)

        self.actors: List[ActorBase] = []
        self.camera: CameraBase = None # type: ignore -> exception is raised later on if it is None

        self.frame_count = 0
        
    def set_camera(self, camera: CameraBase):
        self.camera = camera

    def register_actor(self, actor: ActorBase):
        self.actors.append(actor)

    def set_initial_geometries(self, geometries: List[o3d.geometry.Geometry]):
        self._add_geometries(geometries)

    def _has_next_actor_update(self) -> bool:
        if any(actor.has_next_step() for actor in self.actors):
            self.frame_count += 1
            print(f"Frame count: {self.frame_count} \t\t", flush=True, end="\r")
            return True
        return False

    def go(self, save_path: Optional[str] = None):
        self._setup()
        while self._has_next_actor_update():
            self._update_actors()
            self._add_new_actor_geometries()
            self._update_camera()
            if save_path is not None:
                self._save_image(save_path)
            self._remove_old_actor_geometries()

    def _save_image(self, save_path):
        self.vis.capture_screen_image(f"{save_path}/frame_{self.frame_count:05}.png")

    def _update_camera(self):
        view: CameraBase.View = self.camera.get_view()
        vc = self.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        self.vis.poll_events()
        self.vis.update_renderer()

    def _update_actors(self) -> None:
        for actor in self.actors:
            actor.next_step()

    def _remove_old_actor_geometries(self) -> None:
        for actor in self.actors:
            self._remove_geometries(actor.get_geometries_to_remove())

    def _add_new_actor_geometries(self):
        for actor in self.actors:
            self._add_geometries(actor.get_geometries_to_add())

    def _add_geometries(self, geometries: List[o3d.geometry.Geometry]):
        for geometry in geometries:
            self.vis.add_geometry(geometry)

    def _remove_geometries(self, geometries: List[o3d.geometry.Geometry]):
        for geometry in geometries:
            self.vis.remove_geometry(geometry)

    def _setup(self):
        if self.camera is None:
            raise ValueError("Camera not set")



