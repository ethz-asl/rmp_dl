from typing import Callable, Dict, List, Optional

import numpy as np
from rmp_dl.vis3d.vis3d import KeyModifier, Plot3D
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class Plot3DStepped(Plot3D):
    def __init__(self, initial_idx=0, initial_modifier=0):
        """Class that supports easy stepping through geometries
        Use arrow keys to step through geometries, and spacebar to toggle a modifier that the step geometry 
        getters can use to change their geometry

        You can add data to the observation dictionary, which is a dict of 
        Dict[str, list[Dict]] where each list entry is a step geometry. The lists don't need to be of equal length.
        """
        super().__init__()

        # We have callbacks for step geometries, where we can use the left right arrow keys to step through them
        # and the up down arrow keys to step through a 2nd index (can be to toggle anything) (if the getter supports it)
        # Spacebar is used to toggle a modifier, which the step geometry getters can use to change their geometry
        # Generally the geometry getters show their output for only 1 modifier, and the spacebar can be used to toggle between them
        # The current step geometry index
        self.step_geometry_idx = initial_idx
        # 2nd counter for step geometry callbacks. Each callback can decide what to do with this
        self.step_geometry_idx2 = 0

        self.step_geometry_getters = []
        self.step_geometries = []
        self.step_geometry_modifier = initial_modifier  # Pressing spacebar will toggle this between [0, steps) (see variable below), this modifier is passed onto the callbacks
        self.step_geometry_modifier_steps = 0 # Incremented by calling 
        self.current_step_geometry_modifier_steps = 0 

        # We also have callbacks for global visualization changes (so 1d indexing, meant for some global geometry changes)
        self.global_geometry_getters = []
        self.global_geometries = []
        self.global_geometry_modifier = 0
        self.global_geometry_modifier_steps = 0

        self.observations: Dict[str, List[Dict]] = {}
        self.setup_callbacks()

    def update_step_geometries(self):
        for step_geometry in self.step_geometries:
            try:
                self.vis.remove_geometry(step_geometry)
            except: pass
        self.step_geometries = []
        for getter in self.step_geometry_getters:
            try:
                geometry = getter(self.observations, self.step_geometry_idx, self.step_geometry_modifier, self.step_geometry_idx2)
            except: 
                # The 2nd index is not always implemented/used
                geometry = getter(self.observations, self.step_geometry_idx, self.step_geometry_modifier)

            if not geometry: continue
            try:  # So the getters can also return an iterable of geometries
                for g in geometry:
                    self.step_geometries.append(g)
                    self.vis.add_geometry(g)
            except TypeError:
                self.step_geometries.append(geometry)
                self.vis.add_geometry(geometry)
            except: continue
    
    def update_global_geometries(self):
        for global_geometry in self.global_geometries:
            try:
                self.vis.remove_geometry(global_geometry)
            except: pass
        self.global_geometries = []
        for getter in self.global_geometry_getters:
            try:
                geometry = getter(self.observations, self.global_geometry_modifier)
            except: continue
            if not geometry: continue
            try:  # So the getters can also return an iterable of geometries
                for g in geometry:
                    self.global_geometries.append(g)
                    self.vis.add_geometry(g)
            except TypeError:
                self.global_geometries.append(geometry)
                self.vis.add_geometry(geometry)
            except: continue

    def setup_callbacks(self):
        def arrow_callback(action, mods, increment):
            print(action, increment, mods)
            if action == 0: return
            if mods == KeyModifier.SHIFT: increment *= 10  # Holding shift increases the jump with 10x
            elif mods == KeyModifier.ALT: increment *= 100  # Holding alt increases the jump with 100x

            self.step_geometry_idx += increment
            print(self.step_geometry_idx)
            self.update_step_geometries()

        # glfw key values see: https://www.glfw.org/docs/latest/group__keys.html 
        # Arrow keys
        self.register_callback(Plot3D.ARROW_RIGHT, lambda action, mods: arrow_callback(action, mods, 1)) 
        self.register_callback(Plot3D.ARROW_LEFT, lambda action, mods: arrow_callback(action, mods, -1))

        def up_down_arrow_callback(action, mods, increment):
            if action == 0: return
            if mods == KeyModifier.SHIFT: increment *= 10  # Holding shift increases the jump with 10x
            elif mods == KeyModifier.ALT: increment *= 100  # Holding alt increases the jump with 100x

            self.step_geometry_idx2 += increment
            print(self.step_geometry_idx2)
            self.update_step_geometries()

        self.register_callback(Plot3D.ARROW_UP, lambda action, mods: up_down_arrow_callback(action, mods, 1))
        self.register_callback(Plot3D.ARROW_DOWN, lambda action, mods: up_down_arrow_callback(action, mods, -1))

        # When pressing spacebar we toggle a modifier, which the step geometry getters can use to change their geometry
        def modifier_callback(mods):
            increment = 1 if mods == 0 else -1 # Increment if no modifiers (e.g. shift), decrement otherwise
            self.step_geometry_modifier = (self.step_geometry_modifier + increment) % (self.step_geometry_modifier_steps + 1)
            print(self.step_geometry_modifier)
            self.update_step_geometries()

        def num_callback(mods, val):
            self.step_geometry_modifier = val
            print(self.step_geometry_modifier)
            self.update_step_geometries()
        
        self.register_callback(Plot3D.SPACEBAR, lambda action, mods: modifier_callback(mods)) # Spacebar
        for i in range(9):
            self.register_callback(ord(str(i)), lambda action, mods, val=i: num_callback(mods, val))

        # We use m and n to cycle through the global geometries
        def global_modifier_callback(action, mods, increment):
            if action == 0: return
            if mods != 0: increment *= 10
            self.global_geometry_modifier += increment
            print(self.global_geometry_modifier)
            self.update_global_geometries()

        self.register_callback(ord('M'), lambda action, mods: global_modifier_callback(action, mods, 1))
        self.register_callback(ord('N'), lambda action, mods: global_modifier_callback(action, mods, -1))
    
    def increment_and_get_step_geometry_idx(self):
        self.current_step_geometry_modifier_steps += 1
        self.step_geometry_modifier_steps = max(self.step_geometry_modifier_steps, self.current_step_geometry_modifier_steps)
        return self.current_step_geometry_modifier_steps
        
    def register_step_geometry_change(self, getter):
        self.step_geometry_getters.append(getter)
        self.update_step_geometries()

    def register_global_geometry_change(self, getter):
        self.global_geometry_getters.append(getter)
        self.update_global_geometries()

    def increment_and_get_global_geometry_idx(self):
        self.global_geometry_modifier_steps += 1
        return self.global_geometry_modifier_steps