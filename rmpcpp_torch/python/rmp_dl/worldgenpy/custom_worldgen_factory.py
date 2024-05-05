import numpy as np
from rmp_dl.worldgenpy.custom_worldgen import CustomWorldgen
from rmp_dl.worldgenpy.worldgen_base import WorldgenSettings


class CustomWorldgenFactory:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def SingleWall():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([10.0, 10.0, 10.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        )
        worldgen = CustomWorldgen(settings)
        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 4.0]))
        worldgen.add_cube(np.array([5.0, 5.0, 2.0]), np.array([3.0, 3.0, 0.4]))
        return worldgen
    
    @staticmethod
    def OverhangingWall():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([10.0, 10.0, 10.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        )
        worldgen = CustomWorldgen(settings)
        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 4.0]))
        worldgen.add_cube(np.array([5.0, 5.0, 2.0]), np.array([3.0, 3.0, 0.4]))
        return worldgen


    @staticmethod
    def SimpleWorld():
        settings = WorldgenSettings(
            world_limits = (
                np.array([-5, -5, -5]),
                np.array([21, 21, 21])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 8.0, 1.0]))
        worldgen.add_sphere(np.array([5.0, 8.0, 9.0]), 3.0)
        worldgen.add_cube(np.array([5.0, 16.0, 5.0]), np.array([12.0, 1.0, 12.0]))
        worldgen.add_sphere(np.array([5.0, 3.0, 1.0]), 2.0)

        worldgen.add_sphere(np.array([21.0, 21.0, 21.0]), 0.2)

        worldgen.add_sphere(np.array([5.0, -5.0, -5.0]), 0.2)

        worldgen.set_start(np.array([5.0, 0.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 14.0, 15.0]))

        return worldgen
    
    @staticmethod
    def SingleWallEmpty():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([10.0, 10.0, 10.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([4.0, 5.0, 3.5]), np.array([1.0, 7.0, 4.0]))

        worldgen.set_start(np.array([5.0, 5.0, 5.0]))
        worldgen.set_goal(np.array([8.0, 8.0, 8.0]))

        return worldgen
    
    @staticmethod
    def SingleWallEmptyLargeWorld():
        settings = WorldgenSettings(
            world_limits = (
                np.array([-10.0, -10.0, -10.0]),
                np.array([20.0, 20.0, 20.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([4.0, 5.0, 3.5]), np.array([1.0, 12.0, 8.0]))

        worldgen.set_start(np.array([5.2, 5.0, 8.0]))
        worldgen.set_goal(np.array([0.0, 5.0, 0.0]))
        # worldgen.add_bounds_automatic()
        return worldgen
    
    

    @staticmethod
    def SingleCube():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([1.8, 1.8, 2.8])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([1.0, 1.0, 1.0]), np.array([0.40, 0.40, 0.4]))
        
        worldgen.set_start(np.array([0.9, 0.9, 0.3]))
        worldgen.set_goal(np.array([0.9, 0.9, 1.9]))

        # worldgen.add_bounds_automatic()

        return worldgen


    @staticmethod
    def BigWall():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 5.0, 5.0]), np.array([12.0, 8.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def HoleWallEasy():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 7.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    

    @staticmethod
    def HoleWall():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 8.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def HoleWallHarder():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 9.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def HoleWallHarder2():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 10.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def HoleWallHarder3():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 11.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen


    @staticmethod 
    def SnakeWalls():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([50.0, 10.0, 4.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([10.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([15.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([20.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([25.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([30.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([35.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([40.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([45.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))

        worldgen.set_start(np.array([2.0, 5.0, 2.0]))
        worldgen.set_goal(np.array([48.0, 5.0, 2.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    


    @staticmethod
    def HoleWallY():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 0.0, 5.0]), np.array([1.0, 14.0, 10.0]))

        worldgen.set_start(np.array([1.0, 5.0, 5.0]))
        worldgen.set_goal(np.array([7.5, 5.0, 5.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    @staticmethod 
    def SnakeWallsClose():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([50.0, 10.0, 4.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([10.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([15.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([20.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([25.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([30.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([35.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([40.0, 10.0, 2.0]), np.array([1.0, 16.0, 4.0]))
        worldgen.add_cube(np.array([45.0, 0.0, 2.0]), np.array([1.0, 16.0, 4.0]))

        worldgen.set_start(np.array([1.0, 5.0, 2.0]))
        worldgen.set_goal(np.array([7.5, 5.0, 2.0]))

        worldgen.add_bounds_automatic()

        return worldgen


    @staticmethod
    def BucketLike():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 2.0, 5.0]), np.array([12.0, 10.0, 1.0]))
        worldgen.add_cube(np.array([5.0, 7.0, 4.0]), np.array([12.0, 1.0, 3.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def Overhang():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 1.5, 5.0]), np.array([12.0, 10.0, 1.0]))
        worldgen.add_cube(np.array([5.0, 6.5, 5.2]), np.array([12.0, 1.0, 1.0]))

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 8.5]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def OverhangX():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 5.0, 1.5]), np.array([1.0, 12.0, 10.0]))
        worldgen.add_cube(np.array([5.2, 5.0, 6.5]), np.array([1.0, 12.0, 1.0]))

        worldgen.set_start(np.array([1.0, 5.0, 5.0]))
        worldgen.set_goal(np.array([8.5, 5.0, 5.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def OverhangY():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([1.5, 5.0, 5.0]), np.array([10.0, 1.0, 12.0]))
        worldgen.add_cube(np.array([6.5, 5.2, 5.0]), np.array([1.0, 1.0, 12.0]))

        worldgen.set_start(np.array([5.0, 1.0, 5.0]))
        worldgen.set_goal(np.array([5.0, 8.5, 5.0]))

        worldgen.add_bounds_automatic()

        return worldgen

    
    @staticmethod
    def SlitWall():
        settings = WorldgenSettings.from_yaml_general_config()

        worldgen = CustomWorldgen(settings)
        worldgen.add_cube(np.array([5.0, 0.0, 5.0]), np.array([10.0, 9.8, 1.0]))
        worldgen.add_cube(np.array([5.0, 10.0, 5.0]), np.array([10.0, 9.8, 1.0]))
        

        worldgen.set_start(np.array([5.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([5.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    
    @staticmethod
    def MultipleSlitWall():
        settings = WorldgenSettings(
            world_limits = (
                np.array([0.0, 0.0, 0.0]),
                np.array([20.0, 10.0, 10.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        x_locs = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        slit_widths = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        for x_loc, slit_width in zip(x_locs, slit_widths):
            worldgen.add_cube(np.array([x_loc, 0.0, 5.0]), np.array([2.0, 10.0 - slit_width, 1.0]))
            worldgen.add_cube(np.array([x_loc, 10.0, 5.0]), np.array([2.0, 10.0 - slit_width, 1.0])) 

        worldgen.set_start(np.array([1.0, 5.0, 1.0]))
        worldgen.set_goal(np.array([1.0, 5.0, 9.0]))

        worldgen.add_bounds_automatic()

        return worldgen
    

    @staticmethod
    def WallDensities():
        xmin = 0.0
        xmax = 20.0
        ymin = 0.0
        ymax = 10.0
        zmin = 0.0
        zmax = 10.0

        wall_height = 8.0
        wall_width = 0.8

        settings = WorldgenSettings(
            world_limits = (
                np.array([xmin, ymin, zmin]),
                np.array([xmax, ymax, zmax])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        
        worldgen.set_start(np.array([xmin + 1, 5.0, 6.5]))
        worldgen.set_goal(np.array([xmax - 1, 5.0, 6.5]))

        wall_gaps = [0.6, 0.5, 0.4, 0.3, 0.15, 0.075]
        wall_locs = np.linspace(xmin + 3, xmax - 3, len(wall_gaps))

        for wall_loc, wall_gap in zip(wall_locs, wall_gaps):
            for y_loc in np.arange(ymin, ymax, wall_width + 2 * wall_gap):
                worldgen.add_cube(np.array([wall_loc, y_loc, 0.0]), np.array([0.3, wall_width, 2 * wall_height]))

        worldgen.add_bounds_automatic()

        return worldgen


    @staticmethod
    def NarrowGaps():
        xmin = 0.0
        xmax = 5.0
        ymin = 0.0
        ymax = 10.0
        zmin = 0.0
        zmax = 10.0

        square_size = 0.8

        settings = WorldgenSettings(
            world_limits = (
                np.array([xmin, ymin, zmin]),
                np.array([xmax, ymax, zmax])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        
        worldgen.set_start(np.array([xmin + 1, 5.0, 5.0]))
        worldgen.set_goal(np.array([xmax - 1, 5.0, 5.0]))

        wall_gaps = [0.1]
        wall_locs = [3.0]

        for wall_loc, wall_gap in zip(wall_locs, wall_gaps):
            for y_loc in np.arange(ymin, ymax, square_size + 2 * wall_gap):
                for z_loc in np.arange(zmax - 3.0 + square_size / 2, zmin, - (square_size + 2 * wall_gap)):
                    worldgen.add_cube(np.array([wall_loc, y_loc, z_loc]), np.array([0.3, square_size, square_size]))

        worldgen.add_bounds_automatic()

        return worldgen



    @staticmethod
    def BigSpheres():
        settings = WorldgenSettings(
            world_limits = (
                np.array([-0.8, 0.0, 0.0]),
                np.array([30.0, 10.0, 10.0])
            ),
            voxel_size=0.2,
            voxel_truncation_distance_vox=4.0
        ) # type: ignore

        worldgen = CustomWorldgen(settings)
        worldgen.add_sphere(np.array([5.0, 4.0, 5.0]), 3.0)


        center = np.array([12.0, 5.0, 5.0])
        radius = 4.5
        sphere_radius = 4.0

        N = 50
        for i in range(N):
            angle = np.pi * 2 * i / N
            sphere_center = center + np.array([0.0, np.cos(angle) * radius, np.sin(angle) * radius])
            worldgen.add_sphere(sphere_center, sphere_radius)
        
        worldgen.set_start(np.array([1.0, 5.0, 5.0]))
        worldgen.set_goal(np.array([15.0, 5.0, 5.0]))


        worldgen.add_bounds_automatic()

        return worldgen