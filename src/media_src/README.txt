*** Textures
The SVG files are opened in Inkscape and act as the background texture for the 
map.

The texture is automatically loaded by Blender but it does not get rendered 
into Gazebo.

If texture is modified save it as PNG to be loaded by Gazebo as a texture for
the DAE file describing the environment shape.

*** Relief
The relief is described by the .blend files. Modify it in Blender to match 
desired needs and then export it as a Collada .dae file.

*** Import into ROS:
Once modified the files need to be copied to the following folders for Gazebo
to load:

* the 2023_Competition_Driving_Surface_v0.2.png file needs to be moved to:
2023_competition/enph353/enph353_gazebo/models/full_town/materials/textures

* the Terrain_2023_v2.dae file needs to be moved to:
2023_competition/enph353/enph353_gazebo/models/full_town/meshes
