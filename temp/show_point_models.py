import open3d as o3d
import os

def main():
    # Prompt user for OFF file path
    file_path = input("Please enter the path to the OFF file (Windows format is supported): ")
    file_path = os.path.normpath(file_path)  # Normalize path for cross-platform compatibility
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist!")
        return
    
    # Check if file is an OFF file
    if not file_path.lower().endswith('.off'):
        print(f"Error: File '{file_path}' is not an OFF file!")
        return
    
    try:
        # Read the mesh from OFF file
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        # Check if mesh has vertices
        if len(mesh.vertices) == 0:
            print("Error: No vertices found in the OFF file!")
            return
        
        # Prompt user for number of sampling points
        try:
            num_points = int(input("Please enter the number of sampling points (default: 10000): "))
            if num_points <= 0:
                raise ValueError
        except ValueError:
            print("Invalid input. Using default number of points: 10000")
            num_points = 10000
        
        # Sample point cloud from mesh using Poisson disk sampling
        # This increases point density while preserving model shape
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
        
        # If sampling fails, use original vertices
        if len(pcd.points) == 0:
            print("Warning: Poisson disk sampling failed. Using original vertices.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
        
        # Set all points to black (RGB: 0, 0, 0)
        pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(pcd.points))
        
        # Create visualization options
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Black and White Point Cloud")
        
        # Set background color to white
        opt = vis.get_render_option()
        opt.background_color = [1, 1, 1]  # White background
        opt.point_size = 2.0  # Set point size for better visibility
        
        # Add point cloud to visualization
        vis.add_geometry(pcd)
        
        # Run visualization loop
        vis.run()
        
        # Destroy window when done
        vis.destroy_window()
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    main()