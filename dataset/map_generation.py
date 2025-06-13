import torch
import matplotlib.pyplot as plt
import numpy as np

def map2d_generation(goal_list, obs_list, scale, resolution):
    # assert resolution%2 == 1
    map_tensor = torch.zeros((3, resolution, resolution), dtype=torch.float32)
    
    center_x = center_y = resolution//2
    map_tensor[0, center_x, center_y] = 1.0
    
    for (x, y) in goal_list:
        # Convert floating point coordinates to integer indices
        i = round(x/scale) + resolution//2
        j = round(y/scale) + resolution//2
        
        # Make sure the indices are within the bounds of the map
        if 0 <= i < resolution and 0 <= j < resolution:
            map_tensor[1, i, j] = 1.0
        
    
    for (x, y) in obs_list:
        # Convert floating point coordinates to integer indices
        i = round(x/scale) + resolution//2
        j = round(y/scale) + resolution//2
        
        # Make sure the indices are within the bounds of the map
        if 0 <= i < resolution and 0 <= j < resolution:
            map_tensor[2, i, j] = 1.0

    return map_tensor

def map2d_bilinear_generation(goal_list, obs_list, scale, resolution):
    # assert resolution%2 == 1
    N = resolution
    map_tensor = torch.zeros((3, resolution, resolution), dtype=torch.float32)
    
    center_x = center_y = resolution//2
    map_tensor[0, center_x, center_y] = 1.0
    
    
    for (x, y) in goal_list:
        # Get the integer parts
        x_int = round(x)
        y_int = round(y)

        # Get the fractional parts
        x_frac = abs(x - x_int)
        y_frac = abs(y - y_int)

        # Calculate the weights for the four neighboring pixels
        w00 = (1 - x_frac) * (1 - y_frac)
        w01 = (1 - x_frac) * y_frac
        w10 = x_frac * (1 - y_frac)
        w11 = x_frac * y_frac

        i = round(x*scale) + resolution//2
        j = round(y*scale) + resolution//2

        # Distribute the weights to the four neighboring pixels
        if 0 <= i < N and 0 <= j < N:
            map_tensor[1, i, j] += w00
        if 0 <= i < N and 0 <= j + 1 < N:
            map_tensor[1, i, j + 1] += w01
        if 0 <= i + 1 < N and 0 <= j < N:
            map_tensor[1, i + 1, j] += w10
        if 0 <= i + 1 < N and 0 <= j + 1 < N:
            map_tensor[1, i + 1, j + 1] += w11
        
    for (x, y) in obs_list:
        # Get the integer parts
        x_int = round(x)
        y_int = round(y)

        # Get the fractional parts
        x_frac = abs(x - x_int)
        y_frac = abs(y - y_int)

        # Calculate the weights for the four neighboring pixels
        w00 = (1 - x_frac) * (1 - y_frac)
        w01 = (1 - x_frac) * y_frac
        w10 = x_frac * (1 - y_frac)
        w11 = x_frac * y_frac

        i = round(x*scale) + resolution//2
        j = round(y*scale) + resolution//2

        # Distribute the weights to the four neighboring pixels
        if 0 <= i < N and 0 <= j < N:
            map_tensor[2, i, j] += w00
        if 0 <= i < N and 0 <= j + 1 < N:
            map_tensor[2, i, j + 1] += w01
        if 0 <= i + 1 < N and 0 <= j < N:
            map_tensor[2, i + 1, j] += w10
        if 0 <= i + 1 < N and 0 <= j + 1 < N:
            map_tensor[2, i + 1, j + 1] += w11

    map_tensor.clip_(0,1)
    return map_tensor


def map_visualize(ax, segmentation, num_classes):
    """
    Visualize the semantic segmentation result.

    Args:
    image (torch.Tensor): The input image tensor of shape (C, H, W).
    segmentation (torch.Tensor): The segmentation result tensor of shape (H, W).
    num_classes (int): The number of classes.
    alpha (float): The transparency of the segmentation overlay.

    Returns:
    None
    """
    # Generate a color map for the segmentation
    colors = plt.cm.get_cmap('jet', num_classes)
    # segmentation = segmentation[0] + segmentation[1]*2 + segmentation[2]*3

    # Create an RGB image to overlay the segmentation
    # segmentation_rgb = colors(segmentation.cpu().numpy() / 3)[..., :3]
    
    ax.imshow(segmentation.permute(1,2,0).cpu().numpy(), cmap='jet', vmin=0, vmax=num_classes)

def map2d_bilinear_generation2(goal_list, obs_list, scale, resolution):
    # assert resolution%2 == 1
    N = resolution
    dim = 2
    radius = 1
    map_tensor = np.zeros((3, resolution, resolution), dtype=np.float32)
    xx, yy = np.meshgrid(np.linspace(0, resolution-1, resolution), np.linspace(0, resolution-1, resolution), indexing='ij')

    map_coord = np.stack((xx,yy), axis=-1) - np.ones((1,1,2)) * (resolution//2) # (N, N, dim)

    center_x = center_y = resolution//2
    map_tensor[0, center_x, center_y] = 1.0

    map_coord = map_coord.reshape((-1, dim))

    # goal mapping
    if len(goal_list) != 0:
        goals = np.stack(goal_list, axis=0) * scale # (N_goals, dim)

        goal_rel_dist = map_coord[:, None] - goals[None] # (N*N, N_goals, 2)
        goal_rel_dist[goal_rel_dist>=radius] = 0
        goal_rel_dist[goal_rel_dist<=-radius] = 0
        goal_rel_dist = np.abs(goal_rel_dist)
        goal_rel_dist[goal_rel_dist>0] = radius - goal_rel_dist[goal_rel_dist>0]

        map_tensor[1,:,:] = (goal_rel_dist[:,:,0].reshape(N,N,-1) * goal_rel_dist[:,:,1].reshape(N,N,-1)).sum(-1)

    if len(obs_list) != 0:
        obs = np.stack(obs_list, axis=0) * scale # (N_goals, dim)

        obs_rel_dist = map_coord[:, None] - obs[None] # (N*N, N_goals, 2)
        obs_rel_dist[obs_rel_dist>=radius] = 0
        obs_rel_dist[obs_rel_dist<=-radius] = 0
        obs_rel_dist = np.abs(obs_rel_dist)
        obs_rel_dist[obs_rel_dist>0] = radius - obs_rel_dist[obs_rel_dist>0] 

        map_tensor[2,:,:] = (obs_rel_dist[:,:,0].reshape(N,N,-1) * obs_rel_dist[:,:,1].reshape(N,N,-1)).sum(-1)
    map_tensor = np.clip(map_tensor, 0, 1)

    # img = cv2.resize(map_tensor.transpose(1,2,0), (224,224))*255.0
    # cv2.imshow("img",img.astype(np.uint8))
    # cv2.waitKey(1)

    # img = cv2.resize(map_tensor.transpose(1,2,0), (224,224))
    # plt.imshow(img)
    # plt.show()


    # plt.show()
    return map_tensor
if __name__=="__main__":
    goal_lists = [(2.3, 3.7), (1.5, 1.2)]
    obs_lists = [(-2.3, -3.7), (-1.5, -1.2)]
    N = 15
    map_tensor = map2d_bilinear_generation(goal_lists, obs_lists, 0.5, N)

    fig, ax = plt.subplots(1,1,figsize=(7,7))
    map_visualize(ax, map_tensor, num_classes=3)
    plt.show()
    print()
