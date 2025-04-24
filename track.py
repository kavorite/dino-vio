from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
import itertools as it
from einops import einsum, rearrange
import torch
import PyNvVideoCodec as nvc
from sea_raft.raft import RAFT
import torch.nn.functional as F
import json
import argparse

# from https://github.com/princeton-vl/SEA-RAFT/blob/main/config/eval/spring-S.json
with open("spring-S.json", "r") as f:
    data = json.load(f)
args = argparse.Namespace()
args_dict = args.__dict__
for key, value in data.items():
    args_dict[key] = value


flows = RAFT.from_pretrained(
    "MemorySlices/Tartan-C-T-TSKH-spring540x960-S", args=args
).to("cuda")
_yuv_to_rgb = torch.tensor(
    [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]],
    device="cuda",
)


def decode(frame_nv12, full_res=False):
    height = frame_nv12.shape[0] * 2 // 3
    width = frame_nv12.shape[1]
    y_plane = frame_nv12[:height, :].float()
    uv_plane_interleaved = frame_nv12[height:, :]
    uv_planes = uv_plane_interleaved.reshape(height // 2, width // 2, 2).float()
    u_plane_nchw = uv_planes[:, :, 0].unsqueeze(0).unsqueeze(0)
    v_plane_nchw = uv_planes[:, :, 1].unsqueeze(0).unsqueeze(0)

    u_upsampled_nchw = u_plane_nchw
    v_upsampled_nchw = v_plane_nchw

    if full_res:
        u_upsampled_nchw = F.interpolate(
            u_plane_nchw, size=(height, width), mode="bilinear", align_corners=False
        )
        v_upsampled_nchw = F.interpolate(
            v_plane_nchw, size=(height, width), mode="bilinear", align_corners=False
        )
    else:
        y_plane = (
            F.interpolate(
                y_plane.unsqueeze(0).unsqueeze(0),
                size=(height // 2, width // 2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        u_upsampled_nchw = u_plane_nchw
        v_upsampled_nchw = v_plane_nchw

    u_upsampled = u_upsampled_nchw.squeeze(0).squeeze(0)
    v_upsampled = v_upsampled_nchw.squeeze(0).squeeze(0)
    u_adj = u_upsampled - 128.0
    v_adj = v_upsampled - 128.0
    yuv_adj_hwc = torch.stack([y_plane, u_adj, v_adj], dim=-1)
    rgb_tensor_hwc = torch.matmul(yuv_adj_hwc, _yuv_to_rgb.T)
    return rgb_tensor_hwc.clamp(0, 255).byte()


def frames():
    demuxer = nvc.CreateDemuxer(filename="snail.mp4.mkv")
    decoder = nvc.CreateDecoder(usedevicememory=True)

    def _nv12_frames():
        for packet in demuxer:
            for frame_data in decoder.Decode(packet):
                yield torch.from_dlpack(frame_data)

    for batch in it.batched(_nv12_frames(), 8):
        yield torch.vmap(decode)(torch.stack(batch))


model_slug = "apple/DepthPro-hf"

depth = DepthProForDepthEstimation.from_pretrained(model_slug).to(
    device="cuda", dtype=torch.bfloat16
)
image_processor = DepthProImageProcessorFast.from_pretrained(model_slug)


def estimate_depth(images):
    # Manual preprocessing according to DepthPro config
    # https://huggingface.co/apple/DepthPro-hf/resolve/main/preprocessor_config.json

    # Original images are in NCHW format
    images = images.to(dtype=torch.float32)

    # 1. Rescale if needed (rescale_factor: 0.00392156862745098 = 1/255)
    if images.max() > 1.0:
        images = images * 0.00392156862745098  # rescale_factor

    # 2. Resize to target size (1536x1536)
    # Store original size for later
    orig_size = images.shape[-2:]
    target_size = (1536, 1536)

    # Only resize if necessary
    if orig_size != target_size:
        images = F.interpolate(
            images, size=target_size, mode="bilinear", align_corners=False
        )

    # 3. Normalize with mean=0.5, std=0.5
    images = (images - 0.5) / 0.5

    # Create the inputs dict to match the model's expected input format
    inputs = {"pixel_values": images}

    # Get depth prediction and focal length estimation
    outputs = depth(**inputs)
    outputs = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[images.shape[-2:]] * images.shape[0],
    )

    # Manual post-processing
    batch_size = images.shape[0]
    depths = []
    focal_lengths = []
    fields_of_view = []

    # Extract predicted depth maps, resize to original size if needed
    for i in range(batch_size):
        # Get depth prediction
        predicted_depth = outputs[i]["predicted_depth"]

        # Get camera parameters (FOV in degrees and focal length)
        fov = outputs[i]["field_of_view"].item()
        focal_length = outputs[i]["focal_length"].item()

        # Resize depth to original resolution if needed
        if orig_size != target_size:
            predicted_depth = (
                F.interpolate(
                    predicted_depth.unsqueeze(0).unsqueeze(0),
                    size=orig_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

        depths.append(predicted_depth)
        focal_lengths.append(focal_length)
        fields_of_view.append(fov)

    depths_tensor = torch.stack(depths)

    # Extra data for compatibility with existing code
    # (patch features for the previous model architecture)
    # This is a placeholder and might need adjustment
    patch_features = torch.zeros(
        (images.shape[0], target_size[0] // 16, target_size[1] // 16, 384),
        device=images.device,
    )

    return depths_tensor, patch_features, focal_lengths, fields_of_view


def estimate_delta(flow, intrinsics):
    # Avoid modifying the original flow tensor
    flow = flow.clone()
    # Create normalized pixel coordinates grid
    h, w = flow.shape[1:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )

    # Calculate inverse intrinsics
    K_inv = torch.inverse(intrinsics)

    # Create homogeneous pixel coordinates for frame 1
    ones = torch.ones_like(grid_x)
    # Shape [3, H*W]
    pixel_coords_hom = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1).float()

    # Create homogeneous pixel coordinates for frame 2 using flow
    # Shape [3, H*W]
    pixel_coords2_hom = (
        torch.stack([grid_x + flow[0], grid_y + flow[1], ones], dim=0)
        .reshape(3, -1)
        .float()
    )

    # Normalize coordinates using K_inv
    # Shape [3, N] where N = H*W
    pts1_norm = K_inv @ pixel_coords_hom
    pts2_norm = K_inv @ pixel_coords2_hom

    # Reshape to [N, 3] matrices for SVD
    pts1 = pts1_norm.T
    pts2 = pts2_norm.T

    # Build the constraint matrix efficiently using broadcasting
    pts1_expand = pts1.unsqueeze(2)  # [N, 3, 1]
    pts2_expand = pts2.unsqueeze(1)  # [N, 1, 3]
    A = (pts2_expand @ pts1_expand).reshape(-1, 9)  # [N, 9]

    # Solve for E using SVD
    _, _, V = torch.svd(A)
    E_vec = V[:, -1]  # The last column of V corresponds to the smallest singular value
    E = E_vec.reshape(3, 3)

    # Project E to essential matrix space (enforce rank 2 and equal singular values)
    U, S, Vt = torch.svd(E)
    # Force the singular values to be (1, 1, 0)
    S_enforced = torch.tensor([1.0, 1.0, 0.0], device=E.device)
    E_enforced = U @ torch.diag(S_enforced) @ Vt

    return E_enforced


def count_points_with_positive_depth(R, t, depth_map, intrinsics, flow):
    """Evaluate solution quality using reprojection error and depth consistency

    Instead of just counting points with positive depth, this function:
    1. Transforms points from frame 1 to frame 2 using candidate R,t
    2. Projects these points back to image plane 2
    3. Compares against the actual observed locations from optical flow
    4. Also checks that transformed points are in front of both cameras
    """
    h, w = depth_map.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=depth_map.device),
        torch.arange(w, device=depth_map.device),
        indexing="ij",
    )

    # Create homogeneous pixel coordinates
    ones = torch.ones_like(grid_x)
    pixel_coords = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1).float()

    # Flow vectors for all pixels
    flow_x = flow[0].reshape(-1)
    flow_y = flow[1].reshape(-1)

    # Pixel coordinates in second image according to flow
    pixel_coords2_from_flow = torch.stack(
        [grid_x.reshape(-1) + flow_x, grid_y.reshape(-1) + flow_y, ones.reshape(-1)],
        dim=0,
    )

    # Get 3D points in reference frame
    K_inv = torch.inverse(intrinsics)
    rays = K_inv @ pixel_coords  # Shape: [3, H*W]
    depths = depth_map.reshape(-1)  # Flatten depth

    # Normalize depth to reduce bias in absolute values
    valid_mask = depths > 0
    if valid_mask.sum() > 0:
        depths = depths / depths[valid_mask].mean()

    points3D = rays * depths.unsqueeze(0)  # Scale rays by depth

    # Transform points to second frame
    points3D_transformed = R @ points3D + t.unsqueeze(1)

    # Project transformed points to second image
    projected_points = intrinsics @ points3D_transformed
    # Avoid division by zero
    z_eps = 1e-10
    projected_points_2d = projected_points[:2, :] / (
        projected_points[2, :].unsqueeze(0) + z_eps
    )

    # Calculate reprojection error - distance between flow-predicted and transform-predicted locations
    reprojection_error = torch.sqrt(
        (projected_points_2d[0, :] - pixel_coords2_from_flow[0, :]) ** 2
        + (projected_points_2d[1, :] - pixel_coords2_from_flow[1, :]) ** 2
    )

    # Consider both depth consistency and reprojection error
    # Points should be in front of both cameras and have reasonable reprojection error
    consistent_points = (
        (depths > 0) & (points3D_transformed[2, :] > 0) & (reprojection_error < 5.0)
    )  # 5 pixel threshold

    # Return count as a quality metric
    return consistent_points.sum().item()


def select_correct_solution(
    solutions,
    depth_map=None,
    intrinsics=None,
    is_first_frame=False,
    flow=None,
):
    """
    Select the correct solution from the essential matrix decomposition:
    - Choose solution with most points having consistent geometry
    """
    best_solution = None
    max_consistent_points = -1

    for R, t in solutions:
        # Count points with consistent geometry
        consistent_count = count_points_with_positive_depth(
            R, t, depth_map, intrinsics, flow
        )

        if consistent_count > max_consistent_points:
            max_consistent_points = consistent_count
            best_solution = (R, t)

    return best_solution


def calculate_camera_movement(E, depth_map, intrinsics, flow, is_first_frame=False):
    """Calculate camera movement including out-of-frame regions"""

    # 1. Decompose E into R and t (unit vector)
    solutions = decompose_essential_matrix(E)

    # 2. Choose the correct solution without relying on previous poses
    R, t_unit = select_correct_solution(
        solutions,
        depth_map=depth_map,
        intrinsics=intrinsics,
        is_first_frame=is_first_frame,
        flow=flow,
    )

    # 3. Determine scale of translation using depth
    # Create point correspondences (image coords + depth)
    h, w = depth_map.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=depth_map.device),
        torch.arange(w, device=depth_map.device),
        indexing="ij",
    )

    # Points in image 1
    K_inv = torch.inverse(intrinsics)
    ones = torch.ones_like(grid_x)
    pixel_coords1 = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1).float()

    # Points in image 2 using flow
    pixel_coords2 = (
        torch.stack([grid_x + flow[0], grid_y + flow[1], ones], dim=0)
        .reshape(3, -1)
        .float()
    )

    # Normalize to get ray directions
    rays1 = K_inv @ pixel_coords1  # Shape: [3, H*W]
    rays2 = K_inv @ pixel_coords2  # Shape: [3, H*W]

    # Use depth to get 3D points in frame 1
    depths = depth_map.reshape(-1)  # Flatten depth
    points3D_1 = rays1 * depths.unsqueeze(0)  # Scale rays by depth

    # Estimate scale: For true corresponding points, if X2 = R*X1 + scale*t,
    # then rays2 should be parallel to X2 (in camera frame 2)
    # This means cross product should be zero
    X2_unscaled = R @ points3D_1 + t_unit.unsqueeze(1)  # Without scale

    # For each point, minimize ||cross(rays2, X2)||²
    cross_products = torch.cross(
        rays2.transpose(0, 1), X2_unscaled.transpose(0, 1), dim=1
    )

    # Solve for scale using least squares
    # We want scale such that: rays2 ≈ R @ (points3D_1) + scale * t_unit
    A = torch.cross(rays2.transpose(0, 1), t_unit.unsqueeze(1).transpose(0, 1), dim=1)
    b = -torch.cross(rays2.transpose(0, 1), (R @ points3D_1).transpose(0, 1), dim=1)

    # Least squares for scale (average over all points and dimensions)
    scale = (A * b).sum() / (A * A).sum()

    # Now we have fully scaled camera motion: R and scale*t_unit
    t = scale * t_unit

    return R, t, scale


def align_coordinates_with_old_depths(R, t, old_depth_map, intrinsics):
    """Align current frame coordinates with old depth map"""

    # Camera extrinsics matrix (from old to new)
    # [R | t]
    extrinsics = torch.zeros((3, 4), device=R.device)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    # Create pixel grid for current frame
    h, w = old_depth_map.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=R.device),
        torch.arange(w, device=R.device),
        indexing="ij",
    )

    # Get pixel locations with padding for out-of-frame motion
    # Add padding based on maximum expected motion
    pad = int(max(torch.norm(t).item() * max(intrinsics[0, 0], intrinsics[1, 1]), 100))

    # Extended grid
    h_ext, w_ext = h + 2 * pad, w + 2 * pad
    y_ext, x_ext = torch.meshgrid(
        torch.arange(-pad, h + pad, device=R.device),
        torch.arange(-pad, w + pad, device=R.device),
        indexing="ij",
    )

    # Create homogeneous coords for extended grid
    ones = torch.ones_like(x_ext)
    pixel_coords = torch.stack([x_ext, y_ext, ones], dim=0).reshape(3, -1).float()

    # Backproject to 3D rays (normalized directions)
    K_inv = torch.inverse(intrinsics)
    rays = K_inv @ pixel_coords  # Shape: [3, H_ext*W_ext]

    # For each ray in current extended frame, find where it intersects the
    # old frame's depth surface when transformed by [R|t]

    # We solve: λ * ray = R^T * (X1 - t)
    # where X1 is a point on the old depth surface

    # For each ray, we check the old image grid
    # and interpolate depth at intersection

    # Transform rays to old frame coordinates
    R_inv = R.transpose(0, 1)  # R^T = R^-1 for rotation matrices
    transformed_rays = R_inv @ rays

    # Compute intersection with z=1 plane in old frame
    # ray_z * λ = 1 => λ = 1/ray_z
    lambda_vals = 1.0 / transformed_rays[2, :]

    # Get (x,y) intersection with z=1 plane
    intersect_x = -R_inv[0, :] @ t + transformed_rays[0, :] * lambda_vals
    intersect_y = -R_inv[1, :] @ t + transformed_rays[1, :] * lambda_vals

    # Convert to pixel coordinates in old frame
    old_px_x = intersect_x * intrinsics[0, 0] + intrinsics[0, 2]
    old_px_y = intersect_y * intrinsics[1, 1] + intrinsics[1, 2]

    # Check which points are inside the old frame boundaries
    valid_points = (old_px_x >= 0) & (old_px_x < w) & (old_px_y >= 0) & (old_px_y < h)

    # For valid points, sample the old depth map
    # (using grid_sample for proper interpolation)
    depth_alignment_map = torch.zeros((h_ext, w_ext), device=R.device)

    # Normalize coordinates to [-1, 1] for grid_sample
    sample_points = torch.stack(
        [2 * old_px_x / (w - 1) - 1, 2 * old_px_y / (h - 1) - 1], dim=1
    )
    sample_points = sample_points.reshape(1, -1, 2)

    # Only use valid samples
    valid_samples = sample_points[:, valid_points, :]

    # Sample from old depth map
    old_depth_expanded = old_depth_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    sampled_depths = F.grid_sample(
        old_depth_expanded,
        valid_samples.reshape(1, -1, 1, 2),
        mode="bilinear",
        align_corners=True,
    )

    # Place sampled depths back into result map
    depth_alignment_map.reshape(-1)[valid_points] = sampled_depths.reshape(-1)

    # Return alignment map and validity mask
    return depth_alignment_map, valid_points.reshape(h_ext, w_ext)


def decompose_essential_matrix(E):
    """Decompose essential matrix into 4 possible rotation and translation pairs"""
    U, S, Vt = torch.svd(E)

    # Ensure proper rotation matrices by fixing determinants
    if torch.det(U) < 0:
        U = -U
    if torch.det(Vt) < 0:
        Vt = -Vt

    # Create the W matrix for decomposition
    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=E.device, dtype=E.dtype)

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Ensure these are proper rotation matrices (det=1)
    if torch.det(R1) < 0:
        R1 = -R1
    if torch.det(R2) < 0:
        R2 = -R2

    # Translation direction (unit vector)
    t = U[:, 2]

    # Four possible solutions: (R1,t), (R1,-t), (R2,t), (R2,-t)
    solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    return solutions


def main():
    # The intrinsics will be dynamically updated based on DepthPro's estimation
    # We'll initialize with a placeholder that will be updated
    intrinsics = None

    # State variables for tracking
    prev_frame = None  # Store previous frame
    prev_depth = None  # Store previous depth map
    frame_count = 0

    # Process video frames
    for batch in frames():
        # Process each frame in the batch individually
        for current_frame in batch:
            # Convert to normalized RGB
            current_frame_nchw = (
                rearrange(current_frame, "h w c -> c h w").float().unsqueeze(0)
            )

            # Calculate depth for current frame and get camera parameters
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                depth_map, _, focal_lengths, fovs = estimate_depth(current_frame_nchw)
            current_depth = depth_map[0]

            # Extract camera parameters from current frame
            # field_of_view is in degrees, convert to radians
            fov_rad = torch.deg2rad(torch.tensor(fovs[0], device="cuda"))
            focal_length = torch.tensor(focal_lengths[0], device="cuda")

            # Get image dimensions
            h, w = current_depth.shape

            # Calculate the focal length in pixels using the field of view
            # For a camera with horizontal FOV θ and image width W:
            # fx = W / (2 * tan(θ/2))
            fx = w / (2 * torch.tan(fov_rad / 2))
            # Assuming same vertical and horizontal FOV scaled by aspect ratio
            fy = h / (2 * torch.tan(fov_rad / 2 * h / w))

            # Image center (principal point)
            cx = w / 2
            cy = h / 2

            # Update intrinsics matrix
            intrinsics = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                device="cuda",
                dtype=torch.float32,
            )

            # First frame becomes the initial reference
            if frame_count == 0:
                prev_frame = current_frame_nchw
                prev_depth = current_depth
                frame_count += 1
                continue

            # Compute optical flow from previous frame to current frame
            flow_output = flows(prev_frame, current_frame_nchw)
            flow = flow_output["flow"][-1].squeeze(0)  # [2, H, W]

            # Estimate essential matrix and camera pose
            E = estimate_delta(flow, intrinsics)
            R, t, scale = calculate_camera_movement(
                E,
                prev_depth,
                intrinsics,
                flow,
                is_first_frame=False,
            )

            # Translational velocity is the translation tensor itself
            trans_velocity = t

            # Angular velocity can be represented by the rotation matrix directly
            angular_velocity = R

            print(
                f"Frame {frame_count}: Velocity Tensors\n"
                f"Translation: {trans_velocity}\n"
                f"Rotation: {angular_velocity}"
            )

            # Update previous frame and depth for next iteration
            prev_frame = current_frame_nchw
            prev_depth = current_depth

            frame_count += 1


if __name__ == "__main__":
    with torch.no_grad():
        main()
