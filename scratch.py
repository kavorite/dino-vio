from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import itertools as it
import torch
import PyNvVideoCodec as nvc
import torch.nn.functional as F

_yuv_to_rgb = torch.tensor(
    [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]],
    device="cuda",
)


def decode(frame_nv12):
    height = frame_nv12.shape[0] * 2 // 3
    width = frame_nv12.shape[1]
    y_plane = frame_nv12[:height, :].float()
    uv_plane_interleaved = frame_nv12[height:, :]
    uv_planes = uv_plane_interleaved.reshape(height // 2, width // 2, 2).float()
    u_plane_nchw = uv_planes[:, :, 0].unsqueeze(0).unsqueeze(0)
    v_plane_nchw = uv_planes[:, :, 1].unsqueeze(0).unsqueeze(0)

    u_upsampled_nchw = F.interpolate(
        u_plane_nchw, size=(height, width), mode="bilinear", align_corners=False
    )
    v_upsampled_nchw = F.interpolate(
        v_plane_nchw, size=(height, width), mode="bilinear", align_corners=False
    )

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

    for batch in it.batched(_nv12_frames(), 16):
        yield torch.vmap(decode)(torch.stack(batch))


model_slug = "depth-anything/Depth-Anything-V2-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_slug)
model = AutoModelForDepthEstimation.from_pretrained(model_slug)


def estimate_depth(images):
    inputs = image_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[images.shape[-2:]] * images.shape[0],
    )
    x_patches, y_patches = torch.tensor(inputs["pixel_values"][0].shape[1:]) // 14
    patch_features = outputs.hidden_states[-1][0, 1:, :]  # ignore cls token
    patch_features = patch_features.reshape(y_patches, x_patches, -1)
    depths = [output["predicted_depth"] for output in post_processed_output]
    return torch.stack(depths), patch_features


for images in frames():
    inputs = image_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        depth, patch_features = estimate_depth(images.moveaxis(-1, -3))

    pass
