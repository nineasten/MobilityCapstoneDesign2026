#!/usr/bin/env python3
"""Export NetVLAD and SuperPoint to ONNX, and verify against PyTorch.

This is a standalone, read-only preparation step for TensorRT conversion on
the Jetson Orin. It does NOT modify the live hloc_localization pipeline.

TensorRT engines are tied to the GPU architecture/compute capability they
were built on, so an engine built here (RTX3060, sm_86) cannot be used on
Orin (sm_87) -- only the ONNX files are portable. Run `trtexec` on the Orin
itself against these .onnx files (see tools/orin/README.md).

SuperPoint is exported as its "dense" backbone + heads only (everything up
to and including the NMS'd score map and the normalized dense descriptor
map). Keypoint extraction (torch.nonzero / top-k / border removal) and
descriptor sampling have data-dependent output shapes and are not
ONNX/TensorRT-friendly, so they stay as lightweight PyTorch/numpy
post-processing on the host, unchanged from the current pipeline.

Usage:
    python3 export_onnx_models.py [--max-keypoints 512] [--resize 800x500]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "external" / "hloc"))
sys.path.insert(0, str(REPO_ROOT / "external" / "hloc" / "third_party"))

OUTPUT_DIR = REPO_ROOT / "model_cache" / "onnx"


class NetVLADWrapper(torch.nn.Module):
    """Adapts the hloc NetVLAD model's dict-based interface to a plain tensor."""

    def __init__(self, netvlad):
        super().__init__()
        self.netvlad = netvlad

    def forward(self, image):
        return self.netvlad({"image": image})["global_descriptor"]


class DenseSuperPoint(torch.nn.Module):
    """SuperPoint backbone + heads, stopping before keypoint extraction."""

    def __init__(self, net, nms_radius: int):
        super().__init__()
        self.net = net
        self.nms_radius = nms_radius

    @staticmethod
    def _simple_nms(scores, nms_radius: int):
        # scores: (b, H, W). ONNX's MaxPool requires a 4D (N, C, H, W)
        # input, unlike PyTorch's eager max_pool2d which also accepts 3D.
        assert nms_radius >= 0

        def max_pool(x):
            return F.max_pool2d(
                x.unsqueeze(1),
                kernel_size=nms_radius * 2 + 1,
                stride=1,
                padding=nms_radius,
            ).squeeze(1)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def forward(self, image):
        net = self.net
        x = net.relu(net.conv1a(image))
        x = net.relu(net.conv1b(x))
        x = net.pool(x)
        x = net.relu(net.conv2a(x))
        x = net.relu(net.conv2b(x))
        x = net.pool(x)
        x = net.relu(net.conv3a(x))
        x = net.relu(net.conv3b(x))
        x = net.pool(x)
        x = net.relu(net.conv4a(x))
        x = net.relu(net.conv4b(x))

        cPa = net.relu(net.convPa(x))
        scores = net.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores_dense = self._simple_nms(scores, self.nms_radius)

        cDa = net.relu(net.convDa(x))
        descriptors_dense = net.convDb(cDa)
        descriptors_dense = F.normalize(descriptors_dense, p=2, dim=1)

        return scores_dense, descriptors_dense


def load_models(max_keypoints: int):
    from hloc import extract_features, extractors
    from hloc.utils.base_model import dynamic_load

    retrieval_conf = extract_features.confs["netvlad"]
    local_conf = {
        "name": "superpoint",
        "nms_radius": 3,
        "max_keypoints": max_keypoints,
    }

    netvlad_cls = dynamic_load(extractors, retrieval_conf["model"]["name"])
    netvlad = netvlad_cls(retrieval_conf["model"]).eval()

    superpoint_cls = dynamic_load(extractors, local_conf["name"])
    superpoint = superpoint_cls(local_conf).eval()

    return netvlad, superpoint, local_conf["nms_radius"]


def export_and_verify(model, dummy_input, output_path: Path, input_names, output_names, dynamic_axes, model_label: str):
    import onnx
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
    if isinstance(torch_out, dict):
        torch_out = tuple(torch_out.values())
    elif not isinstance(torch_out, tuple):
        torch_out = (torch_out,)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(output_path))

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: (t.numpy() if isinstance(t, torch.Tensor) else t)
        for name, t in zip(input_names, dummy_input if isinstance(dummy_input, tuple) else (dummy_input,))
    }
    ort_out = sess.run(output_names, ort_inputs)

    print(f"\n[{model_label}] exported to {output_path}")
    for name, t_out, o_out in zip(output_names, torch_out, ort_out):
        diff = np.abs(t_out.numpy() - o_out).max()
        print(f"  {name}: shape={tuple(o_out.shape)} max_abs_diff(torch vs onnxruntime)={diff:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--resize", default="800x500", help="WxH used for the dummy export input")
    args = parser.parse_args()

    width, height = (int(v) for v in args.resize.split("x"))

    netvlad, superpoint, nms_radius = load_models(args.max_keypoints)

    # NetVLAD expects a 3-channel image in [0, 1].
    netvlad_input = torch.rand(1, 3, height, width)
    export_and_verify(
        NetVLADWrapper(netvlad).eval(),
        netvlad_input,
        OUTPUT_DIR / "netvlad.onnx",
        input_names=["image"],
        output_names=["global_descriptor"],
        dynamic_axes={"image": {2: "height", 3: "width"}},
        model_label="NetVLAD",
    )

    # SuperPoint expects a 1-channel grayscale image in [0, 1].
    sp_input = torch.rand(1, 1, height, width)
    dense_sp = DenseSuperPoint(superpoint.net, nms_radius).eval()
    export_and_verify(
        dense_sp,
        sp_input,
        OUTPUT_DIR / "superpoint_dense.onnx",
        input_names=["image"],
        output_names=["scores_dense", "descriptors_dense"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "scores_dense": {1: "height", 2: "width"},
            "descriptors_dense": {2: "height8", 3: "width8"},
        },
        model_label="SuperPoint (dense)",
    )


if __name__ == "__main__":
    main()
