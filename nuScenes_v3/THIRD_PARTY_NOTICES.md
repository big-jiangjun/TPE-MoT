# Third-Party Notices

TPE-MoT is a research fork built on open-source components. This repository
retains original copyright and license notices in files derived from those
projects. Use and redistribution must comply with every upstream license.

| Component | Role in this repository | Notice |
| --- | --- | --- |
| UniDriveVLA | Initial VLA driving framework and training integration | Derived files retain the original Apache-2.0 headers. |
| MMDetection / MMCV / MMDetection3D | Detection framework, runners, operators, and utilities | Refer to the licenses of the installed or vendored OpenMMLab packages. |
| Qwen3-VL | Vision-language backbone | Obtain weights and comply with the upstream Qwen license. We do not redistribute Qwen weights. |
| VGGT-Omega | Frozen spatial encoder | Source is vendored under `third_party/vggt-omega/`; its included FAIR Noncommercial Research License applies. We do not redistribute VGGT-Omega weights. |
| nuScenes | Dataset and evaluation assets | Obtain data and follow the official nuScenes terms. No dataset assets are included. |

The TPE-MoT-specific work in this release comprises the temporal six-camera
video integration, action/occupancy-free MoT execution path, Omega prefix
fusion, Omega global-memory perception fusion, training audit, and release
entry points.
