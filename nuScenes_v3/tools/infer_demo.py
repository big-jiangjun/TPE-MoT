"""
Simple inference demo — runs 1 batch on 1 GPU to verify the model works.
Usage:
    python tools/infer_demo.py <config> <checkpoint>
"""
import argparse
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None

    # Build model & load checkpoint
    print('Building model...')
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print(f'Checkpoint meta: {checkpoint.get("meta", {}).get("epoch", "N/A")}')
    model = model.cuda().eval()

    # Build dataloader (only need 1 batch)
    cfg.data.test.test_mode = True
    cfg.data.test.pop('ann_file', None)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                                   dist=False, shuffle=False)

    # Run inference on first batch
    with torch.no_grad():
        batch = next(iter(data_loader))
        # Move tensors to cuda
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()
            elif isinstance(v, list):
                batch[k] = [x.cuda() if torch.is_tensor(x) else x for x in v]

        print(f'\nRunning inference on 1 sample...')
        result = model(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize()
        print('Done!\n')

    # Inspect results
    print(f'Result type: {type(result)}')
    print(f'Result length: {len(result)}')
    for i, r in enumerate(result):
        print(f'\n--- Sample {i} ---')
        ib = r.get('img_bbox', r)
        for k, v in ib.items():
            if torch.is_tensor(v):
                print(f'  {k}: shape={tuple(v.shape)}, device={v.device}, '
                      f'min={v.min().item():.4f}, max={v.max().item():.4f}')
            elif isinstance(v, list):
                print(f'  {k}: list len={len(v)}')
            elif isinstance(v, dict):
                print(f'  {k}: dict keys={list(v.keys())[:8]}')
            else:
                print(f'  {k}: {type(v).__name__}')

    # Summary
    ib = result[0].get('img_bbox', result[0])
    has_det = ib.get('boxes_3d') is not None
    has_map = ib.get('vectors') is not None
    has_traj = ib.get('final_planning') is not None

    print(f'\n=== Summary ===')
    print(f'Detection (boxes_3d): {"YES" if has_det else "NO"}')
    if has_det:
        print(f'  boxes_3d shape: {ib["boxes_3d"].shape}')
        print(f'  scores_3d range: [{ib["scores_3d"].min().item():.4f}, {ib["scores_3d"].max().item():.4f}]')
    print(f'Map (vectors):       {"YES" if has_map else "NO"}')
    print(f'Planning (traj):     {"YES" if has_traj else "NO (expected, action expert removed)"}')
    print(f'\nInference OK — model works on gpu8001!')


if __name__ == '__main__':
    main()
