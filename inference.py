import argparse
import os
import sys
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict
import json
import subprocess
import tempfile
from pathlib import Path
import torch
import torch.utils.checkpoint
from torchvision.utils import make_grid, save_image
from accelerate.utils import  set_seed
from tqdm.auto import tqdm
import torch.nn.functional as F
from einops import rearrange
from rembg import remove, new_session
import pdb
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from mvdiffusion.models_unclip.unet_mv2d_condition import UNetMV2DConditionModel
from econdataset import SMPLDataset
from reconstruct import ReMesh
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
]
session = new_session(providers=providers)

weight_dtype = torch.float16
def tensor_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    # save_single_views: bool
    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: float
    validation_grid_nrow: int

    num_views: int
    enable_xformers_memory_efficient_attention: bool
    with_smpl: Optional[bool]
    
    recon_opt: Dict
    prompt: Optional[str] = None
    color_prompt: Optional[str] = None
    normal_prompt: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Multi-view I/O hooks (added so we can study / replace the diffusion
    # outputs that feed the mesh-carving stage).
    #
    # If `mv_dump_dir` is set, after the multiview diffusion runs we save the
    # post-rembg RGBA PNGs that are about to be passed to ReMesh.optimize_case
    # into  <mv_dump_dir>/<scene>/   as
    #   color_<view>_masked.png    # 6 colour views (RGBA, bg-removed)
    #   normals_<view>_masked.png  # 6 normal views (front_face has the close-
    #                              # up face composited into top-right quadrant)
    #   cond_input.png             # the conditioning input image fed to the
    #                              # multiview diffusion
    #   contact_sheet.png          # 7×2 grid: input | (color/normal × 6 views)
    # File names match the on-disk convention used by ReMesh.load_training_data.
    #
    # If `mv_inject_dir` is set, the diffusion pipeline is SKIPPED for each
    # scene whose <mv_inject_dir>/<scene>/ folder contains the 12 PNGs above.
    # Those images are loaded straight off disk (resized to crop_size if
    # needed), background-removed if alpha is missing, and passed to
    # ReMesh.optimize_case unchanged. This lets you swap in manual / morphed /
    # alternate multiview images and study how the carving + texture
    # projection responds, with no other code changes.
    # ------------------------------------------------------------------ #
    mv_dump_dir: Optional[str] = None
    mv_inject_dir: Optional[str] = None
    # Force back-view replacements
    force_back_image: Optional[str] = None  # path to back photo to force into 'back' color view
    force_back_normals_from_depthpro: bool = False
    flowier_python: Optional[str] = None  # python to run depthpro normals helper
    depthpro_normals_script: Optional[str] = None
    # DepthPro confidence-blend: replace low-confidence diffusion normals with
    # DepthPro-derived normals across all 6 views.  ``depthpro_normals_blend_thresh``
    # is the **norm-deviation threshold**: the decoded normal vector should be unit-
    # length; pixels where |1 - ||decode(n)|||  > thresh are degenerate and get
    # replaced by DepthPro.  Default 0.3 (catches near-black degenerate normals
    # such as [18,18,18] which decode to magnitude 1.49, while leaving valid
    # unit normals like (128,128,255) or (255,128,128) untouched).
    depthpro_normals_blend: bool = False
    depthpro_normals_blend_thresh: float = 0.3
    # Original PSHuman pastes the 7th generated close-up normal tile into the
    # upper-right quadrant of the front normal map.  This can corrupt shoulders
    # and elbows when the close-up tile is not registered to the full body.
    front_normal_face_patch: bool = True
    # Real-ESRGAN x4 super-resolution applied to the 6 colour + 6 normal views
    # between diffusion and ``ReMesh.preprocess``.  Runs in the seed venv via
    # a subprocess (PSHuman conda env doesn't have basicsr/realesrgan).
    # ReMesh.preprocess will then bilinear-resize to ``recon_opt.resolution``,
    # so raising both together (e.g. SR to 1536 and recon resolution 1536)
    # actually delivers extra texture detail.
    mv_color_upscale: bool = False
    mv_normal_upscale: bool = True   # gated by mv_color_upscale
    mv_upscale_final_size: int = 1536  # 0 = keep native SR size (4x)
    mv_upscale_python: Optional[str] = None
    mv_upscale_script: Optional[str] = None
    mv_upscale_ckpt: Optional[str] = None


# ── DepthPro normals blend helper ──────────────────────────────────────────
# Azimuths for views that are NOT horizontally flipped by PSHuman (j not in
# [3,4]).  For hflipped views (back=j3, left=j4) the image is already in a
# front-facing orientation, so we use azimuth=0 so DepthPro treats it as a
# front-facing camera and produces correct camera-space depth gradients.
_MV_VIEW_AZIMUTHS = {
    'front_face': 0.0,
    'front_right': 45.0,
    'right': 90.0,
    'back': 0.0,    # already hflipped by PSHuman
    'left': 0.0,    # already hflipped by PSHuman
    'front_left': 315.0,
}


def _blend_depthpro_normals(
    colors: List,
    normals: List,
    mv_views: List[str],
    cfg,
) -> None:
    """Run DepthPro on each colour view; blend into diffusion normals where
    the diffusion result is near-neutral (low confidence).

    Design:
      INVARIANT: pixels where diffusion normal is a valid unit vector
        (||decode(n)|| ≈ 1.0, deviation < thresh) are unchanged.
      INVARIANT: pixels with degenerate diffusion normals
        (||decode(n)|| >> 1 or << 1, e.g. near-black [18,18,18] → mag 1.49)
        are fully replaced by DepthPro.
      DOMAIN identity: a perfectly-unit diffusion normal map → unchanged.
      PROOF: w_keep = clip((thresh - (|mag-1| - 0.05)) / thresh, 0, 1).
        When norm_dev <= 0.05 (valid), w_keep=1 → keep diffusion.
        When norm_dev >= thresh+0.05 (degenerate), w_keep=0 → use DepthPro.
        Linear blend between.  QED.
    """
    import numpy as np

    flowier_py = getattr(cfg, 'flowier_python', None) or '/build/flowier/.venv/bin/python'
    dp_script  = getattr(cfg, 'depthpro_normals_script', None) or '/build/seed/scripts/depthpro_normals.py'
    thresh = float(getattr(cfg, 'depthpro_normals_blend_thresh', 0.3))

    if not os.path.exists(flowier_py) or not os.path.exists(dp_script):
        print(f'[depthpro-blend] skipped: py={flowier_py} exists={os.path.exists(flowier_py)} '
              f'script={dp_script} exists={os.path.exists(dp_script)}', flush=True)
        return

    total_replaced = 0
    with tempfile.TemporaryDirectory() as tdir:
        for vi, view in enumerate(mv_views):
            c_im = colors[vi]
            n_im = normals[vi]
            azimuth = _MV_VIEW_AZIMUTHS.get(view, 0.0)
            rgb_path  = os.path.join(tdir, f'dp_rgb_{view}.png')
            mask_path = os.path.join(tdir, f'dp_mask_{view}.png')
            out_path  = os.path.join(tdir, f'dp_normals_{view}.png')
            try:
                r, g, b, a = c_im.split()
                Image.merge('RGB', (r, g, b)).save(rgb_path)
                c_im.save(mask_path)
                cmd = [
                    flowier_py, dp_script,
                    '--image', rgb_path,
                    '--output', out_path,
                    '--mask', mask_path,
                    '--azimuth', str(azimuth),
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=180)
                if result.stderr:
                    print(result.stderr.decode(), flush=True)
            except Exception as e:
                print(f'[depthpro-blend] {view} depthpro failed: {e}', flush=True)
                continue

            if not os.path.exists(out_path):
                continue

            dp_nrm = Image.open(out_path).convert('RGBA')
            n_arr  = np.array(n_im.convert('RGBA'), dtype=np.float32)
            dp_arr = np.array(
                dp_nrm.resize(n_im.size, Image.BILINEAR), dtype=np.float32
            )

            # Norm-deviation confidence: decode normal to [-1,1] and check
            # how far its magnitude deviates from 1.0 (unit-vector constraint).
            # Valid normals have dev≈0; degenerate near-black ones have dev>>0.
            n_decoded = n_arr[:, :, :3] * (2.0 / 255.0) - 1.0
            n_mag     = np.linalg.norm(n_decoded, axis=-1)
            norm_dev  = np.abs(n_mag - 1.0)
            # w_keep ∈ [0,1]: 0 = fully replace with DepthPro, 1 = keep diffusion
            # Small grace band of 0.05 before blend starts.
            w_keep = np.clip(1.0 - (norm_dev - 0.05) / max(thresh, 1e-6), 0.0, 1.0)[:, :, np.newaxis]

            # Only blend subject pixels (alpha > 10 in both sources)
            subj = (n_arr[:, :, 3:] > 10) & (dp_arr[:, :, 3:] > 10)
            blended_rgb = w_keep * n_arr[:, :, :3] + (1.0 - w_keep) * dp_arr[:, :, :3]
            out_arr = n_arr.copy()
            out_arr[:, :, :3] = np.where(subj, blended_rgb, n_arr[:, :, :3])

            n_replaced = int(((w_keep[:, :, 0] < 0.5) & subj[:, :, 0]).sum())
            total_replaced += n_replaced
            print(f'[depthpro-blend] {view}: {n_replaced} px DepthPro-dominant '
                  f'(thresh={thresh})', flush=True)

            normals[vi] = Image.fromarray(
                out_arr.clip(0, 255).astype(np.uint8), 'RGBA'
            )

    print(f'[depthpro-blend] done — {total_replaced} px replaced across '
          f'{len(mv_views)} views', flush=True)


def _upscale_mv_views(
    colors: List,
    normals: List,
    mv_views: List[str],
    cfg,
) -> None:
    """Real-ESRGAN x4 SR on the colour (and optionally normal) views.

    Runs in seed's venv via subprocess. The PSHuman conda env doesn't have
    basicsr/realesrgan. Each PIL image is written to a temp dir, SR script
    upscales them in batch, results are loaded back in place of the
    originals. RGBA is preserved (alpha is upscaled via the base.py split/merge
    path inside the upscaler).
    """
    py = getattr(cfg, 'mv_upscale_python', None) or '/workspace/seed/.venv/bin/python3'
    script = getattr(cfg, 'mv_upscale_script', None) or '/workspace/seed/scripts/realesrgan_mv_views.py'
    ckpt = getattr(cfg, 'mv_upscale_ckpt', None)
    final_size = int(getattr(cfg, 'mv_upscale_final_size', 0) or 0)
    do_normals = bool(getattr(cfg, 'mv_normal_upscale', True))

    if not os.path.exists(py) or not os.path.exists(script):
        print(f'[mv-upscale] skipped: py={py} exists={os.path.exists(py)} '
              f'script={script} exists={os.path.exists(script)}', flush=True)
        return

    with tempfile.TemporaryDirectory() as tdir:
        in_dir  = os.path.join(tdir, 'in')
        out_dir = os.path.join(tdir, 'out')
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        manifest = []  # (kind, vi, name)
        for vi, view in enumerate(mv_views):
            cname = f'c_{vi:02d}_{view}.png'
            colors[vi].save(os.path.join(in_dir, cname))
            manifest.append(('color', vi, cname))
            if do_normals:
                nname = f'n_{vi:02d}_{view}.png'
                normals[vi].save(os.path.join(in_dir, nname))
                manifest.append(('normal', vi, nname))

        cmd = [
            py, script,
            '--output-dir', out_dir,
            '--final-size', str(final_size),
            '--device', 'cuda',
        ]
        if ckpt:
            cmd += ['--ckpt', ckpt]
        for _, _, name in manifest:
            cmd += ['--input', os.path.join(in_dir, name)]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            if result.stderr:
                sys.stdout.write(result.stderr.decode())
                sys.stdout.flush()
        except subprocess.CalledProcessError as e:
            print(f'[mv-upscale] failed: rc={e.returncode}', flush=True)
            if e.stderr:
                sys.stdout.write(e.stderr.decode())
            return

        for kind, vi, name in manifest:
            out_path = os.path.join(out_dir, name)
            if not os.path.exists(out_path):
                print(f'[mv-upscale] missing output: {name}', flush=True)
                continue
            sr = Image.open(out_path).convert('RGBA')
            if kind == 'color':
                colors[vi] = sr
            else:
                normals[vi] = sr

    fsz = final_size if final_size > 0 else '4x'
    print(f'[mv-upscale] done — {len(mv_views)} colour views'
          f"{' + normals' if do_normals else ''} -> {fsz}px", flush=True)


def convert_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def convert_to_pil(tensor):
    return Image.fromarray(convert_to_numpy(tensor))

def save_image(tensor, fp):
    ndarr = convert_to_numpy(tensor)
    # pdb.set_trace()
    save_image_numpy(ndarr, fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


# --------------------------------------------------------------------------- #
# Multi-view I/O helpers (dump / inject) — see TestConfig docstring.
# --------------------------------------------------------------------------- #
def _dump_mv_layers(dump_root: str, scene: str, view_names: List[str],
                     colors_pil: List, normals_pil: List, cond_tensor) -> None:
    """Persist the 6 colour + 6 normal RGBA PNGs (post-rembg) plus the cond
    image and a contact sheet, all into <dump_root>/<scene>/."""
    out_dir = os.path.join(dump_root, scene)
    os.makedirs(out_dir, exist_ok=True)
    for view, c_im, n_im in zip(view_names, colors_pil, normals_pil):
        # Save post-rembg RGBA layers (full canvas)
        c_path = os.path.join(out_dir, f"color_{view}_masked.png")
        n_path = os.path.join(out_dir, f"normals_{view}_masked.png")
        c_im.save(c_path)
        n_im.save(n_path)

        # Also dump the alpha masks as separate L images, same resolution
        try:
            c_a = c_im.split()[-1]
            n_a = n_im.split()[-1]
            c_a.save(os.path.join(out_dir, f"mask_color_{view}.png"))
            n_a.save(os.path.join(out_dir, f"mask_normals_{view}.png"))
        except Exception as _:
            pass
    # Cond input (no alpha, RGB tensor in [0,1])
    cond_pil = convert_to_pil(cond_tensor.detach().clamp(0, 1))
    cond_pil.save(os.path.join(out_dir, "cond_input.png"))

    # Contact sheet: row0 = cond | colors..., row1 = blank | normals...
    cell = colors_pil[0].size[0]
    sheet = Image.new("RGBA", (cell * (1 + len(view_names)), cell * 2), (0, 0, 0, 0))
    sheet.paste(cond_pil.convert("RGBA").resize((cell, cell)), (0, 0))
    for k, (c_im, n_im) in enumerate(zip(colors_pil, normals_pil)):
        sheet.paste(c_im.convert("RGBA").resize((cell, cell)),
                     ((k + 1) * cell, 0))
        sheet.paste(n_im.convert("RGBA").resize((cell, cell)),
                     ((k + 1) * cell, cell))
    sheet.save(os.path.join(out_dir, "contact_sheet.png"))
    print(f"[mv-dump] scene={scene}  wrote 12 layers + cond + sheet → {out_dir}")


def _try_load_injected_mv(inject_dir: str, view_names: List[str], crop_size: int):
    """Load 6 color + 6 normal RGBA PNGs from <inject_dir>/.

    Returns (colors_pil, normals_pil) lists (each PIL.RGBA at crop_size²) or
    None if any image is missing. If an image lacks alpha, rembg is run on it.
    """
    if not inject_dir or not os.path.isdir(inject_dir):
        return None
    colors_pil, normals_pil = [], []
    for view in view_names:
        cp = os.path.join(inject_dir, f"color_{view}_masked.png")
        np_ = os.path.join(inject_dir, f"normals_{view}_masked.png")
        if not (os.path.isfile(cp) and os.path.isfile(np_)):
            return None
        c_im = Image.open(cp).convert("RGBA").resize((crop_size, crop_size), Image.BILINEAR)
        n_im = Image.open(np_).convert("RGBA").resize((crop_size, crop_size), Image.BILINEAR)
        # Fallback alpha extraction if injected images are pure-RGB-with-bg.
        # Only rembg if alpha channel is fully opaque AND there's a non-white
        # background (heuristic: corner pixels not transparent).
        if all(c_im.getextrema()[3][k] == 255 for k in (0, 1)):
            c_im = remove(c_im.convert("RGB"), session=session)
        if all(n_im.getextrema()[3][k] == 255 for k in (0, 1)):
            n_im = remove(n_im.convert("RGB"), session=session)
        colors_pil.append(c_im)
        normals_pil.append(n_im)
    return colors_pil, normals_pil


def build_prompt_embeddings(pipeline, num_views: int, cfg: TestConfig) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not (cfg.prompt or cfg.color_prompt or cfg.normal_prompt):
        return None

    view_names_7 = ["front", "front_right", "right", "back", "left", "front_left", "face"]
    view_names_9 = ["front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left", "face"]
    if num_views == 7:
        view_names = view_names_7
    elif num_views == 9:
        view_names = view_names_9
    else:
        view_names = [f"view_{idx}" for idx in range(num_views)]

    default_color = "a rendering image of 3D human, {view} view, color map."
    default_normal = "a rendering image of 3D human, {view} view, normal map."
    color_template = cfg.color_prompt or cfg.prompt or default_color
    normal_template = cfg.normal_prompt or cfg.prompt or default_normal

    def _expand(template: str) -> List[str]:
        prompts = []
        for view in view_names:
            if "{view}" in template:
                prompts.append(template.format(view=view))
            else:
                prompts.append(f"{template}, {view} view")
        return prompts

    color_prompts = _expand(color_template)
    normal_prompts = _expand(normal_template)

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    device = pipeline.unet.device

    def _encode(prompts: List[str]) -> torch.Tensor:
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask
        else:
            attention_mask = None
        embeds = text_encoder(text_inputs.input_ids, attention_mask=attention_mask)[0]
        return embeds.detach().cpu()

    return _encode(normal_prompts), _encode(color_prompts)

def run_inference(dataloader, econdata, pipeline, carving, cfg: TestConfig,  save_dir):
    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.unet.device).manual_seed(cfg.seed)

    # View ordering used by ReMesh.load_training_data — keep in sync.
    MV_VIEWS = ['front_face', 'front_right', 'right', 'back', 'left', 'front_left']

    prompt_overrides = build_prompt_embeddings(pipeline, cfg.num_views, cfg)
    images_cond, pred_cat = [], defaultdict(list)
    for case_id, batch in tqdm(enumerate(dataloader)):
        images_cond.append(batch['imgs_in'][:, 0])
        scene = batch['filename'][0].split('.')[0] if 'filename' in batch else f"case_{case_id:03d}"

        # ---------------- MV-INJECT: skip diffusion, load from disk ----------------
        inject_loaded = None
        if cfg.mv_inject_dir:
            inject_dir = os.path.join(cfg.mv_inject_dir, scene)
            inject_loaded = _try_load_injected_mv(inject_dir, MV_VIEWS, cfg.validation_dataset.crop_size)
            if inject_loaded is not None:
                colors_inj, normals_inj = inject_loaded
                print(f"[mv-inject] scene={scene}  loaded 6+6 RGBA from {inject_dir}")
                pose = econdata.__getitem__(case_id)
                carving.optimize_case(scene, pose, colors_inj, normals_inj)
                torch.cuda.empty_cache()
                continue
            else:
                print(f"[mv-inject] scene={scene}  NO usable images at {inject_dir} — running diffusion as fallback")

        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")# (B*Nv, 3, H, W)
        if cfg.with_smpl:
            smpl_in = torch.cat([batch['smpl_imgs_in']]*2, dim=0)
            smpl_in = rearrange(smpl_in, "B Nv C H W -> (B Nv) C H W")
        else:
            smpl_in = None

        if prompt_overrides is not None:
            normal_prompt_embeddings, clr_prompt_embeddings = prompt_overrides
            normal_prompt_embeddings = normal_prompt_embeddings.unsqueeze(0).repeat(batch['imgs_in'].shape[0], 1, 1, 1)
            clr_prompt_embeddings = clr_prompt_embeddings.unsqueeze(0).repeat(batch['imgs_in'].shape[0], 1, 1, 1)
        else:
            normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'], batch['color_prompt_embeddings']
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")

        with torch.autocast("cuda"):
            # B*Nv images
            guidance_scale = cfg.validation_guidance_scales
            unet_out = pipeline(
                imgs_in, None, prompt_embeds=prompt_embeddings,
                dino_feature=None, smpl_in=smpl_in,
                generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, 
                **cfg.pipe_validation_kwargs
            )
            
            out = unet_out.images
            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:] 
            if cfg.save_mode == 'concat': ## save concatenated color and normal---------------------
                pred_cat[f"cfg{guidance_scale:.1f}"].append(torch.cat([normals_pred, images_pred], dim=-1)) # b, 3, h, w
                cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}-seed{cfg.seed}-smpl-{cfg.with_smpl}")
                os.makedirs(cur_dir, exist_ok=True)
                for i in range(bsz//num_views):
                    scene =  batch['filename'][i].split('.')[0]

                    img_in_ = images_cond[-1][i].to(out.device)
                    vis_ = [img_in_]
                    for j in range(num_views):
                        idx = i*num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]
                        
                        vis_.append(color)
                        vis_.append(normal)

                    out_filename = f"{cur_dir}/{scene}.png"
                    vis_ = torch.stack(vis_, dim=0)
                    vis_ = make_grid(vis_, nrow=len(vis_), padding=0, value_range=(0, 1))
                    save_image(vis_, out_filename)
            elif cfg.save_mode in ('rgb', 'rgba'):
                for i in range(bsz//num_views):
                    scene =  batch['filename'][i].split('.')[0]

                    img_in_ = images_cond[-1][i].to(out.device)
                    normals, colors = [], []
                    for j in range(num_views):
                        idx = i*num_views + j
                        normal = normals_pred[idx]
                        if j == 0:
                            color = imgs_in[0].to(out.device)
                        else:
                            color = images_pred[idx]
                        if j in [3, 4]:
                            normal = torch.flip(normal, dims=[2])
                            color = torch.flip(color, dims=[2])
                            
                        colors.append(color)
                        if j == 6:
                            normal = F.interpolate(normal.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                        normals.append(normal)
                        
                        ## save color and normal---------------------
                        # normal_filename = f"normals_{view}_masked.png"
                        # rgb_filename = f"color_{view}_masked.png"
                        # save_image(normal, os.path.join(scene_dir, normal_filename))
                        # save_image(color, os.path.join(scene_dir, rgb_filename))
                    if cfg.front_normal_face_patch:
                        normals[0][:, :256, 256:512] = normals[-1]
                    
                    colors = [remove(convert_to_pil(tensor), session=session) for tensor in colors[:6]]
                    normals = [remove(convert_to_pil(tensor), session=session) for tensor in normals[:6]]

                    # Optional forced back-view replacements
                    back_idx = MV_VIEWS.index('back')
                    if cfg.force_back_image:
                        try:
                            forced = Image.open(cfg.force_back_image).convert('RGBA')
                            # Remove background if no/misleading alpha
                            if forced.getbands()[-1] != 'A' or forced.getextrema()[-1] == (255, 255):
                                forced = remove(forced.convert('RGB'), session=session)
                            # Preprocess using the SAME logic as load_image() in testdata_with_smpl.py:
                            #   1. Tight bbox crop around subject alpha
                            #   2. Scale so max(h,w) = crop_size  (aspect-ratio preserving)
                            #   3. Center-pad to crop_size × crop_size
                            # This is exactly what PSHuman does with the front input photo, so
                            # the back subject lands at the same scale on the same canvas.
                            # No comparison to the diffusion-generated back view is needed or wanted.
                            import numpy as np
                            crop_size = cfg.validation_dataset.crop_size
                            # Final canvas must match the diffusion output size (image_size, not crop_size)
                            # load_image() in testdata_with_smpl.py does: scale to crop_size, then
                            # add_margin(size=image_size).  Use the front view's actual pixel size.
                            image_size = colors[0].size[0]
                            alpha_np = np.asarray(forced)[:, :, 3]
                            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
                            if len(coords):
                                min_x, min_y = np.min(coords, 0)
                                max_x, max_y = np.max(coords, 0)
                                forced = forced.crop((int(min_x), int(min_y), int(max_x), int(max_y)))
                            h, w = forced.height, forced.width
                            scale = crop_size / max(h, w)
                            forced = forced.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
                            canvas = Image.new('RGBA', (image_size, image_size), (0, 0, 0, 0))
                            canvas.paste(forced, ((image_size - forced.width) // 2, (image_size - forced.height) // 2))
                            colors[back_idx] = canvas
                            print(f"[mv-force-back] back photo → tight-crop({w}x{h}) scale={scale:.4f} "
                                  f"→ {forced.width}x{forced.height} → padded {image_size}x{image_size} (image_size={image_size} crop_size={crop_size})")
                            # Optionally compute normals from depthpro on the forced back
                            if cfg.force_back_normals_from_depthpro:
                                py = cfg.flowier_python or "/build/flowier/.venv/bin/python"
                                script = cfg.depthpro_normals_script or "/build/seed/scripts/depthpro_normals.py"
                                with tempfile.TemporaryDirectory() as tdir:
                                    rgb_path = os.path.join(tdir, 'back_rgb.png')
                                    mask_path = os.path.join(tdir, 'back_mask.png')
                                    out_path = os.path.join(tdir, 'back_normals.png')
                                    # Save RGB and mask
                                    r, g, b, a = canvas.split()
                                    Image.merge('RGB', (r, g, b)).save(rgb_path)
                                    Image.merge('RGBA', (r, g, b, a)).save(mask_path)
                                    cmd = [py, script, '--image', rgb_path, '--output', out_path, '--mask', mask_path]
                                    try:
                                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                        normals[back_idx] = Image.open(out_path).convert('RGBA')
                                    except subprocess.CalledProcessError as e:
                                        print(f"[mv-force-back] depthpro normals failed: {e}")
                        except Exception as e:
                            print(f"[mv-force-back] failed to replace back view: {e}")

                    # ── DepthPro confidence-blend on diffusion normals ────────
                    if cfg.depthpro_normals_blend:
                        _blend_depthpro_normals(colors, normals, MV_VIEWS, cfg)

                    # ── Real-ESRGAN x4 SR on colour (and optionally normal) views ─
                    if cfg.mv_color_upscale:
                        _upscale_mv_views(colors, normals, MV_VIEWS, cfg)

                    # ---------------- MV-DUMP: save the carving inputs ----------
                    if cfg.mv_dump_dir:
                        _dump_mv_layers(cfg.mv_dump_dir, scene, MV_VIEWS,
                                         colors, normals, img_in_)
        pose = econdata.__getitem__(case_id)
        carving.optimize_case(scene, pose, colors, normals)
        torch.cuda.empty_cache()   
               
     

def load_pshuman_pipeline(cfg):
    unet = UNetMV2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        unet=unet,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, 'enable_vae_tiling'):
        pipeline.enable_vae_tiling()
    if torch.cuda.is_available():
        pipeline.to('cuda')
    return pipeline

def main(
    cfg: TestConfig
):

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)
    pipeline = load_pshuman_pipeline(cfg)
    

    if cfg.with_smpl:
        from mvdiffusion.data.testdata_with_smpl import SingleImageDataset
    else:
        from mvdiffusion.data.single_image_dataset import SingleImageDataset
        
    # Get the  dataset
    validation_dataset = SingleImageDataset(
        **cfg.validation_dataset
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )
    dataset_param = {'image_dir': validation_dataset.root_dir, 'seg_dir': None, 'colab': False, 'has_det': True, 'hps_type': 'pixie'}
    econdata = SMPLDataset(dataset_param, device='cuda')

    carving = ReMesh(cfg.recon_opt, econ_dataset=econdata)
    run_inference(validation_dataloader, econdata, pipeline, carving, cfg, cfg.save_dir)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
