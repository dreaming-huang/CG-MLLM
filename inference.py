"""CG-MLLM inference: 3D generation and multimodal understanding."""

import argparse
import copy
import io
import os
import random
from argparse import Namespace

import numpy as np
import torch
import trimesh
import yaml
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from easydict import EasyDict
from PIL import Image
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.cgmllm import (
    CGMLLM,
    CGMLLMConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from pointbert.point_encoder import PointTransformer
from transformers import Qwen2VLForConditionalGeneration as Qwen2VIT

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as Qwen2_5VIT
except ImportError:
    Qwen2_5VIT = None

try:
    from transformers import Qwen3VLForConditionalGeneration as Qwen3VIT
except ImportError:
    Qwen3VIT = None


class DefaultNoneNamespace(Namespace):
    def __getattr__(self, name):
        return None


DEFAULT_IMAGE = "examples/chairo.png"
DEFAULT_IMG_UND_PROMPT = "Describe the object in the image."
DEFAULT_T2OBJ_PROMPT = (
    "A medieval knight in T-pose, metallic armor with decorative trim, holding a sword and shield."
)
DEFAULT_OBJ_NPY = "examples/0ea33b6617174530b97d6b7a92c275fb_8192.npy"
DEFAULT_OBJ_UND_PROMPT = "What does this collection of points represent?"
DEFAULT_T2T_PROMPT = "What is Computer Graphics?"


def prompt_with_default(message, default):
    value = input(f"{message} [{default}]: ").strip()
    return value or default


UND_OBJ_ENCODER_DIMS = {
    "hunyuan": 64,
    "uni3d": 1408,
    "pointbert": 1152,
}
POINTBERT_NUM_POINTS = 8192
UNI3D_NUM_POINTS = 10000
OCTREE_RESOLUTION = 512


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except Exception:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


def preprocess_image(image_path, quality=85):
    image = Image.open(image_path)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue()))


# Hunyuan3D-2.1 ImageProcessorV2.__call__ passes border_ratio=0.15 into recenter.
HUNYUAN_RECENTER_BORDER_RATIO = 0.15


def composite_rgba_on_white(image):
    """Composite a straight-alpha RGBA cutout onto opaque white."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def recenter_rgba(image, border_ratio=HUNYUAN_RECENTER_BORDER_RATIO):
    """Hunyuan3D-2.1 ImageProcessorV2.recenter, before the white composite.

    Builds a square canvas of max(H, W), scales the foreground box to
    (1 - border_ratio) of that canvas, and centers it. Transparent pixels stay
    transparent. Hunyuan then composites this onto white; it does not JPEG-encode.
    """
    import cv2

    rgba = np.asarray(image.convert("RGBA"))
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(f"Expected an RGBA image, got shape {getattr(rgba, 'shape', None)}.")
    mask = rgba[..., 3]
    coords = np.nonzero(mask)
    if coords[0].size == 0:
        raise ValueError("Background removal left an empty foreground, so recenter has nothing to place.")
    height, width, channels = rgba.shape
    size = max(height, width)
    x_min, x_max = int(coords[0].min()), int(coords[0].max())
    y_min, y_max = int(coords[1].min()), int(coords[1].max())
    box_h = x_max - x_min
    box_w = y_max - y_min
    if box_h <= 0 or box_w <= 0:
        raise ValueError("Background removal left an empty foreground, so recenter has nothing to place.")
    desired = int(size * (1 - border_ratio))
    if desired <= 0:
        raise ValueError(f"Image is too small to recenter (size={size}).")
    scale = desired / max(box_h, box_w)
    out_h = int(box_h * scale)
    out_w = int(box_w * scale)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Background removal left an empty foreground, so recenter has nothing to place.")
    resized = cv2.resize(
        rgba[x_min:x_max, y_min:y_max],
        (out_w, out_h),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size, channels), dtype=np.uint8)
    x2_min = (size - out_h) // 2
    y2_min = (size - out_w) // 2
    canvas[x2_min:x2_min + out_h, y2_min:y2_min + out_w] = resized
    return Image.fromarray(canvas, mode="RGBA")


def get_background_remover(args=None):
    """Lazy Hunyuan3D-2.1 BackgroundRemover (rembg new_session() -> u2net)."""
    if args is not None and getattr(args, "_bg_remover", None) is not None:
        return args._bg_remover
    from hy3dshape.rembg import BackgroundRemover
    remover = BackgroundRemover()
    if args is not None:
        args._bg_remover = remover
    return remover


def remove_background_hunyuan(image, args=None):
    """Match Hunyuan3D-2.1 hy3dshape/rembg.py: rembg u2net, bgcolor=[255,255,255,0]."""
    remover = get_background_remover(args)
    return remover(image)


def pc_norm(pc):
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / scale
    return np.concatenate((xyz, other_feature), axis=1)


def _sanitize_filename(text, max_len=100):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in text.strip())
    return (safe.strip("._") or "prompt")[:max_len]


def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                outputs.append(trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f))
        return outputs
    mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
    return trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)


def get_und_obj_encoder_kind(path):
    if path is None:
        return "hunyuan"
    name = os.path.basename(str(path)).lower()
    if "point_bert" in name or "pointbert" in name:
        return "pointbert"
    if "uni3d" in name:
        return "uni3d"
    return "hunyuan"


def get_und_obj_vae_dim(path):
    return UND_OBJ_ENCODER_DIMS[get_und_obj_encoder_kind(path)]


def load_und_obj_encoder(path, fallback_hunyuan_vae=None):
    kind = get_und_obj_encoder_kind(path)
    print(f"[und_obj] encoder={kind}, path={path}")
    if kind == "hunyuan":
        if fallback_hunyuan_vae is None:
            raise ValueError("Hunyuan understanding encoder needs the already-loaded shape VAE.")
        return fallback_hunyuan_vae

    if kind == "uni3d":
        try:
            import Uni3D.models.uni3d as uni3d
        except ImportError as exc:
            raise ImportError("Uni3D is not installed. Provide a Hunyuan or PointBERT encoder instead.") from exc
        args = DefaultNoneNamespace(
            model="create_uni3d",
            npoints=UNI3D_NUM_POINTS,
            num_group=512,
            group_size=64,
            pc_encoder_dim=512,
            clip_model="EVA02-E-14-plus",
            pretrained="/path/to/clip_model/open_clip_pytorch_model.bin",
            pc_model="eva_giant_patch14_560",
            pc_feat_dim=1408,
            embed_dim=1024,
            evaluate_3d=True,
            ckpt_path=path,
            patch_dropout=0,
        )
        model = getattr(uni3d, args.model)(args=args).cuda()
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = checkpoint["module"]
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model.eval()

    if kind == "pointbert":
        config_path = os.path.join(os.path.dirname(__file__), "pointbert", "PointTransformer_base_8192point.yaml")
        print(f"Loading PointBERT config from {config_path}.")
        point_bert_config = cfg_from_yaml_file(config_path)
        point_bert_config.model.point_dims = 6
        use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)
        model = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
        model.load_checkpoint(path)
        return model.cuda().eval()

    raise ValueError(f"Unsupported understanding encoder: {path}")


def load_shape_vae(obj_vae_path, obj_vae_len):
    kwargs = dict(use_safetensors=False, variant="fp16")
    if obj_vae_path == "tencent/Hunyuan3D-2.1":
        if obj_vae_len == 4096:
            return ShapeVAE.from_pretrained(obj_vae_path, **kwargs)
        if obj_vae_len in (2048, 1024, 512):
            return ShapeVAE.from_pretrained(
                obj_vae_path,
                num_latents=obj_vae_len,
                pc_size=obj_vae_len * 20,
                pc_sharpedge_size=0,
                **kwargs,
            )
    elif obj_vae_path == "tencent/Hunyuan3D-2":
        return ShapeVAE.from_pretrained(
            obj_vae_path,
            num_latents=obj_vae_len,
            pc_size=obj_vae_len * 10,
            pc_sharpedge_size=obj_vae_len * 10,
            subfolder="hunyuan3d-vae-v2-0-withencoder",
            **kwargs,
        )
    elif obj_vae_path == "tencent/Hunyuan3D-2mini":
        return ShapeVAE.from_pretrained(
            obj_vae_path,
            num_latents=obj_vae_len,
            pc_size=obj_vae_len * 10,
            pc_sharpedge_size=obj_vae_len * 10,
            subfolder="hunyuan3d-vae-v2-mini-withencoder",
            **kwargs,
        )
    raise ValueError(f"Unsupported shape VAE: {obj_vae_path} with length {obj_vae_len}")


def load_qwen_vit(llm_base_path):
    if "Qwen2-VL-2B-Instruct" in llm_base_path:
        if Qwen2VIT is None:
            raise ImportError("Qwen2-VL is not available in this transformers version.")
        model_vl = Qwen2VIT.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    elif "Qwen2.5-VL" in llm_base_path:
        if Qwen2_5VIT is None:
            raise ImportError("Qwen2.5-VL is not available in this transformers version.")
        model_vl = Qwen2_5VIT.from_pretrained(llm_base_path)
    elif "Qwen3-VL" in llm_base_path:
        if Qwen3VIT is None:
            raise ImportError("Qwen3-VL is not available in this transformers version.")
        model_vl = Qwen3VIT.from_pretrained(llm_base_path)
    else:
        raise ValueError(f"Cannot infer a Qwen ViT from llm_base_path={llm_base_path}")
    vit_model = copy.deepcopy(model_vl.visual)
    vit_config = vit_model.config
    del model_vl
    return vit_model, vit_config


def setup_model(args):
    print("Loading model...")
    llm_config = (
        Qwen2Config.from_pretrained(args.llm_base_path)
        if not args.use_qwen_vl
        else Qwen2VLConfig.from_pretrained(args.llm_base_path)
    )
    if hasattr(llm_config, "text_config") and llm_config.text_config is not None:
        for key, value in llm_config.text_config.items():
            setattr(llm_config, key, value)
        del llm_config.text_config
    llm_config.qk_norm = args.qk_norm
    llm_config.tie_word_embeddings = True
    if args.mot:
        llm_config.layer_module = "Qwen2MoTDecoderLayer" if not args.use_qwen_vl else "Qwen2VLMoTDecoderLayer"
    else:
        llm_config.layer_module = "Qwen2DecoderLayer" if not args.use_qwen_vl else "Qwen2VLDecoderLayer"

    if args.use_qwen_vit:
        vit_model, vit_config = load_qwen_vit(args.llm_base_path)
    else:
        vit_config = SiglipVisionConfig.from_json_file(args.vit_config)
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        vit_model = SiglipVisionModel(vit_config)

    vae_model, vae_config = load_ae(local_path=args.image_vae_path)
    language_model = Qwen2ForCausalLM(llm_config) if not args.use_qwen_vl else Qwen2VLForConditionalGeneration(llm_config)

    config = CGMLLMConfig(
        visual_gen=False,
        visual_und=True,
        obj_gen=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
        obj_vae_len=args.obj_vae_len,
        need_obj_pe=args.need_obj_pe,
        und_obj_vae_dim=get_und_obj_vae_dim(args.und_obj_vae_path),
        use_dinov2=args.use_dinov2,
        dinov2_model_name=args.dinov2_model_name,
        dinov2_hidden_size=args.dinov2_hidden_size,
        dinov2_image_size=args.dinov2_image_size,
        dinov2_patch_size=args.dinov2_patch_size,
        dinov2_fusion=args.dinov2_fusion,
        dinov2_gate_init=args.dinov2_gate_init,
        freeze_dinov2=True,
    )
    model = CGMLLM(language_model, vit_model, config)
    if not args.use_qwen_vit:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.llm_base_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, getattr(vit_config, "patch_size", 14), 2 if args.use_qwen_vit else 1)

    if args.mot:
        no_split_classes = ["CGMLLM", "Qwen2MoTDecoderLayer" if not args.use_qwen_vl else "Qwen2VLMoTDecoderLayer"]
    else:
        no_split_classes = ["CGMLLM", "Qwen2DecoderLayer" if not args.use_qwen_vl else "Qwen2VLDecoderLayer"]

    device_map = infer_auto_device_map(
        model,
        max_memory={i: args.max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=no_split_classes,
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
        "obj2llm",
        "llm2obj",
        "dinov2_connector",
        "dinov2_gate",
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for key in same_device_modules:
        if torch.cuda.device_count() == 1:
            device_map[key] = first_device if key in device_map else "cuda:0"
        elif key in device_map:
            device_map[key] = first_device

    checkpoint_path = os.path.join(args.checkpoint, "ema.safetensors")
    print(f"Loading weights from {checkpoint_path}")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload",
        strict=False,
    ).eval()
    print("Model loaded.")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def export_obj_from_latents(latents, output_path, vae, z_scale_factor, octree_resolution=OCTREE_RESOLUTION):
    latents = (1.0 / z_scale_factor) * latents.to("cuda", dtype=torch.float16).unsqueeze(0)
    latents = vae.decode(latents)
    mesh_path = output_path.replace(".pth", ".obj")
    if "Michelangelo" in type(vae).__name__:
        mesh = vae.extract_geometry(
            latents,
            mc_level=0.0,
            bounds=1.05,
            octree_resolution=max(octree_resolution, 256),
            enable_pbar=True,
        )
        mesh = trimesh.Trimesh(vertices=mesh[0].verts.cpu().numpy(), faces=mesh[0].faces.cpu().numpy())
        mesh.fix_normals()
        mesh.export(mesh_path, file_type="obj")
    else:
        mesh = vae.latents2mesh(
            latents,
            output_type="trimesh",
            bounds=1.01,
            mc_level=0.0,
            num_chunks=20000,
            octree_resolution=octree_resolution,
            mc_algo="mc",
            enable_pbar=True,
        )
        mesh = export_to_trimesh(mesh)[0]
        mesh.export(mesh_path, file_type="obj")
    print(f"Saved mesh to {mesh_path}")
    return mesh_path


def image_understanding(inferencer, image_path=None, prompt=None):
    print("\n[Image understanding]")
    if image_path is None:
        image_path = prompt_with_default("Image path", DEFAULT_IMAGE)
    if prompt is None:
        prompt = prompt_with_default("English question about the image", DEFAULT_IMG_UND_PROMPT)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    image = Image.open(image_path)

    output = inferencer(
        image=image,
        text=prompt,
        understanding_output=True,
        max_think_token_n=1000,
        do_sample=False,
    )
    text = output.get("text") or ""
    print("Model answer:")
    print(text)
    return text


def _transform_rgb_channels(transform):
    norm = getattr(transform, "normalize_transform", None)
    mean = getattr(norm, "mean", None)
    if mean is None:
        return None
    return len(tuple(mean))


def model_image_from_rgba_cutout(cutout_rgba, inferencer):
    """RGB view of a transparent cutout. Never composites onto white.

    ImageTransform.Normalize is 3-channel, so alpha cannot be fed as a fourth
    channel. PIL RGBA->RGB on this Pillow drops alpha instead of compositing, and rembg
    stores white RGB under alpha 0. Composite onto black so the model sees
    black where the cutout was transparent. Skip the JPEG round-trip.
    """
    if cutout_rgba.mode != "RGBA":
        cutout_rgba = cutout_rgba.convert("RGBA")
    channels = []
    for name in ("vae_transform", "vit_transform"):
        nchan = _transform_rgb_channels(getattr(inferencer, name, None))
        if nchan is not None:
            channels.append(f"{name}={nchan}")
    rgba = np.asarray(cutout_rgba)
    mask = rgba[:, :, 3] == 0
    dropped = cutout_rgba.convert("RGB")
    dropped_arr = np.asarray(dropped)
    dropped_max = int(dropped_arr[mask].max()) if mask.any() else -1
    # This Pillow RGBA->RGB drops alpha and keeps rembg's white RGB (bgcolor 255,255,255,0).
    # Composite onto black so former transparent pixels are black, never white.
    black = Image.new("RGBA", cutout_rgba.size, (0, 0, 0, 255))
    rgb = Image.alpha_composite(black, cutout_rgba).convert("RGB")
    rgb_arr = np.asarray(rgb)
    mx = int(rgb_arr[mask].max()) if mask.any() else -1
    print(
        "[i2obj] Transparent background: NOT composited on white. "
        f"PIL RGBA->RGB alone leaves max RGB {dropped_max} where alpha==0 "
        "(alpha dropped, rembg RGB kept). "
        "Model input is an explicit composite onto black; JPEG round-trip skipped. "
        "normalize " + (", ".join(channels) or "unknown")
        + f". max RGB where alpha==0 after black composite is {mx}."
    )
    if mx > 0:
        raise RuntimeError(
            f"Transparent pixels are not black after black composite (max={mx}). Refusing to continue."
        )
    return rgb


def image_to_obj(inferencer, args, image_path=None):
    print("\n[Image to 3D object]")
    if image_path is None:
        image_path = prompt_with_default("Image path", DEFAULT_IMAGE)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    if getattr(args, "remove_bg", True):
        bg_mode = getattr(args, "bg", None) or "white"
        if bg_mode not in ("white", "transparent"):
            raise ValueError(f"Unsupported --bg {bg_mode!r}; choose white or transparent.")
        print(f"[i2obj] Removing background with rembg (Hunyuan3D-2.1 / u2net), bg={bg_mode}...")
        cutout_rgba = remove_background_hunyuan(Image.open(image_path), args=args)
        if cutout_rgba.mode != "RGBA":
            cutout_rgba = cutout_rgba.convert("RGBA")
        border_ratio = float(getattr(args, "border_ratio", HUNYUAN_RECENTER_BORDER_RATIO))
        if not 0.0 <= border_ratio < 1.0:
            raise ValueError(f"--border_ratio must be in [0, 1), got {border_ratio}.")
        cutout_rgba = recenter_rgba(cutout_rgba, border_ratio=border_ratio)
        print(
            "[i2obj] Recentered foreground on a square canvas "
            f"(border_ratio={border_ratio}, object fills {1 - border_ratio:.0%} of the long side)."
        )
        os.makedirs(args.output_dir, exist_ok=True)
        cutout_path = os.path.join(args.output_dir, f"{stem}_rmbg.png")
        if bg_mode == "transparent":
            cutout_rgba.save(cutout_path, format="PNG")
            print(f"[i2obj] Saved transparent RGBA cutout to {cutout_path}")
            image = model_image_from_rgba_cutout(cutout_rgba, inferencer)
        else:
            cutout_rgb = composite_rgba_on_white(cutout_rgba)
            cutout_rgb.save(cutout_path)
            print(f"[i2obj] Saved white-background cutout to {cutout_path}")
            image = cutout_rgb
    else:
        if (getattr(args, "bg", "white") or "white") != "white":
            print("[i2obj] --bg is ignored because background removal is off (--no_remove_bg).")
        print("[i2obj] Skipping background removal (--no_remove_bg).")
        image = preprocess_image(image_path)
    print(f"[i2obj] Model input image mode={image.mode} size={image.size}")
    output_path = os.path.join(args.output_dir, stem + "_to_obj.pth")
    output = inferencer(
        image=image,
        vae_obj_output=True,
        cfg_text_scale=args.txt_cfg,
        cfg_img_scale=args.img_cfg,
        cfg_interval=[0.4, 1.0],
        timestep_shift=args.timestep_shift,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        obj_shapes=args.obj_vae_len,
    )
    return export_obj_from_latents(output["vae_obj"], output_path, args.shape_vae, args.z_scale_factor)


def text_to_obj(inferencer, args, prompt=None):
    print("\n[Text to 3D object]")
    if prompt is None:
        prompt = prompt_with_default("English text prompt", DEFAULT_T2OBJ_PROMPT)
    output_path = os.path.join(args.output_dir, _sanitize_filename(prompt) + "_t2obj.pth")
    output = inferencer(
        text=prompt,
        vae_obj_output=True,
        cfg_text_scale=args.txt_cfg,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=args.timestep_shift,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        obj_shapes=args.obj_vae_len,
    )
    return export_obj_from_latents(output["vae_obj"], output_path, args.shape_vae, args.z_scale_factor)


def text_to_text(inferencer, prompt=None):
    print("\n[Text to text]")
    if prompt is None:
        prompt = prompt_with_default("English prompt", DEFAULT_T2T_PROMPT)
    output = inferencer(
        text=prompt,
        think=False,
        understanding_output=True,
        max_think_token_n=100,
        do_sample=False,
    )
    text = output.get("text") or ""
    print("Model answer:")
    print(text)
    return text


def encode_obj_for_understanding(args, obj_path):
    kind = get_und_obj_encoder_kind(args.und_obj_vae_path)
    if kind == "hunyuan":
        surface = args.und_obj_loader(obj_path).to("cuda", dtype=torch.float16)
        with torch.no_grad():
            latents = args.shape_vae.encode(surface)
        return latents.view(-1, args.obj_vae_len, 64)

    if kind == "uni3d":
        surface = args.und_obj_loader(obj_path).to("cuda", dtype=torch.float16)
        pc = surface[:, :UNI3D_NUM_POINTS, :3]
        rgb = torch.ones_like(pc).float() * 0.4
        feature = torch.cat((pc, rgb), dim=-1)
        with torch.no_grad():
            pc_features = args.und_obj_vae.encode_pc(feature)
        return pc_features.view(-1, pc_features.shape[-1]).unsqueeze(0)

    if kind == "pointbert":
        point_cloud = np.load(obj_path)
        point_cloud = pc_norm(point_cloud)
        feature = torch.from_numpy(point_cloud.astype(np.float32))
        device = next(args.und_obj_vae.parameters()).device
        with torch.no_grad():
            pc_features = args.und_obj_vae(feature.unsqueeze(0).to(device))
        return pc_features.view(-1, pc_features.shape[-1]).unsqueeze(0)

    raise ValueError(f"Unsupported understanding encoder kind: {kind}")


def obj_understanding(inferencer, args, obj_path=None, prompt=None):
    print("\n[3D object understanding]")
    kind = get_und_obj_encoder_kind(args.und_obj_vae_path)
    if obj_path is None:
        if kind == "pointbert":
            obj_path = prompt_with_default("Point-cloud .npy path", DEFAULT_OBJ_NPY)
        else:
            obj_path = input("Mesh path (.obj / .glb): ").strip()
    if prompt is None:
        prompt = prompt_with_default("English question about the 3D object", DEFAULT_OBJ_UND_PROMPT)
    if not os.path.exists(obj_path):
        print(f"File not found: {obj_path}")
        return None

    obj_latent = encode_obj_for_understanding(args, obj_path)
    output = inferencer(
        obj=obj_latent,
        text=prompt,
        understanding_output=True,
        max_think_token_n=1000,
        do_sample=False,
    )
    text = output.get("text") or ""
    print("Model answer:")
    print(text)
    return text


def run_interactive(inferencer, args):
    while True:
        print("\n" + "=" * 44)
        print("            CG-MLLM Inference")
        print("=" * 44)
        print("1. Image understanding")
        print("2. Image to 3D object")
        print("3. Text to 3D object")
        print("4. Text to text")
        print("5. 3D object understanding")
        print("0. Exit")
        print("=" * 44)
        choice = input("Select a function: ").strip()
        if choice == "1":
            image_understanding(inferencer)
        elif choice == "2":
            image_to_obj(inferencer, args)
        elif choice == "3":
            text_to_obj(inferencer, args)
        elif choice == "4":
            text_to_text(inferencer)
        elif choice == "5":
            obj_understanding(inferencer, args)
        elif choice == "0":
            print("Done.")
            break
        else:
            print("Invalid choice. Enter a number from the menu.")


def parse_args():
    parser = argparse.ArgumentParser(description="CG-MLLM inference for 3D generation and understanding.")
    parser.add_argument("--llm_base_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--checkpoint", "--custom_model_path", dest="checkpoint", type=str, required=True,
                        help="Checkpoint directory that contains ema.safetensors.")
    parser.add_argument("--image_vae_path", type=str, default="models/BAGEL-7B-MoT/ae.safetensors")
    parser.add_argument("--vit_config", type=str, default="models/BAGEL-7B-MoT/vit_config.json")
    parser.add_argument("--obj_vae_path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--und_obj_vae_path", type=str, default="models/point_bert_v1.1.pt",
                        help="3D understanding encoder. Default is PointBERT at models/point_bert_v1.1.pt. "
                             "Also accepts a Hunyuan3D repo id or uni3d_g.pt. "
                             "PointBERT and Uni3D are selected from the filename.")
    parser.add_argument("--obj_vae_len", type=int, default=4096)
    parser.add_argument("--txt_cfg", type=float, default=4.0)
    parser.add_argument("--img_cfg", type=float, default=7.5)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--z_scale_factor", type=float, default=1.0)
    parser.add_argument("--use_qwen_vit", action="store_true")
    parser.add_argument("--use_qwen_vl", action="store_true")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--need_obj_pe", action="store_true")
    parser.add_argument("--use_dinov2", action="store_true")
    parser.add_argument("--dinov2_model_name", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--dinov2_hidden_size", type=int, default=1024)
    parser.add_argument("--dinov2_image_size", type=int, default=518)
    parser.add_argument("--dinov2_patch_size", type=int, default=14)
    parser.add_argument("--dinov2_fusion", type=str, default="gated_add", choices=["gated_add", "add"])
    parser.add_argument("--dinov2_gate_init", type=float, default=0.0)
    parser.add_argument("--mot", type=lambda x: x.lower() not in ("false", "0", "no"), default=True)
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "img_und", "i2obj", "t2obj", "t2t", "obj_und"],
        help="interactive menu, or a single task.",
    )
    parser.add_argument("--image", type=str, default=None, help="Input image for i2obj / img_und.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt or question.")
    parser.add_argument("--obj", type=str, default=None, help="Mesh or point-cloud file for obj_und.")
    parser.add_argument(
        "--remove_bg",
        dest="remove_bg",
        action="store_true",
        default=True,
        help="Remove image background with rembg u2net (Hunyuan3D-2.1 style) before i2obj. Default: on.",
    )
    parser.add_argument(
        "--no_remove_bg",
        dest="remove_bg",
        action="store_false",
        help="Disable Hunyuan-style rembg foreground removal for i2obj.",
    )
    parser.add_argument(
        "--border_ratio",
        type=float,
        default=HUNYUAN_RECENTER_BORDER_RATIO,
        help=(
            "Empty margin after rembg, as a fraction of the square canvas. "
        ),
    )
    parser.add_argument(
        "--bg",
        type=str,
        default="white",
        choices=["white", "transparent"],
        help=(
            "Background after rembg and Hunyuan recenter. white (default): composite the "
            "cutout onto white, with no JPEG round-trip. "
            "transparent: keep the RGBA cutout and save it as PNG. "
            "Ignored with --no_remove_bg. The 4B model should stay on white."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = args.output_dir or os.path.join(args.checkpoint, "output_images")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    args.shape_vae = load_shape_vae(args.obj_vae_path, args.obj_vae_len).cuda().half()
    args.und_obj_vae = load_und_obj_encoder(args.und_obj_vae_path, fallback_hunyuan_vae=args.shape_vae)
    args.und_obj_loader = SharpEdgeSurfaceLoader(
        num_uniform_points=args.obj_vae_len * 20,
        num_sharp_points=0,
    )

    inferencer = setup_model(args)

    if args.mode == "interactive":
        run_interactive(inferencer, args)
    elif args.mode == "img_und":
        image_understanding(inferencer, image_path=args.image, prompt=args.prompt)
    elif args.mode == "i2obj":
        image_to_obj(inferencer, args, image_path=args.image)
    elif args.mode == "t2obj":
        text_to_obj(inferencer, args, prompt=args.prompt)
    elif args.mode == "t2t":
        text_to_text(inferencer, prompt=args.prompt)
    elif args.mode == "obj_und":
        obj_understanding(inferencer, args, obj_path=args.obj, prompt=args.prompt)


if __name__ == "__main__":
    main()
