# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.cgmllm.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = ""
GEN_THINK_SYSTEM_PROMPT = "You should first think about the planning process in the mind."


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self):
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=False, vit=True):
        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_obj(self, obj_latents, gen_context):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input, kv_lens, ropes = self.model.prepare_und_obj(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            obj_latents=[obj_latents],
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_obj(past_key_values, **generation_input)
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def gen_obj(
        self,
        obj_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=4.0,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
    ):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        if isinstance(obj_shape, (list, tuple)):
            obj_shape = obj_shape[0]
        obj_sizes = [int(obj_shape)]
        generation_input = self.model.prepare_vae_obj_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            obj_sizes=obj_sizes,
            new_token_ids=self.new_token_ids,
        )

        cfg_text_past_key_values = cfg_text_precontext["past_key_values"]
        generation_input_cfg_text = self.model.prepare_vae_obj_latent_cfg(
            curr_kvlens=cfg_text_precontext["kv_lens"],
            curr_rope=cfg_text_precontext["ropes"],
            obj_sizes=obj_sizes,
        )
        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]
        generation_input_cfg_img = self.model.prepare_vae_obj_latent_cfg(
            curr_kvlens=cfg_img_precontext["kv_lens"],
            curr_rope=cfg_img_precontext["ropes"],
            obj_sizes=obj_sizes,
        )

        unpacked_latent = self.model.generate_obj(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
        )
        return unpacked_latent[0]

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        print("gen_text:", output)
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image, torch.Tensor]],
        think=False,
        understanding_output=False,
        vae_obj_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=None,
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        obj_shapes=512,
        **unused,
    ) -> List[Union[str, torch.Tensor]]:
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vit_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=False, vit=True)
                    cfg_text_context = deepcopy(gen_context)
                elif isinstance(input_term, torch.Tensor):
                    gen_context = self.update_context_obj(input_term, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                output_list.append(
                    self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                )
            elif vae_obj_output:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)
                output_list.append(
                    self.gen_obj(
                        obj_shapes,
                        gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                )

        return output_list

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        obj: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        output_dict = {"image": None, "text": None, "vae_obj": None}
        if image is None and text is None and obj is None:
            print("Please provide at least one input: image, text, or obj.")
            return output_dict

        input_list = []
        if obj is not None:
            input_list.append(obj)
        if text is not None:
            input_list.append(text)
        if image is not None:
            input_list.append(image)

        for item in self.interleave_inference(input_list, **kwargs):
            if isinstance(item, str):
                output_dict["text"] = item
            elif isinstance(item, torch.Tensor):
                output_dict["vae_obj"] = item
        return output_dict
