"""
Caption prompts for the LILIE captioning loss.
"""
from __future__ import annotations

import os

import numpy as np
import yaml
from PIL import Image
from transformers import AutoProcessor

from common.data import CAPTION_TYPES

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


class CaptionIO:
    """Tokenizer-side helper: renders the four prompts and packs captions into fixed rows."""

    def __init__(self, model_dir: str, image_size: int, max_caption_tokens: int):
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.stop_token_id = self.tokenizer.eos_token_id  # <end_of_utterance>
        self.max_caption_tokens = max_caption_tokens
        # One unsplit 512 px tile per image -> 64 image tokens in the prompt.
        for obj in (self.processor, self.processor.image_processor):
            obj.image_seq_len = 64
            obj.do_image_splitting = False
        rendered = [self._render(name, image_size) for name in CAPTION_TYPES]
        self.plen = max(len(ids) for ids in rendered)
        self.prompt_ids = np.full((len(rendered), self.plen), self.pad_token_id, np.int32)
        self.prompt_mask = np.zeros((len(rendered), self.plen), np.int32)
        for i, ids in enumerate(rendered):
            self.prompt_ids[i, self.plen - len(ids):] = ids
            self.prompt_mask[i, self.plen - len(ids):] = 1
        self.width = self.plen + max_caption_tokens

    def _render(self, caption_type: str, image_size: int) -> np.ndarray:
        with open(os.path.join(PROMPT_DIR, f"caption_{caption_type}.yaml")) as handle:
            prompt = yaml.safe_load(handle)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt["prompt"]}]},
            {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (image_size, image_size))},
                                         {"type": "text", "text": prompt["user_prompt"]}]},
        ]
        out = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")
        ids = np.asarray(out["input_ids"][0], dtype=np.int32)
        assert int((ids == self.image_token_id).sum()) == 64, "prompt must carry exactly 64 image tokens"
        return ids

    def collate(self, captions: list[str], caption_types: list[str]) -> dict[str, np.ndarray]:
        """Token ids, attention mask and labels (``-100`` outside the caption) for a batch."""
        B = len(captions)
        input_ids = np.full((B, self.width), self.pad_token_id, np.int32)
        attention = np.zeros((B, self.width), np.int32)
        labels = np.full((B, self.width), -100, np.int32)
        # "Assistant:" has no trailing space. BPE puts the space in the first answer token (" A" != "A")
        encoded = self.tokenizer([" " + c.strip() for c in captions], add_special_tokens=False)["input_ids"]
        for i, (ids, caption_type) in enumerate(zip(encoded, caption_types)):
            ids = list(ids)[: self.max_caption_tokens - 1] + [self.stop_token_id]
            style = CAPTION_TYPES.index(caption_type)
            end = self.plen + len(ids)
            input_ids[i, : self.plen] = self.prompt_ids[style]
            attention[i, : self.plen] = self.prompt_mask[style]
            input_ids[i, self.plen:end] = ids
            attention[i, self.plen:end] = 1
            labels[i, self.plen:end] = ids
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
