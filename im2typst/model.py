"""Build the image→Typst model: pretrained TrOCR encoder + our-vocab decoder.

We load a pretrained TrOCR ``VisionEncoderDecoderModel`` (keeping its vision
encoder's pretraining, which already knows how to *read* rendered glyphs) and
resize the decoder's token embeddings to **our** :class:`CharTokenizer` vocab.
The encoder + its bundled image processor are reused as-is; only the decoder's
input/output vocabulary is swapped to ours.

Because the vision and text sides are decoupled, swapping the tokenizer later
(e.g. char-level → byte-level BPE in v2) only touches the decoder vocab here —
the encoder and image processor stay fixed.
"""

from __future__ import annotations

from transformers import AutoImageProcessor, VisionEncoderDecoderModel

from .tokenizer import CharTokenizer

#: Small TrOCR checkpoint — light enough for CPU smoke tests; printed variant
#: matches our clean rendered formulas better than the handwritten one.
DEFAULT_TROCR = "microsoft/trocr-small-printed"


def load_image_processor(trocr_name: str = DEFAULT_TROCR):
    """The encoder's bundled image processor (resize/normalize → pixel tensor)."""
    return AutoImageProcessor.from_pretrained(trocr_name)


def build_model(tokenizer: CharTokenizer, trocr_name: str = DEFAULT_TROCR,
                max_length: int = 128) -> VisionEncoderDecoderModel:
    """Load pretrained TrOCR and re-point its decoder at our tokenizer's vocab."""
    model = VisionEncoderDecoderModel.from_pretrained(trocr_name)

    # Resize the decoder's embedding + output layers to our (much smaller) vocab.
    # The transformer layers keep their pretrained weights (a warm start); only
    # the token embedding / lm_head rows are re-sized for our 54-token alphabet.
    model.decoder.resize_token_embeddings(tokenizer.vocab_size)

    # Wire our special-token IDs into the config so training (teacher forcing,
    # which builds decoder inputs by shifting labels) uses the right start/pad.
    cfg = model.config
    cfg.decoder_start_token_id = tokenizer.bos_id
    cfg.pad_token_id = tokenizer.pad_id
    cfg.eos_token_id = tokenizer.eos_id
    cfg.vocab_size = tokenizer.vocab_size

    # Generation params live on generation_config in transformers v5 (setting
    # max_length on model.config is rejected at generate() time).
    gen = model.generation_config
    gen.decoder_start_token_id = tokenizer.bos_id
    gen.pad_token_id = tokenizer.pad_id
    gen.eos_token_id = tokenizer.eos_id
    gen.max_length = max_length
    return model
