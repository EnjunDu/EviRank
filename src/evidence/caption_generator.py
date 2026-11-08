"""
P-CAP: Detailed Caption Generation (Section I.1, Figure 5).

Generates a comprehensive visual description of a query image to serve as
input for subsequent evidence mining stages.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from src.models.teacher_model import TeacherModel


# Prompt template from Figure 5 of the paper
PROMPT_P_CAP = """You are an expert at generating comprehensive and detailed visual descriptions for product images. Your task is to create a thorough description that captures both the overall appearance and fine-grained details of the product.

Query Image: <attached>

Generate a detailed product description that includes:
1. Overall Appearance - Product type/category; general shape and form factor; overall style (modern, vintage, classic, sporty, etc.).
2. Color & Material - Primary colors and any color patterns/combinations; material composition (metal, plastic, leather, fabric, wood, etc.); surface texture and finish (matte, glossy, brushed, etc.).
3. Structural Details - Key components and parts; construction and assembly details; size proportions and dimensions (relative terms).
4. Design Features - Unique design elements or patterns; branding, logos, or text if present; decorative elements or embellishments.
5. Functional Elements - Visible functional parts; hardware, fasteners, or connectors; any moving parts or adjustable features.

Please provide a comprehensive description in 3-5 detailed sentences that would allow someone to accurately identify and distinguish this specific product from similar items. Be as specific and descriptive as possible about visual characteristics.

Output: Only the detailed description text, nothing else."""


class CaptionGenerator:
    """
    Generates detailed captions for query images using an MLLM teacher.

    This implements the P-CAP stage (Section I.1): given a query image,
    produce a detailed textual description capturing visual identity.
    """

    def __init__(self, teacher_model: TeacherModel):
        self.teacher = teacher_model

    def generate_caption(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a detailed caption for the given image.

        Args:
            image: PIL Image of the query.
            custom_prompt: Override the default P-CAP prompt if provided.

        Returns:
            A detailed textual description of the image.
        """
        prompt = custom_prompt or PROMPT_P_CAP
        caption = self.teacher.generate_text_from_image(
            image=image,
            prompt=prompt,
        )
        return caption.strip()
