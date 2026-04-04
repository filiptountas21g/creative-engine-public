"""
Dataclasses for pipeline step inputs and outputs.
Each step takes one typed input and returns one typed output.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineInput:
    """Initial input to the pipeline."""
    client: str
    brief: str
    platform: str = "linkedin"
    template_override: Optional[str] = None  # force a specific template
    format: str = "square"  # "square" (1080x1080) or "landscape" (1920x1080)


@dataclass
class ResearchResult:
    """Step 1 output: Perplexity research."""
    trends: str
    raw_response: str = ""


@dataclass
class BrainContext:
    """Step 2 output: Big Brain data."""
    brand: dict = field(default_factory=dict)
    past_concepts: list = field(default_factory=list)
    taste_context: dict = field(default_factory=dict)
    client_profile: dict = field(default_factory=dict)


@dataclass
class CreativeConcept:
    """Step 3 output: Opus creative concept."""
    object: str = ""
    why: str = ""
    emotional_direction: str = ""
    composition_note: str = ""
    background: str = "#F5F4F0"
    lighting: str = "soft studio daylight"
    what_to_avoid: str = ""


@dataclass
class CopyOptions:
    """Step 4 output: Sonnet copywriting."""
    headlines: list = field(default_factory=list)  # 3 options
    subtext: str = ""
    cta: str = ""


@dataclass
class ImageResult:
    """Step 5 output: generated image."""
    image_path: str = ""
    image_url: str = ""
    model_used: str = ""  # "flux2" or "ideogram3"
    prompt_used: str = ""


@dataclass
class CreativeDecisions:
    """Step 6 output: Opus editor decisions."""
    headline: str = ""
    headline_reason: str = ""
    font_headline: str = "Inter"
    font_headline_weight: int = 800
    font_headline_size: int = 68
    font_headline_tracking: str = "-0.02em"
    font_headline_line_height: float = 1.05
    font_headline_case: str = "uppercase"
    font_subtext: str = "DM Sans"
    font_subtext_weight: int = 400
    font_subtext_size: int = 18
    color_bg: str = "#F5F4F0"
    color_text: str = "#2A2A2A"
    color_accent: str = "#C4A77D"
    color_subtext: str = "#6B6B6B"
    headline_margin_x: int = 64
    headline_margin_y: int = 64
    headline_max_width: str = "75%"
    image_padding: int = 100
    template: str = "object-hero"
    subtext: str = ""
    cta: str = ""


@dataclass
class RenderResult:
    """Step 7 output: final PNG."""
    final_image_path: str = ""
    width: int = 1080
    height: int = 1080


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    success: bool = False
    image_path: str = ""
    concept: Optional[CreativeConcept] = None
    copy: Optional[CopyOptions] = None
    decisions: Optional[CreativeDecisions] = None
    hero_image: Optional[ImageResult] = None
    extra_images: Optional[list] = None
    template_html: str = ""
    logo_b64: Optional[str] = None
    error: str = ""
