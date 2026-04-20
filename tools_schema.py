"""Anthropic tool-use schema for the Telegram bot.

Pure data. No imports, no logic — just the TOOLS list consumed by the
tool-use loop in bot.py.
"""

TOOLS = [
    {
        "name": "generate_post",
        "description": (
            "Generate a brand new social media post from scratch with a new concept and image. "
            "Use ONLY when the user wants something completely new — a new idea, new concept, new design. "
            "Do NOT use this for modifications to an existing post (use edit_post instead)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {
                    "type": "string",
                    "description": "Client/brand name (e.g. 'LMW', 'Georgoulis', 'Somamed'). Use 'ALL' if not specified.",
                },
                "brief": {
                    "type": "string",
                    "description": "Creative direction, theme, or topic for the post. Be specific.",
                },
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "instagram", "facebook"],
                    "description": "Social media platform. Default: linkedin",
                },
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                    "description": "Image source preference. 'stock' = stock photos only, 'ai' = AI-generated only, 'auto' = try stock first then AI. Default: auto. Use 'stock' when user explicitly asks for stock/real photos.",
                },
                "format": {
                    "type": "string",
                    "enum": ["square", "landscape"],
                    "description": "Canvas format. 'square' = 1080x1080 (default, Instagram/LinkedIn). 'landscape' = 1920x1080 (16:9, presentations, LinkedIn documents). Use 'landscape' when the user asks for landscape, 16:9, widescreen, or horizontal format.",
                },
                "use_last_inspiration": {
                    "type": "boolean",
                    "description": "Set to true ONLY when the user explicitly asks to make a post 'like this', 'similar to this', 'based on this' referring to an inspiration image they just sent. This forces the template to replicate that specific layout. Do NOT set this when generating a normal post.",
                },
                "style_overrides": {
                    "type": "object",
                    "description": (
                        "Override specific design decisions AFTER the AI picks them. Use when the user "
                        "specifies constraints like 'use black and white', 'use Montserrat font', 'make it red'. "
                        "Any field from edit_post works here: color_bg, color_text, color_accent, color_subtext, "
                        "font_headline, font_headline_weight, font_headline_size, font_headline_case, etc."
                    ),
                    "properties": {
                        "color_bg": {"type": "string"},
                        "color_text": {"type": "string"},
                        "color_accent": {"type": "string"},
                        "color_subtext": {"type": "string"},
                        "font_headline": {"type": "string"},
                        "font_headline_weight": {"type": "integer"},
                        "font_headline_size": {"type": "integer"},
                        "font_headline_case": {"type": "string"},
                        "font_subtext": {"type": "string"},
                    },
                },
            },
            "required": ["client", "brief"],
        },
    },
    {
        "name": "generate_carousel",
        "description": (
            "Generate multiple cohesive posts as a carousel/series with the same design language. "
            "Use when the user asks for multiple posts, a carousel, a series, or slides."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client/brand name"},
                "brief": {"type": "string", "description": "Theme/direction for the carousel"},
                "count": {
                    "type": "integer",
                    "description": "Number of posts (default 6, max 10)",
                    "minimum": 2,
                    "maximum": 10,
                },
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "instagram", "facebook"],
                },
            },
            "required": ["client", "brief"],
        },
    },
    {
        "name": "edit_post",
        "description": (
            "Modify the last generated post — change text, colors, fonts, sizes, layout, translate text, "
            "add/remove logo, adjust background color. The hero image stays the same. "
            "Use for ANY tweak to the existing post: translating, restyling, changing text, adjusting layout, "
            "changing colors (including background color). "
            "For visual/layout feedback like 'make pictures bigger', 'move text to the right', 'more spacing', "
            "'make the image cover the background', 'darker color' — use the 'feedback' field. "
            "IMPORTANT: Only include fields that need to change. Combine ALL user requests into ONE call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string",
                    "description": (
                        "Freeform visual feedback from the user that requires HTML/CSS changes. "
                        "Use for layout tweaks like 'make the pictures bigger', 'move text to the right', "
                        "'add more space between elements', 'make the grid 2x2', etc. "
                        "Opus will directly modify the template HTML/CSS based on this feedback."
                    ),
                },
                "headline": {"type": "string", "description": "New headline text (change for translation or rewording)"},
                "subtext": {"type": "string", "description": "New supporting text"},
                "cta": {"type": "string", "description": "New call-to-action text"},
                "template": {
                    "type": "string",
                    "enum": ["object-hero", "text-dominant", "split", "full-bleed"],
                    "description": "Layout template. Only change if user explicitly wants different layout.",
                },
                "font_headline": {"type": "string", "description": "Google Font name"},
                "font_headline_weight": {"type": "integer", "description": "Font weight (400-900)"},
                "font_headline_size": {"type": "integer", "description": "Font size in pixels (40-120)"},
                "font_headline_tracking": {"type": "string", "description": "Letter spacing CSS value"},
                "font_headline_case": {
                    "type": "string",
                    "enum": ["uppercase", "lowercase", "none"],
                },
                "color_bg": {"type": "string", "description": "Background color (hex). Use for any color change — 'darker orange', 'change to blue', 'lighter', etc."},
                "color_text": {"type": "string", "description": "Text color (hex)"},
                "color_accent": {"type": "string", "description": "Accent color (hex)"},
                "color_subtext": {"type": "string", "description": "Subtext color (hex)"},
                "headline_margin_x": {"type": "integer", "description": "Horizontal margin (0-200px)"},
                "headline_margin_y": {"type": "integer", "description": "Vertical margin (0-200px)"},
                "headline_max_width": {"type": "string", "description": "Max width CSS (e.g. '75%')"},
                "image_padding": {"type": "integer", "description": "Image padding (0-200px)"},
                "add_logo": {"type": "boolean", "description": "Set true to add the client's logo"},
                "remove_logo": {"type": "boolean", "description": "Set true to remove the logo"},
                "remove_background": {"type": "boolean", "description": "Set true to remove the background from the hero image (make it transparent). Use when user says 'remove the background', 'cut out the person', 'transparent background', etc."},
                "add_element": {
                    "type": "string",
                    "description": (
                        "Description of a visual element to ADD to the design. "
                        "Use when the user wants to add a photo/person/object that's NOT already in the post. "
                        "Just describe WHAT — e.g. 'the woman from the inspiration', 'a portrait', 'a product photo'. "
                        "Opus will look at the reference image and write the AI generation prompt itself. "
                        "Pair with feedback for WHERE to place it — e.g. 'on the right side, 40% width'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "replace_image",
        "description": (
            "Generate a completely NEW hero image for the last post. Keeps all design decisions (font, colors, layout, text). "
            "ONLY use when the user explicitly wants a different photo/image — 'replace the photo', 'use a stock image', 'change the picture'. "
            "Do NOT use for color changes, layout tweaks, or resizing — those go to edit_post. "
            "Use background_style ONLY when user explicitly asks for AI gradient/texture/abstract background — "
            "e.g. 'make an abstract gradient background', 'smoky texture background'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "REQUIRED when the user describes what they want. What the new image should show — "
                        "e.g. 'skyscrapers at sunset', 'business people in a meeting', 'abstract tech pattern'. "
                        "Extract this directly from the user's message. If not provided, reuses the original concept."
                    ),
                },
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                    "description": "Image source preference. 'stock' = stock photos only, 'ai' = AI-generated only, 'auto' = try stock first then AI. Use 'stock' when user asks for stock/real photos.",
                },
                "background_style": {
                    "type": "string",
                    "description": (
                        "Custom description for AI background generation. Use when the user wants a specific "
                        "background style like 'warm orange gradient', 'deep blue abstract', 'dark smoky texture', "
                        "'amber glow', etc. This overrides the concept and forces AI generation with this exact style. "
                        "Example: 'rich orange gradient with warm glow, abstract, no text'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "restore_image",
        "description": (
            "Restore a previously used image from the vault. Use when the user says "
            "'put back the woman', 'restore the old picture', 'use the previous image', "
            "'bring back the photo you removed'. The vault keeps all images from this session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Description of which image to restore — e.g. 'woman with bob cut', 'the first hero image', 'the portrait'. Fuzzy matched against stored image prompts.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "save_favorite",
        "description": (
            "Save the current post design as a favorite/liked template. Use when the user approves, "
            "loves, or wants to save the current post. Can include modifications to save alongside."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client to save for (overrides post's client)"},
                "modifications": {
                    "type": "object",
                    "description": "Design tweaks to save (e.g. {'color_accent': '#FF6B00'})",
                },
            },
            "required": [],
        },
    },
    {
        "name": "delete_template",
        "description": (
            "Delete a saved/liked template from memory. Use when the user doesn't like the current style, "
            "says 'delete this template', 'remove this style', 'I don't like this', 'forget this design'. "
            "Deletes the liked template that matches the current post's style."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Only delete templates for this client. Optional."},
                "delete_all": {"type": "boolean", "description": "Delete ALL liked templates. Only if user explicitly asks."},
            },
            "required": [],
        },
    },
    {
        "name": "process_feedback",
        "description": (
            "Process user feedback on the last taste analysis of an inspiration image. "
            "Use when user confirms, corrects, or refines observations about a photo they sent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback_text": {"type": "string", "description": "The user's feedback on the analysis"},
            },
            "required": ["feedback_text"],
        },
    },
    {
        "name": "get_taste_profile",
        "description": (
            "Show what the system has learned about the user's design taste and preferences. "
            "Use when user asks about their taste, style, preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "manage_templates",
        "description": "Show existing templates or rebuild them from taste data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["show", "rebuild"],
                    "description": "'show' to list templates, 'rebuild' to regenerate from taste",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "save_client_rule",
        "description": (
            "Save a permanent design rule or preference for a specific client. "
            "Use when the user says things like 'never use orange for Georgoulis', "
            "'always use dark backgrounds for LMW', 'Georgoulis should always use serif fonts'. "
            "These rules persist in the Brain and are applied to ALL future posts for that client."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client name the rule applies to"},
                "rule": {"type": "string", "description": "The design rule in clear language, e.g. 'never use orange or warm accent colors'"},
                "avoid_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific hex colors to avoid, e.g. ['#D97706', '#EA580C', '#F59E0B']",
                },
                "prefer_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred hex colors for this client",
                },
                "prefer_fonts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred fonts for this client",
                },
                "avoid_fonts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fonts to avoid for this client",
                },
            },
            "required": ["client", "rule"],
        },
    },
    {
        "name": "resend_last_post",
        "description": "Re-send the last generated post image. Use when user says 'send it', 'show me', etc.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "scout_designs",
        "description": (
            "Search the internet for fresh design inspiration using Perplexity. "
            "Returns a PREVIEW list of design references — does NOT save anything yet. "
            "The user must approve which ones to save (using approve_scout). "
            "Use when user says 'find me fresh designs', 'search for new layouts', "
            "'I'm bored of the layouts', 'bring me something new', 'find inspiration'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {
                    "type": "string",
                    "description": "Client name — searches for industry-specific designs. Use 'ALL' for general.",
                },
                "focus": {
                    "type": "string",
                    "description": "Specific focus for the search — e.g. 'minimalist healthcare', 'bold typography', 'editorial layouts with serif fonts'. Captures what the user specifically wants to find.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "approve_scout",
        "description": (
            "Approve specific design references from the scout results and save them to Brain. "
            "Use AFTER scout_designs has shown the preview list. "
            "User says which ones to keep: '1, 3, 5' or 'all' or 'none'. "
            "Only approved designs get processed and stored. "
            "If the user says WHY they don't like the results (e.g. 'none, too colorful' or "
            "'skip, I hate these layouts'), capture that reason in the feedback field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selected": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of item numbers to approve (1-based). E.g. [1, 3, 5]",
                },
                "all": {
                    "type": "boolean",
                    "description": "Set to true if user wants all items approved",
                },
                "feedback": {
                    "type": "string",
                    "description": "Why the user didn't like the results, if they said so. E.g. 'too colorful', 'not editorial enough', 'too tutorial-like'. Stored to improve future scouts.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_from_scout",
        "description": (
            "Generate a post using a specific scout design reference as the layout blueprint. "
            "Use when user says 'make a post like number 3', 'use that layout for LMW', "
            "'make one like 2 for Somamed'. Requires pending scout results from scout_designs. "
            "This both saves that specific layout AND generates a post following it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scout_item": {
                    "type": "integer",
                    "description": "The scout item number to use as layout (1-based)",
                },
                "client": {
                    "type": "string",
                    "description": "Client name for the post",
                },
                "brief": {
                    "type": "string",
                    "description": "Creative direction / topic for the post",
                },
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "instagram", "facebook"],
                },
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                },
            },
            "required": ["scout_item", "client", "brief"],
        },
    },
]
