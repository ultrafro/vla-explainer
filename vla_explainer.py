"""
VLA (Vision-Language-Action) Models Explainer
In the style of 3Blue1Brown using Manim Community Edition

Covers:
1. What are VLAs?
2. Architecture deep-dive
3. Training pipeline
4. How inference works
5. Limitations
6. New research directions
"""

from manim import *
import numpy as np

# ─── Color palette (3B1B-inspired) ─────────────────────────────────
BG_COLOR = "#1b1b2f"
BLUE = "#3b82f6"
TEAL = "#14b8a6"
PURPLE = "#a855f7"
PINK = "#ec4899"
ORANGE = "#f97316"
YELLOW = "#facc15"
GREEN = "#22c55e"
GRAY = "#94a3b8"
WHITE_T = "#e2e8f0"

config.background_color = BG_COLOR
config.pixel_width = 1920
config.pixel_height = 1080


# ════════════════════════════════════════════════════════════════════
# SCENE 1 — Title & Motivation
# ════════════════════════════════════════════════════════════════════
class Scene01_Title(Scene):
    def construct(self):
        title = Text("Vision-Language-Action Models", font_size=56, color=WHITE_T)
        subtitle = Text("How Robots Learn to See, Understand, and Act", font_size=32, color=TEAL)
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1.5)
        self.wait(1)

        # Three icons representing V, L, A
        eye = self._make_icon("👁", "Vision", BLUE, LEFT * 4)
        brain = self._make_icon("🧠", "Language", PURPLE, ORIGIN)
        hand = self._make_icon("🤖", "Action", ORANGE, RIGHT * 4)

        self.play(
            FadeOut(title, shift=UP),
            FadeOut(subtitle, shift=UP),
        )

        self.play(
            LaggedStart(
                FadeIn(eye, shift=UP * 0.5),
                FadeIn(brain, shift=UP * 0.5),
                FadeIn(hand, shift=UP * 0.5),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        # Arrows connecting them
        arr1 = Arrow(eye.get_right(), brain.get_left(), color=GRAY, buff=0.3)
        arr2 = Arrow(brain.get_right(), hand.get_left(), color=GRAY, buff=0.3)
        self.play(GrowArrow(arr1), GrowArrow(arr2))
        self.wait(1)

        # Highlight the key idea
        box = SurroundingRectangle(
            VGroup(eye, brain, hand), color=TEAL, buff=0.5, corner_radius=0.2
        )
        caption = Text(
            "One unified model: pixels in → robot actions out",
            font_size=28,
            color=YELLOW,
        ).next_to(box, DOWN, buff=0.5)
        self.play(Create(box), Write(caption), run_time=2)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _make_icon(self, emoji, label_text, color, pos):
        circle = Circle(radius=0.7, color=color, fill_opacity=0.15, stroke_width=3)
        label = Text(label_text, font_size=26, color=color).next_to(circle, DOWN, buff=0.3)
        icon_text = Text(emoji, font_size=42).move_to(circle)
        grp = VGroup(circle, icon_text, label).move_to(pos)
        return grp


# ════════════════════════════════════════════════════════════════════
# SCENE 2 — The Problem: Why VLAs?
# ════════════════════════════════════════════════════════════════════
class Scene02_WhyVLAs(Scene):
    def construct(self):
        heading = Text("The Problem", font_size=48, color=PINK).to_edge(UP, buff=0.6)
        self.play(Write(heading))

        # Traditional pipeline
        trad_title = Text("Traditional Robotics Pipeline", font_size=28, color=GRAY)
        trad_title.next_to(heading, DOWN, buff=0.6).shift(LEFT * 0.0)

        boxes_data = [
            ("Perception", BLUE),
            ("State\nEstimation", TEAL),
            ("Planning", PURPLE),
            ("Control", ORANGE),
        ]
        boxes = VGroup()
        for i, (txt, col) in enumerate(boxes_data):
            r = RoundedRectangle(
                width=2.2, height=1.2, corner_radius=0.15,
                color=col, fill_opacity=0.15, stroke_width=2,
            )
            t = Text(txt, font_size=20, color=col).move_to(r)
            boxes.add(VGroup(r, t))
        boxes.arrange(RIGHT, buff=0.6)
        boxes.next_to(trad_title, DOWN, buff=0.5)

        self.play(Write(trad_title))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in boxes], lag_ratio=0.2))

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                color=GRAY, buff=0.15, stroke_width=2,
            )
            arrows.add(a)
        self.play(*[GrowArrow(a) for a in arrows])
        self.wait(1)

        # Problems with this
        problems = VGroup(
            Text("✗ Each module hand-engineered", font_size=22, color="#ef4444"),
            Text("✗ Errors cascade between stages", font_size=22, color="#ef4444"),
            Text("✗ Brittle to new environments", font_size=22, color="#ef4444"),
            Text("✗ Can't leverage internet-scale data", font_size=22, color="#ef4444"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        problems.next_to(boxes, DOWN, buff=0.8)

        for p in problems:
            self.play(FadeIn(p, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(1)

        # Transition to VLA
        vla_box = RoundedRectangle(
            width=8, height=1.5, corner_radius=0.2,
            color=TEAL, fill_opacity=0.2, stroke_width=3,
        )
        vla_label = Text("VLA: End-to-End Learning", font_size=30, color=TEAL)
        vla_sub = Text(
            "Image + Language Instruction → Robot Action (directly)",
            font_size=22, color=WHITE_T,
        )
        vla_grp = VGroup(vla_label, vla_sub).arrange(DOWN, buff=0.2).move_to(vla_box)
        vla_full = VGroup(vla_box, vla_grp).next_to(problems, DOWN, buff=0.7)

        self.play(
            FadeIn(vla_box, scale=0.9),
            Write(vla_label),
            FadeIn(vla_sub),
            run_time=2,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 3 — Architecture Deep Dive
# ════════════════════════════════════════════════════════════════════
class Scene03_Architecture(Scene):
    def construct(self):
        heading = Text("VLA Architecture", font_size=48, color=TEAL).to_edge(UP, buff=0.5)
        self.play(Write(heading))
        self.wait(0.5)

        # ── Vision Encoder ──
        ve_box = RoundedRectangle(
            width=2.8, height=3.5, corner_radius=0.15,
            color=BLUE, fill_opacity=0.1, stroke_width=2,
        ).shift(LEFT * 4.5 + DOWN * 0.5)
        ve_title = Text("Vision\nEncoder", font_size=22, color=BLUE).move_to(ve_box.get_top() + DOWN * 0.5)

        # Represent image as pixel grid
        pixel_grid = VGroup()
        for r in range(4):
            for c in range(4):
                sq = Square(side_length=0.28, color=BLUE, fill_opacity=0.05 + 0.06 * (r + c), stroke_width=1)
                sq.move_to(ve_box.get_center() + DOWN * 0.3 + RIGHT * (c - 1.5) * 0.32 + UP * (1.5 - r) * 0.32 + DOWN * 0.5)
                pixel_grid.add(sq)

        vit_label = Text("ViT / SigLIP", font_size=16, color=GRAY).next_to(ve_box, DOWN, buff=0.15)

        # ── Language Encoder ──
        le_box = RoundedRectangle(
            width=2.8, height=3.5, corner_radius=0.15,
            color=PURPLE, fill_opacity=0.1, stroke_width=2,
        ).shift(LEFT * 4.5 + DOWN * 0.5).shift(RIGHT * 3.5)
        le_title = Text("Language\nEncoder", font_size=22, color=PURPLE).move_to(le_box.get_top() + DOWN * 0.5)

        tokens = VGroup()
        words = ["pick", "up", "the", "red", "cup"]
        for i, w in enumerate(words):
            t = Text(w, font_size=16, color=PURPLE)
            t.move_to(le_box.get_center() + DOWN * 0.3 + DOWN * (i - 2) * 0.35)
            tokens.add(t)

        tok_label = Text("Tokenizer + LLM", font_size=16, color=GRAY).next_to(le_box, DOWN, buff=0.15)

        # ── Fusion / Transformer Backbone ──
        bb_box = RoundedRectangle(
            width=3.2, height=3.5, corner_radius=0.15,
            color=TEAL, fill_opacity=0.1, stroke_width=2,
        ).shift(RIGHT * 1.5 + DOWN * 0.5)
        bb_title = Text("Transformer\nBackbone", font_size=22, color=TEAL).move_to(bb_box.get_top() + DOWN * 0.5)

        # Attention pattern visualization
        attn_dots = VGroup()
        for i in range(5):
            for j in range(5):
                opacity = 0.1 + 0.15 * np.exp(-0.5 * ((i - j) ** 2))
                dot = Square(
                    side_length=0.22, fill_opacity=opacity,
                    color=TEAL, stroke_width=0.5, stroke_color=TEAL,
                )
                dot.move_to(bb_box.get_center() + DOWN * 0.3 + RIGHT * (j - 2) * 0.28 + UP * (2 - i) * 0.28 + DOWN * 0.3)
                attn_dots.add(dot)

        attn_label = Text("Cross-Attention", font_size=16, color=GRAY).next_to(bb_box, DOWN, buff=0.15)

        # ── Action Head ──
        ah_box = RoundedRectangle(
            width=2.8, height=3.5, corner_radius=0.15,
            color=ORANGE, fill_opacity=0.1, stroke_width=2,
        ).shift(RIGHT * 5.0 + DOWN * 0.5)
        ah_title = Text("Action\nHead", font_size=22, color=ORANGE).move_to(ah_box.get_top() + DOWN * 0.5)

        action_labels = VGroup()
        dims = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "grip"]
        for i, d in enumerate(dims):
            t = Text(d, font_size=15, color=ORANGE)
            t.move_to(ah_box.get_center() + DOWN * 0.3 + DOWN * (i - 3) * 0.3)
            action_labels.add(t)

        act_label = Text("7-DoF Output", font_size=16, color=GRAY).next_to(ah_box, DOWN, buff=0.15)

        # Animate all four blocks
        all_blocks = [
            (ve_box, ve_title, pixel_grid, vit_label),
            (le_box, le_title, tokens, tok_label),
            (bb_box, bb_title, attn_dots, attn_label),
            (ah_box, ah_title, action_labels, act_label),
        ]

        for box, title, content, label in all_blocks:
            self.play(
                FadeIn(box),
                Write(title),
                FadeIn(content),
                FadeIn(label),
                run_time=1.2,
            )

        # Arrows
        a1 = Arrow(ve_box.get_right(), bb_box.get_left() + UP * 0.5, color=BLUE, buff=0.15, stroke_width=2)
        a2 = Arrow(le_box.get_right(), bb_box.get_left() + DOWN * 0.5, color=PURPLE, buff=0.15, stroke_width=2)
        a3 = Arrow(bb_box.get_right(), ah_box.get_left(), color=TEAL, buff=0.15, stroke_width=2)

        l1 = Text("visual tokens", font_size=14, color=BLUE).next_to(a1, UP, buff=0.08)
        l2 = Text("text tokens", font_size=14, color=PURPLE).next_to(a2, DOWN, buff=0.08)
        l3 = Text("fused repr.", font_size=14, color=TEAL).next_to(a3, UP, buff=0.08)

        self.play(
            GrowArrow(a1), GrowArrow(a2), GrowArrow(a3),
            FadeIn(l1), FadeIn(l2), FadeIn(l3),
            run_time=1.5,
        )
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 4 — Vision Encoder Detail
# ════════════════════════════════════════════════════════════════════
class Scene04_VisionEncoder(Scene):
    def construct(self):
        heading = Text("Vision Encoder: Seeing the World", font_size=42, color=BLUE).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # Image → patches → ViT
        img_rect = Rectangle(width=3, height=3, color=BLUE, fill_opacity=0.15, stroke_width=2)
        img_rect.shift(LEFT * 4.5 + DOWN * 0.5)
        img_label = Text("Camera\nImage", font_size=20, color=WHITE_T).move_to(img_rect)

        self.play(FadeIn(img_rect), Write(img_label))

        # Show patch division
        patches = VGroup()
        colors_list = [BLUE, TEAL, PURPLE, PINK, ORANGE, YELLOW, GREEN, GRAY, BLUE]
        for i in range(3):
            for j in range(3):
                p = Square(
                    side_length=0.95,
                    color=colors_list[i * 3 + j],
                    fill_opacity=0.2,
                    stroke_width=1.5,
                )
                p.move_to(img_rect.get_center() + RIGHT * (j - 1) * 1.0 + UP * (1 - i) * 1.0)
                patches.add(p)

        self.play(
            FadeOut(img_label),
            *[Create(p) for p in patches],
            run_time=1.5,
        )
        patch_label = Text("16×16 patches", font_size=18, color=GRAY).next_to(img_rect, DOWN, buff=0.3)
        self.play(Write(patch_label))

        # Arrow to embedding
        arr1 = Arrow(LEFT * 2.7 + DOWN * 0.5, LEFT * 1.0 + DOWN * 0.5, color=GRAY, buff=0.1)
        self.play(GrowArrow(arr1))

        # Linear projection
        proj_box = RoundedRectangle(
            width=2, height=1.2, corner_radius=0.1,
            color=TEAL, fill_opacity=0.15, stroke_width=2,
        ).move_to(ORIGIN + DOWN * 0.5)
        proj_text = Text("Linear\nProjection", font_size=18, color=TEAL).move_to(proj_box)
        self.play(FadeIn(proj_box), Write(proj_text))

        # Arrow to token sequence
        arr2 = Arrow(RIGHT * 1.0 + DOWN * 0.5, RIGHT * 2.5 + DOWN * 0.5, color=GRAY, buff=0.1)
        self.play(GrowArrow(arr2))

        # Visual tokens
        vtokens = VGroup()
        for i in range(9):
            r = RoundedRectangle(
                width=0.5, height=0.35, corner_radius=0.05,
                color=BLUE, fill_opacity=0.3, stroke_width=1,
            )
            r.move_to(RIGHT * 4.0 + DOWN * 0.5 + DOWN * (i - 4) * 0.42)
            vtokens.add(r)
        vt_label = Text("Visual\nTokens", font_size=18, color=BLUE).next_to(vtokens, RIGHT, buff=0.3)
        self.play(LaggedStart(*[FadeIn(v, shift=LEFT * 0.2) for v in vtokens], lag_ratio=0.08))
        self.play(Write(vt_label))

        # Note about pretrained encoders
        note = Text(
            "Pretrained: SigLIP, DINOv2, or CLIP — frozen or fine-tuned",
            font_size=20, color=YELLOW,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 5 — Language Conditioning
# ════════════════════════════════════════════════════════════════════
class Scene05_Language(Scene):
    def construct(self):
        heading = Text("Language Conditioning", font_size=42, color=PURPLE).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # Instruction
        instr = Text('"Pick up the red cup and place it on the plate"', font_size=24, color=WHITE_T)
        instr.next_to(heading, DOWN, buff=0.7)
        self.play(Write(instr), run_time=2)

        # Tokenization
        arr1 = Arrow(instr.get_bottom(), instr.get_bottom() + DOWN * 1.0, color=GRAY, buff=0.2)
        self.play(GrowArrow(arr1))

        tok_box = RoundedRectangle(
            width=6, height=1.0, corner_radius=0.1,
            color=PURPLE, fill_opacity=0.1, stroke_width=2,
        ).next_to(arr1, DOWN, buff=0.2)
        tok_words = ["pick", "up", "the", "red", "cup", "and", "place", "it", "on", "the", "plate"]
        tok_group = VGroup()
        for i, w in enumerate(tok_words):
            t = Text(w, font_size=16, color=PURPLE)
            tok_group.add(t)
        tok_group.arrange(RIGHT, buff=0.2).move_to(tok_box)
        self.play(FadeIn(tok_box), FadeIn(tok_group))

        # Through LLM
        arr2 = Arrow(tok_box.get_bottom(), tok_box.get_bottom() + DOWN * 1.0, color=GRAY, buff=0.2)
        self.play(GrowArrow(arr2))

        llm_box = RoundedRectangle(
            width=4, height=1.2, corner_radius=0.15,
            color=PURPLE, fill_opacity=0.15, stroke_width=2,
        ).next_to(arr2, DOWN, buff=0.2)
        llm_text = Text("LLM Backbone\n(e.g. Llama 2, PaLM, Gemma)", font_size=18, color=PURPLE).move_to(llm_box)
        self.play(FadeIn(llm_box), Write(llm_text))

        # Output embeddings
        arr3 = Arrow(llm_box.get_bottom(), llm_box.get_bottom() + DOWN * 1.0, color=GRAY, buff=0.2)
        self.play(GrowArrow(arr3))

        emb_label = Text(
            "Contextual language embeddings → fed into transformer backbone",
            font_size=20, color=TEAL,
        ).next_to(arr3, DOWN, buff=0.2)
        self.play(Write(emb_label))

        # Key insight
        insight = Text(
            "Key: The LLM provides semantic understanding of the task",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(insight))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 6 — Action Head / Action Tokenization
# ════════════════════════════════════════════════════════════════════
class Scene06_ActionHead(Scene):
    def construct(self):
        heading = Text("Action Representations", font_size=42, color=ORANGE).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # Two approaches side by side
        # Left: Continuous regression
        left_title = Text("Continuous Regression", font_size=24, color=BLUE).shift(LEFT * 3.5 + UP * 2)
        left_box = RoundedRectangle(
            width=4.5, height=4.5, corner_radius=0.15,
            color=BLUE, fill_opacity=0.05, stroke_width=1.5,
        ).next_to(left_title, DOWN, buff=0.3)

        cont_items = VGroup(
            Text("MLP action head", font_size=18, color=WHITE_T),
            Text("Output: μ and σ per dim", font_size=18, color=WHITE_T),
            Text("Gaussian mixture models", font_size=18, color=WHITE_T),
            Text("Used in: RT-2, Octo", font_size=16, color=GRAY),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(left_box).shift(UP * 0.3)

        # Gaussian curve
        ax_left = Axes(
            x_range=[-3, 3, 1], y_range=[0, 1, 0.5],
            x_length=3.5, y_length=1.5,
            tips=False,
        ).next_to(cont_items, DOWN, buff=0.3)
        gauss = ax_left.plot(lambda x: np.exp(-x**2 / 2), color=BLUE)

        # Right: Discrete tokenization
        right_title = Text("Discrete Tokenization", font_size=24, color=ORANGE).shift(RIGHT * 3.5 + UP * 2)
        right_box = RoundedRectangle(
            width=4.5, height=4.5, corner_radius=0.15,
            color=ORANGE, fill_opacity=0.05, stroke_width=1.5,
        ).next_to(right_title, DOWN, buff=0.3)

        disc_items = VGroup(
            Text("Bin continuous values → tokens", font_size=18, color=WHITE_T),
            Text("256 bins per dimension", font_size=18, color=WHITE_T),
            Text("Autoregressive decoding", font_size=18, color=WHITE_T),
            Text("Used in: RT-2, π₀, OpenVLA", font_size=16, color=GRAY),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(right_box).shift(UP * 0.3)

        # Histogram
        bins = VGroup()
        heights = [0.2, 0.4, 0.8, 1.0, 0.7, 0.3, 0.15]
        for i, h in enumerate(heights):
            bar = Rectangle(
                width=0.4, height=h * 1.2, color=ORANGE,
                fill_opacity=0.4, stroke_width=1,
            )
            bar.move_to(right_box.get_center() + DOWN * 1.2 + RIGHT * (i - 3) * 0.45 + UP * h * 0.6)
            bins.add(bar)

        self.play(
            Write(left_title), FadeIn(left_box),
            Write(right_title), FadeIn(right_box),
        )
        self.play(
            FadeIn(cont_items), FadeIn(disc_items),
            Create(ax_left), Create(gauss),
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bins], lag_ratio=0.1),
            run_time=2,
        )

        # Action dimensions
        dim_label = Text(
            "7-DoF: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(dim_label))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 7 — Training Pipeline
# ════════════════════════════════════════════════════════════════════
class Scene07_Training(Scene):
    def construct(self):
        heading = Text("How VLAs Are Trained", font_size=48, color=GREEN).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # Stage 1: Pre-training
        s1_title = Text("Stage 1: Vision-Language Pre-training", font_size=26, color=BLUE)
        s1_title.next_to(heading, DOWN, buff=0.6).to_edge(LEFT, buff=1)

        s1_items = VGroup(
            Text("• Start with pretrained VLM (PaLM-E, Llama)", font_size=20, color=WHITE_T),
            Text("• Trained on billions of image-text pairs", font_size=20, color=WHITE_T),
            Text("• Learns visual understanding + reasoning", font_size=20, color=WHITE_T),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(s1_title, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(s1_title))
        for item in s1_items:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)

        # Arrow
        down_arr1 = Arrow(
            s1_items.get_bottom() + DOWN * 0.1,
            s1_items.get_bottom() + DOWN * 0.8,
            color=GRAY, buff=0.1,
        )
        self.play(GrowArrow(down_arr1))

        # Stage 2: Robot data fine-tuning
        s2_title = Text("Stage 2: Robot Action Fine-tuning", font_size=26, color=TEAL)
        s2_title.next_to(down_arr1, DOWN, buff=0.2).align_to(s1_title, LEFT)

        s2_items = VGroup(
            Text("• Add action head to the VLM", font_size=20, color=WHITE_T),
            Text("• Train on robot demonstration datasets", font_size=20, color=WHITE_T),
            Text("• Data: (image, instruction, action) tuples", font_size=20, color=WHITE_T),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(s2_title, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(s2_title))
        for item in s2_items:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)

        # Right side: Data sources
        data_title = Text("Key Datasets", font_size=24, color=YELLOW).shift(RIGHT * 3.5 + UP * 1.5)
        datasets = VGroup(
            Text("Open X-Embodiment (1M+ episodes)", font_size=18, color=WHITE_T),
            Text("Bridge V2 (60k demos, 13 skills)", font_size=18, color=WHITE_T),
            Text("RT-1/RT-2 data (130k episodes)", font_size=18, color=WHITE_T),
            Text("DROID (76k episodes, 564 scenes)", font_size=18, color=WHITE_T),
            Text("+ In-house teleoperation data", font_size=18, color=GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(data_title, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(data_title))
        self.play(LaggedStart(*[FadeIn(d) for d in datasets], lag_ratio=0.15))

        # Loss function note
        loss_note = Text(
            "Loss = next-token prediction on action tokens (cross-entropy)",
            font_size=20, color=PINK,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(loss_note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 8 — Inference: How it runs on a robot
# ════════════════════════════════════════════════════════════════════
class Scene08_Inference(Scene):
    def construct(self):
        heading = Text("Inference: Robot in Action", font_size=42, color=TEAL).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # Control loop diagram
        steps = [
            ("1. Capture\nImage", BLUE, LEFT * 4.5),
            ("2. Tokenize\nImage + Text", PURPLE, LEFT * 1.5),
            ("3. Forward\nPass", TEAL, RIGHT * 1.5),
            ("4. Decode\nAction", ORANGE, RIGHT * 4.5),
        ]

        boxes = VGroup()
        for text, color, pos in steps:
            r = RoundedRectangle(
                width=2.5, height=1.5, corner_radius=0.15,
                color=color, fill_opacity=0.15, stroke_width=2,
            ).move_to(pos + DOWN * 0.3)
            t = Text(text, font_size=18, color=color).move_to(r)
            boxes.add(VGroup(r, t))

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in boxes], lag_ratio=0.2))

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                color=GRAY, buff=0.1, stroke_width=2,
            )
            arrows.add(a)
        self.play(*[GrowArrow(a) for a in arrows])

        # Feedback loop arrow
        feedback = CurvedArrow(
            boxes[-1].get_bottom() + DOWN * 0.2,
            boxes[0].get_bottom() + DOWN * 0.2,
            angle=-TAU / 4,
            color=YELLOW,
            stroke_width=2,
        )
        fb_label = Text("5. Execute → next frame", font_size=18, color=YELLOW)
        fb_label.next_to(feedback, DOWN, buff=0.2)

        self.play(Create(feedback), Write(fb_label))

        # Timing info
        timing = VGroup(
            Text("Typical frequency: 5-10 Hz", font_size=22, color=WHITE_T),
            Text("Latency: ~100-200ms per step", font_size=22, color=WHITE_T),
            Text("Action chunking: predict k steps ahead", font_size=22, color=GREEN),
        ).arrange(DOWN, buff=0.25).shift(DOWN * 2.5)

        self.play(FadeIn(timing))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 9 — Key Models
# ════════════════════════════════════════════════════════════════════
class Scene09_KeyModels(Scene):
    def construct(self):
        heading = Text("Landmark VLA Models", font_size=42, color=PURPLE).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        models = [
            ("RT-2 (2023)", "Google DeepMind", "PaLM-E + action tokens\n55B params, co-fine-tuned on robot data", BLUE),
            ("Octo (2024)", "UC Berkeley", "Diffusion action head, multi-robot\nOpen-source, transformer-based", TEAL),
            ("OpenVLA (2024)", "Stanford / TRI", "Llama 2 7B + SigLIP encoder\nOpen-source, fine-tunable", GREEN),
            ("π₀ (2024)", "Physical Intelligence", "Flow matching action head\n3B params, pre-trained on internet data", ORANGE),
            ("Gemini Robotics (2025)", "Google DeepMind", "Gemini 2.0 backbone, ASIC action head\nDexterous manipulation", PINK),
        ]

        cards = VGroup()
        for name, org, desc, color in models:
            card = RoundedRectangle(
                width=11, height=0.9, corner_radius=0.1,
                color=color, fill_opacity=0.08, stroke_width=1.5,
            )
            name_t = Text(name, font_size=20, color=color, weight=BOLD).move_to(card.get_left() + RIGHT * 1.5)
            org_t = Text(org, font_size=16, color=GRAY).move_to(card.get_left() + RIGHT * 4.0)
            desc_t = Text(desc.split('\n')[0], font_size=15, color=WHITE_T).move_to(card.get_left() + RIGHT * 8.0)
            cards.add(VGroup(card, name_t, org_t, desc_t))

        cards.arrange(DOWN, buff=0.15).next_to(heading, DOWN, buff=0.5)

        self.play(
            LaggedStart(*[FadeIn(c, shift=RIGHT * 0.3) for c in cards], lag_ratio=0.2),
            run_time=3,
        )
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 10 — Limitations
# ════════════════════════════════════════════════════════════════════
class Scene10_Limitations(Scene):
    def construct(self):
        heading = Text("Current Limitations", font_size=48, color="#ef4444").to_edge(UP, buff=0.5)
        self.play(Write(heading))

        limitations = [
            ("Data Hungry", "Need massive robot demonstration datasets\nReal robot data is expensive and slow to collect", BLUE),
            ("Generalization Gap", "Struggle with unseen objects, environments,\nand tasks not in training distribution", PURPLE),
            ("Latency", "Large models (7B+) are slow for real-time control\nAction chunking helps but adds complexity", ORANGE),
            ("Safety", "No built-in safety guarantees\nCan't verify actions before execution", "#ef4444"),
            ("Sim-to-Real", "Simulation training doesn't transfer perfectly\nVisual and dynamics domain gaps persist", TEAL),
            ("Long-Horizon Tasks", "Struggle with multi-step tasks requiring\nplanning over many minutes", PINK),
        ]

        items = VGroup()
        for title, desc, color in limitations:
            title_t = Text(title, font_size=22, color=color, weight=BOLD)
            desc_lines = desc.split('\n')
            desc_t = Text(desc_lines[0], font_size=17, color=GRAY)
            row = VGroup(title_t, desc_t).arrange(RIGHT, buff=0.5, aligned_edge=UP)
            items.add(row)

        items.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        items.next_to(heading, DOWN, buff=0.5).to_edge(LEFT, buff=1.0)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.7)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 11 — New Research Directions
# ════════════════════════════════════════════════════════════════════
class Scene11_Research(Scene):
    def construct(self):
        heading = Text("Frontier Research Directions", font_size=42, color=GREEN).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        topics = [
            ("Flow Matching Actions", "Replace discrete tokens with continuous\nflow-based action generation (π₀)", BLUE, LEFT * 3.5),
            ("Diffusion Policies", "Model action distributions with\ndenoising diffusion (Octo, DP)", TEAL, RIGHT * 3.5),
            ("Scaling Laws", "Does more data + bigger model =\nbetter robot? Early evidence says yes", PURPLE, LEFT * 3.5),
            ("Sim-to-Real Transfer", "Train in simulation, deploy in real world\nDomain randomization + adaptation", ORANGE, RIGHT * 3.5),
            ("Multi-Embodiment", "One model controlling different robots\nOpen X-Embodiment initiative", PINK, LEFT * 3.5),
            ("World Models", "Learn physics + dynamics, plan in\nimagination before acting", YELLOW, RIGHT * 3.5),
        ]

        cards = VGroup()
        y_offset = 1.2
        for i, (title, desc, color, x_pos) in enumerate(topics):
            row = i // 2
            card = RoundedRectangle(
                width=5.5, height=1.6, corner_radius=0.15,
                color=color, fill_opacity=0.1, stroke_width=2,
            )
            t = Text(title, font_size=20, color=color, weight=BOLD)
            d = Text(desc.split('\n')[0], font_size=16, color=WHITE_T)
            d2 = Text(desc.split('\n')[1] if '\n' in desc else "", font_size=16, color=GRAY)
            content = VGroup(t, d, d2).arrange(DOWN, buff=0.12).move_to(card)
            grp = VGroup(card, content).move_to(x_pos + DOWN * (row * 1.8 + y_offset))
            cards.add(grp)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in cards], lag_ratio=0.15),
            run_time=3,
        )
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 12 — Summary & Closing
# ════════════════════════════════════════════════════════════════════
class Scene12_Summary(Scene):
    def construct(self):
        heading = Text("VLAs: The Big Picture", font_size=48, color=TEAL).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        summary_points = [
            "VLAs unify vision, language, and action into one model",
            "Built on foundation model backbones (LLMs + Vision Encoders)",
            "Trained on large-scale robot demonstration data",
            "Enable language-conditioned robotic manipulation",
            "Still limited by data, latency, and generalization",
            "Rapid progress: new architectures every few months",
        ]

        items = VGroup()
        for i, point in enumerate(summary_points):
            bullet = Text(f"→  {point}", font_size=22, color=WHITE_T)
            items.add(bullet)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        items.next_to(heading, DOWN, buff=0.7).to_edge(LEFT, buff=1.0)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(1)

        # Final quote
        quote = Text(
            '"The next generation of robots will be\nprogrammed with words, not code."',
            font_size=26, color=YELLOW, slant=ITALIC,
        ).to_edge(DOWN, buff=1.0)
        self.play(FadeIn(quote, shift=UP * 0.3))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # End card
        thanks = Text("Thanks for watching!", font_size=48, color=TEAL)
        self.play(Write(thanks))
        self.wait(2)
        self.play(FadeOut(thanks))
