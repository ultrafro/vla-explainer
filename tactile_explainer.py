"""
Tactile Sensing + VLA Fusion Explainer
3Blue1Brown style using Manim CE — with paper screenshots

Covers:
1. Why Touch Matters
2. Tactile Sensor Types (GelSight, DIGIT, Piezo arrays)
3. 3D-ViTac: Visuo-Tactile 3D Fusion
4. Sparsh: Foundation Model for Touch
5. TANDEM: Active Tactile Exploration
6. TacSL: Simulation for Tactile Learning
7. Fusion Architectures Deep Dive
8. How Tactile Tokens Enter a VLA
9. Training with Tactile Data
10. Limitations & Frontier Research
11. Summary
"""

from manim import *
import numpy as np

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
RED = "#ef4444"

config.background_color = BG_COLOR
config.pixel_width = 1920
config.pixel_height = 1080

FIGURES = "figures/"


# ════════════════════════════════════════════════════════════════════
# SCENE 1 — Title
# ════════════════════════════════════════════════════════════════════
class S01_Title(Scene):
    def construct(self):
        title = Text("Tactile Sensing for Robot Learning", font_size=52, color=WHITE_T)
        subtitle = Text(
            "How Robots Learn to Feel — and Why It Changes Everything",
            font_size=28, color=TEAL,
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1.5)
        self.wait(1)

        # Three modalities
        icons = VGroup()
        for label, color, pos in [
            ("Vision", BLUE, LEFT * 4),
            ("Touch", ORANGE, ORIGIN),
            ("Action", GREEN, RIGHT * 4),
        ]:
            c = Circle(radius=0.7, color=color, fill_opacity=0.15, stroke_width=3)
            t = Text(label, font_size=24, color=color).next_to(c, DOWN, buff=0.3)
            icons.add(VGroup(c, t).move_to(pos + DOWN * 0.5))

        self.play(FadeOut(title, shift=UP), FadeOut(subtitle, shift=UP))
        self.play(LaggedStart(*[FadeIn(i, shift=UP * 0.4) for i in icons], lag_ratio=0.25))

        # Plus signs and fusion
        plus1 = Text("+", font_size=48, color=GRAY).move_to(LEFT * 2 + DOWN * 0.5)
        plus2 = Text("+", font_size=48, color=GRAY).move_to(RIGHT * 2 + DOWN * 0.5)
        self.play(FadeIn(plus1), FadeIn(plus2))

        box = SurroundingRectangle(icons, color=TEAL, buff=0.6, corner_radius=0.2)
        cap = Text(
            "Multimodal fusion: the key to dexterous manipulation",
            font_size=24, color=YELLOW,
        ).next_to(box, DOWN, buff=0.5)
        self.play(Create(box), Write(cap), run_time=2)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 2 — Why Touch Matters
# ════════════════════════════════════════════════════════════════════
class S02_WhyTouch(Scene):
    def construct(self):
        heading = Text("Why Touch Matters", font_size=48, color=ORANGE).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # Left: Vision limitations
        vis_title = Text("Vision Alone Is Not Enough", font_size=26, color=BLUE)
        vis_title.next_to(heading, DOWN, buff=0.6).shift(LEFT * 3)

        vis_problems = VGroup(
            Text("• Can't sense contact forces", font_size=20, color=WHITE_T),
            Text("• Occlusion during grasping", font_size=20, color=WHITE_T),
            Text("• No slip detection", font_size=20, color=WHITE_T),
            Text("• Can't feel object stiffness", font_size=20, color=WHITE_T),
            Text("• Fragile objects break without\n  force feedback", font_size=20, color=WHITE_T),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        vis_problems.next_to(vis_title, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(vis_title))
        for p in vis_problems:
            self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)

        # Right: What touch provides
        touch_title = Text("What Touch Provides", font_size=26, color=ORANGE)
        touch_title.next_to(heading, DOWN, buff=0.6).shift(RIGHT * 3)

        touch_items = VGroup(
            Text("✓ Contact geometry & location", font_size=20, color=GREEN),
            Text("✓ Normal & shear forces", font_size=20, color=GREEN),
            Text("✓ Slip detection in real-time", font_size=20, color=GREEN),
            Text("✓ Material/texture recognition", font_size=20, color=GREEN),
            Text("✓ Precise force control for\n  delicate objects", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        touch_items.next_to(touch_title, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(touch_title))
        for t in touch_items:
            self.play(FadeIn(t, shift=RIGHT * 0.2), run_time=0.5)

        # Key insight
        insight = Text(
            "\"Vision tells you WHAT to grasp. Touch tells you HOW to grasp.\"",
            font_size=22, color=YELLOW, slant=ITALIC,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(insight))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 3 — Sensor Types
# ════════════════════════════════════════════════════════════════════
class S03_Sensors(Scene):
    def construct(self):
        heading = Text("Tactile Sensor Technologies", font_size=42, color=TEAL).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        sensors = [
            ("Optical (GelSight/DIGIT)", BLUE,
             "Camera behind soft gel membrane\nDeformation → RGB image → geometry\nRes: 640×480, ~30 Hz",
             "High-res contact geometry"),
            ("Piezoelectric Arrays", ORANGE,
             "Pressure-sensitive grid on fingertip\n16×16 sensing units per finger pad\nRes: 256 taxels, ~32 Hz",
             "Used in 3D-ViTac"),
            ("Capacitive (RoboSkin)", PURPLE,
             "Flexible skin over robot surface\nMeasures proximity + contact\nLarge area coverage",
             "Whole-body sensing"),
            ("BioTac (SynTouch)", GREEN,
             "Multimodal: force + vibration + temp\n19 impedance electrodes + hydrophone\nClosest to human fingertip",
             "Gold standard, expensive"),
        ]

        cards = VGroup()
        for name, color, desc, note in sensors:
            card = RoundedRectangle(
                width=5.5, height=2.4, corner_radius=0.15,
                color=color, fill_opacity=0.08, stroke_width=2,
            )
            n = Text(name, font_size=20, color=color, weight=BOLD)
            d = Text(desc.split('\n')[0], font_size=16, color=WHITE_T)
            d2 = Text(desc.split('\n')[1], font_size=16, color=WHITE_T)
            d3 = Text(desc.split('\n')[2], font_size=14, color=GRAY)
            nt = Text(note, font_size=14, color=YELLOW)
            content = VGroup(n, d, d2, d3, nt).arrange(DOWN, buff=0.12).move_to(card)
            cards.add(VGroup(card, content))

        cards.arrange_in_grid(rows=2, cols=2, buff=0.3)
        cards.next_to(heading, DOWN, buff=0.4)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in cards], lag_ratio=0.2),
            run_time=3,
        )
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 4 — 3D-ViTac Paper
# ════════════════════════════════════════════════════════════════════
class S04_3DViTac(Scene):
    def construct(self):
        heading = Text("3D-ViTac: Visuo-Tactile 3D Fusion", font_size=40, color=BLUE).to_edge(UP, buff=0.4)
        cite = Text("Huang et al., CoRL 2024 — Columbia / UIUC", font_size=18, color=GRAY)
        cite.next_to(heading, DOWN, buff=0.15)
        self.play(Write(heading), FadeIn(cite))

        # Show paper figure - robot setup
        try:
            robot_img = ImageMobject(FIGURES + "3d_vitac_robot.png")
            robot_img.height = 3.5
            robot_img.shift(LEFT * 3.5 + DOWN * 1.0)
            robot_border = SurroundingRectangle(robot_img, color=BLUE, buff=0.05, stroke_width=1)
            robot_label = Text("Bimanual setup with tactile sensor pads", font_size=16, color=GRAY)
            robot_label.next_to(robot_img, DOWN, buff=0.15)
            self.play(FadeIn(robot_img), Create(robot_border), Write(robot_label))
        except Exception:
            pass

        # Key innovation text on right
        key_points = VGroup(
            Text("Key Innovation:", font_size=22, color=YELLOW, weight=BOLD),
            Text("Fuse vision + touch in 3D space", font_size=20, color=WHITE_T),
            Text("", font_size=10),
            Text("Tactile Hardware:", font_size=20, color=ORANGE, weight=BOLD),
            Text("• Piezoelectric sensor pads", font_size=18, color=WHITE_T),
            Text("• 16×16 = 256 taxels per finger", font_size=18, color=WHITE_T),
            Text("• 1024 total sensing units (bimanual)", font_size=18, color=WHITE_T),
            Text("• ~32 Hz, cost < $20 per pad", font_size=18, color=WHITE_T),
            Text("", font_size=10),
            Text("3D Fusion:", font_size=20, color=TEAL, weight=BOLD),
            Text("• Visual point cloud from RGB-D", font_size=18, color=WHITE_T),
            Text("• Tactile point cloud from sensor grid", font_size=18, color=WHITE_T),
            Text("• Merge into unified 3D representation", font_size=18, color=WHITE_T),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        key_points.shift(RIGHT * 2.5 + DOWN * 0.8)

        self.play(FadeIn(key_points), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 5 — 3D-ViTac Architecture
# ════════════════════════════════════════════════════════════════════
class S05_3DViTacArch(Scene):
    def construct(self):
        heading = Text("3D-ViTac: Architecture Deep Dive", font_size=38, color=BLUE).to_edge(UP, buff=0.4)
        self.play(Write(heading))

        # Show paper architecture figure
        try:
            arch_img = ImageMobject(FIGURES + "3d_vitac_arch.png")
            arch_img.height = 4.0
            arch_img.shift(UP * 0.2)
            arch_border = SurroundingRectangle(arch_img, color=BLUE, buff=0.05, stroke_width=1)
            fig_label = Text("Figure 3 from 3D-ViTac paper — Visuo-Tactile Policy Pipeline", font_size=16, color=GRAY)
            fig_label.next_to(arch_img, DOWN, buff=0.15)
            self.play(FadeIn(arch_img), Create(arch_border), Write(fig_label))
        except Exception:
            pass

        self.wait(2)

        # Annotate the key parts
        annotations = VGroup(
            Text("① RGB-D → 3D Visual Point Cloud (Nvis = 512 pts)", font_size=18, color=BLUE),
            Text("② Tactile readings → 3D Tactile Point Cloud (Ntac = 256 pts)", font_size=18, color=ORANGE),
            Text("③ Merge → Unified 3D Visuo-Tactile Representation", font_size=18, color=TEAL),
            Text("④ PointNet++ backbone → Diffusion Policy head → Actions", font_size=18, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        annotations.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(annotations), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 6 — 3D-ViTac Fusion Detail (animated)
# ════════════════════════════════════════════════════════════════════
class S06_FusionDetail(Scene):
    def construct(self):
        heading = Text("3D-ViTac: The Fusion Mechanism", font_size=38, color=TEAL).to_edge(UP, buff=0.4)
        self.play(Write(heading))

        # Visual pipeline (left)
        vis_box = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.1, color=BLUE, fill_opacity=0.15)
        vis_box.shift(LEFT * 4.5 + UP * 1.5)
        vis_text = Text("RGB-D Camera", font_size=18, color=BLUE).move_to(vis_box)

        vis_proc = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.1, color=BLUE, fill_opacity=0.1)
        vis_proc.next_to(vis_box, DOWN, buff=0.5)
        vis_proc_t = Text("FPS Downsample\n→ Nvis = 512 pts", font_size=16, color=BLUE).move_to(vis_proc)

        vis_out = RoundedRectangle(width=3.0, height=0.8, corner_radius=0.1, color=BLUE, fill_opacity=0.1)
        vis_out.next_to(vis_proc, DOWN, buff=0.5)
        vis_out_t = Text("P_vis ∈ ℝ^(512×3)", font_size=18, color=BLUE, weight=BOLD).move_to(vis_out)

        # Tactile pipeline (right)
        tac_box = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.1, color=ORANGE, fill_opacity=0.15)
        tac_box.shift(RIGHT * 4.5 + UP * 1.5)
        tac_text = Text("Tactile Sensor Pads", font_size=18, color=ORANGE).move_to(tac_box)

        tac_proc = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.1, color=ORANGE, fill_opacity=0.1)
        tac_proc.next_to(tac_box, DOWN, buff=0.5)
        tac_proc_t = Text("FK to 3D coords\n+ extra pressure ch", font_size=16, color=ORANGE).move_to(tac_proc)

        tac_out = RoundedRectangle(width=3.0, height=0.8, corner_radius=0.1, color=ORANGE, fill_opacity=0.1)
        tac_out.next_to(tac_proc, DOWN, buff=0.5)
        tac_out_t = Text("P_tac ∈ ℝ^(256×4)", font_size=18, color=ORANGE, weight=BOLD).move_to(tac_out)

        # Fusion in center
        merge_box = RoundedRectangle(width=4.0, height=1.2, corner_radius=0.15, color=TEAL, fill_opacity=0.2, stroke_width=3)
        merge_box.shift(DOWN * 2.2)
        merge_t = Text("Concatenate → PointNet++ Backbone", font_size=18, color=TEAL, weight=BOLD).move_to(merge_box)

        dim_label = Text(
            "P_combined ∈ ℝ^(768 × 4)  [512 visual + 256 tactile points]",
            font_size=16, color=YELLOW,
        ).next_to(merge_box, DOWN, buff=0.2)

        # Animate
        self.play(FadeIn(vis_box), Write(vis_text), FadeIn(tac_box), Write(tac_text))

        a1 = Arrow(vis_box.get_bottom(), vis_proc.get_top(), color=GRAY, buff=0.1)
        a2 = Arrow(tac_box.get_bottom(), tac_proc.get_top(), color=GRAY, buff=0.1)
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(vis_proc), Write(vis_proc_t), FadeIn(tac_proc), Write(tac_proc_t))

        a3 = Arrow(vis_proc.get_bottom(), vis_out.get_top(), color=GRAY, buff=0.1)
        a4 = Arrow(tac_proc.get_bottom(), tac_out.get_top(), color=GRAY, buff=0.1)
        self.play(GrowArrow(a3), GrowArrow(a4), FadeIn(vis_out), Write(vis_out_t), FadeIn(tac_out), Write(tac_out_t))

        a5 = Arrow(vis_out.get_bottom(), merge_box.get_left() + UP * 0.1, color=BLUE, buff=0.1)
        a6 = Arrow(tac_out.get_bottom(), merge_box.get_right() + UP * 0.1, color=ORANGE, buff=0.1)
        self.play(GrowArrow(a5), GrowArrow(a6), FadeIn(merge_box), Write(merge_t))
        self.play(FadeIn(dim_label))

        # Output
        out_label = Text(
            "→ Conditioned Diffusion Policy → 7-DoF Actions",
            font_size=18, color=GREEN,
        ).next_to(dim_label, DOWN, buff=0.2)
        self.play(FadeIn(out_label))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 7 — Sparsh: Foundation Model for Touch
# ════════════════════════════════════════════════════════════════════
class S07_Sparsh(Scene):
    def construct(self):
        heading = Text("Sparsh: A Foundation Model for Touch", font_size=40, color=PURPLE).to_edge(UP, buff=0.4)
        cite = Text("Higuera et al., 2024 — Meta FAIR", font_size=18, color=GRAY)
        cite.next_to(heading, DOWN, buff=0.15)
        self.play(Write(heading), FadeIn(cite))

        # Paper figure
        try:
            arch_img = ImageMobject(FIGURES + "sparsh_arch.png")
            arch_img.height = 3.2
            arch_img.shift(UP * 0.0)
            arch_border = SurroundingRectangle(arch_img, color=PURPLE, buff=0.05, stroke_width=1)
            fig_label = Text("Sparsh model family — SSL pre-training on 460k+ tactile images", font_size=16, color=GRAY)
            fig_label.next_to(arch_img, DOWN, buff=0.15)
            self.play(FadeIn(arch_img), Create(arch_border), Write(fig_label))
        except Exception:
            pass

        self.wait(2)

        # Key facts
        facts = VGroup(
            Text("• Self-supervised learning (MAE, DINO, JEPA) on tactile images", font_size=18, color=WHITE_T),
            Text("• Pre-trained on 460k+ images from GelSight, DIGIT, GelSight Mini", font_size=18, color=WHITE_T),
            Text("• Generalizes across sensor types without task-specific labels", font_size=18, color=WHITE_T),
            Text("• TacBench: standardized benchmark for tactile representations", font_size=18, color=TEAL),
            Text("• Sparsh (DINO) + Sparsh (JEPA) = best performing models", font_size=18, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        facts.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(facts), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 8 — TANDEM: Active Tactile Exploration
# ════════════════════════════════════════════════════════════════════
class S08_TANDEM(Scene):
    def construct(self):
        heading = Text("TANDEM: Active Tactile Exploration", font_size=40, color=GREEN).to_edge(UP, buff=0.4)
        cite = Text("Xu, Song & Ciocarlie, 2022 — Columbia / Shuran Song Lab", font_size=18, color=GRAY)
        cite.next_to(heading, DOWN, buff=0.15)
        self.play(Write(heading), FadeIn(cite))

        # Paper figures side by side
        try:
            setup_img = ImageMobject(FIGURES + "tandem_setup.png")
            setup_img.height = 3.0
            setup_img.shift(LEFT * 4.0 + DOWN * 0.5)
            setup_border = SurroundingRectangle(setup_img, color=GREEN, buff=0.05, stroke_width=1)
            setup_label = Text("Active exploration setup", font_size=14, color=GRAY)
            setup_label.next_to(setup_img, DOWN, buff=0.1)
            self.play(FadeIn(setup_img), Create(setup_border), Write(setup_label))
        except Exception:
            pass

        try:
            arch_img = ImageMobject(FIGURES + "tandem_arch.png")
            arch_img.height = 2.5
            arch_img.shift(RIGHT * 2.0 + UP * 0.2)
            arch_border = SurroundingRectangle(arch_img, color=GREEN, buff=0.05, stroke_width=1)
            arch_label = Text("Explorer + Discriminator co-training architecture", font_size=14, color=GRAY)
            arch_label.next_to(arch_img, DOWN, buff=0.1)
            self.play(FadeIn(arch_img), Create(arch_border), Write(arch_label))
        except Exception:
            pass

        # Key concept
        concept = VGroup(
            Text("Core Idea: Robot decides WHERE to touch for maximum information", font_size=20, color=YELLOW),
            Text("Explorer: generates touch actions → Discriminator: identifies objects", font_size=18, color=WHITE_T),
            Text("Co-training: explorer learns to provide useful data to discriminator", font_size=18, color=WHITE_T),
            Text("Outperforms random exploration — robust to sensor noise", font_size=18, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        concept.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(concept), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 9 — TacSL: Tactile Simulation
# ════════════════════════════════════════════════════════════════════
class S09_TacSL(Scene):
    def construct(self):
        heading = Text("TacSL: Simulating Touch at Scale", font_size=40, color=PINK).to_edge(UP, buff=0.4)
        cite = Text("Akinola et al., 2024 — NVIDIA / U. Washington", font_size=18, color=GRAY)
        cite.next_to(heading, DOWN, buff=0.15)
        self.play(Write(heading), FadeIn(cite))

        # Paper figure
        try:
            arch_img = ImageMobject(FIGURES + "tacsl_arch.png")
            arch_img.height = 3.2
            arch_img.shift(UP * 0.0)
            arch_border = SurroundingRectangle(arch_img, color=PINK, buff=0.05, stroke_width=1)
            fig_label = Text("TacSL: GPU-accelerated tactile simulation + policy learning toolkit", font_size=16, color=GRAY)
            fig_label.next_to(arch_img, DOWN, buff=0.15)
            self.play(FadeIn(arch_img), Create(arch_border), Write(fig_label))
        except Exception:
            pass

        self.wait(1.5)

        facts = VGroup(
            Text("• GPU-based FEM simulation of GelSight/DIGIT sensors", font_size=18, color=WHITE_T),
            Text("• Generates RGB tactile images + force fields at 350× faster than prior state of the art", font_size=18, color=WHITE_T),
            Text("• Integrated into Isaac Sim for sim-to-real transfer", font_size=18, color=WHITE_T),
            Text("• ACOCD: novel policy learning algorithm with active critic distillation", font_size=18, color=TEAL),
            Text("• Enables large-scale tactile data collection without real robots", font_size=18, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        facts.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(facts), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 10 — Fusion Architecture Taxonomy
# ════════════════════════════════════════════════════════════════════
class S10_FusionTypes(Scene):
    def construct(self):
        heading = Text("Tactile-Visual Fusion: 4 Architectures", font_size=40, color=TEAL).to_edge(UP, buff=0.4)
        self.play(Write(heading))

        # ── Type 1: Early Concatenation ──
        t1 = self._make_fusion_card(
            "1. Early Concatenation",
            ["Image features [B, 512]", "Tactile features [B, 256]", "→ Concat → [B, 768]", "→ MLP → Action"],
            BLUE, LEFT * 4.0 + UP * 0.8,
        )

        # ── Type 2: Cross-Attention ──
        t2 = self._make_fusion_card(
            "2. Cross-Attention Fusion",
            ["Visual tokens: Q", "Tactile tokens: K, V", "→ Attention(Q, K, V)", "→ Fused tokens → Policy"],
            PURPLE, RIGHT * 4.0 + UP * 0.8,
        )

        # ── Type 3: 3D Point Cloud ──
        t3 = self._make_fusion_card(
            "3. 3D Spatial Fusion (3D-ViTac)",
            ["Visual pts ∈ ℝ^(512×3)", "Tactile pts ∈ ℝ^(256×4)", "→ Merge in 3D → PointNet++", "→ Diffusion → Action"],
            TEAL, LEFT * 4.0 + DOWN * 2.2,
        )

        # ── Type 4: Tokenize-and-Interleave ──
        t4 = self._make_fusion_card(
            "4. Token Interleaving (VLA-style)",
            ["[IMG][IMG]...[TAC][TAC]...[TXT]", "All as tokens in one sequence", "→ Transformer self-attn", "→ Action token decoding"],
            ORANGE, RIGHT * 4.0 + DOWN * 2.2,
        )

        self.play(
            LaggedStart(FadeIn(t1, scale=0.9), FadeIn(t2, scale=0.9),
                        FadeIn(t3, scale=0.9), FadeIn(t4, scale=0.9), lag_ratio=0.2),
            run_time=3,
        )

        # Highlight the trend
        trend = Text(
            "Trend: moving from simple concat → unified token sequences in transformers",
            font_size=20, color=YELLOW,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(trend))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _make_fusion_card(self, title, lines, color, pos):
        card = RoundedRectangle(
            width=5.2, height=2.4, corner_radius=0.15,
            color=color, fill_opacity=0.08, stroke_width=2,
        )
        t = Text(title, font_size=18, color=color, weight=BOLD)
        items = VGroup(*[Text(l, font_size=15, color=WHITE_T) for l in lines])
        items.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        content = VGroup(t, items).arrange(DOWN, buff=0.15).move_to(card)
        return VGroup(card, content).move_to(pos)


# ════════════════════════════════════════════════════════════════════
# SCENE 11 — How Tactile Enters a VLA (detailed)
# ════════════════════════════════════════════════════════════════════
class S11_TactileInVLA(Scene):
    def construct(self):
        heading = Text("How Tactile Data Enters a VLA", font_size=40, color=ORANGE).to_edge(UP, buff=0.4)
        self.play(Write(heading))

        # Full pipeline
        y_top = 1.8

        # Camera input
        cam = self._block("Camera\nImage", BLUE, LEFT * 5.5, y_top, w=2.0)
        # Tactile input
        tac = self._block("Tactile\nSensor", ORANGE, LEFT * 5.5, y_top - 2.0, w=2.0)
        # Language
        lang = self._block("Language\nInstruction", PURPLE, LEFT * 5.5, y_top - 4.0, w=2.0)

        self.play(FadeIn(cam), FadeIn(tac), FadeIn(lang))

        # Encoders
        vis_enc = self._block("ViT\nEncoder", BLUE, LEFT * 2.5, y_top, w=2.0)
        tac_enc = self._block("Tactile\nEncoder", ORANGE, LEFT * 2.5, y_top - 2.0, w=2.0)
        lang_enc = self._block("LLM\nTokenizer", PURPLE, LEFT * 2.5, y_top - 4.0, w=2.0)

        a1 = Arrow(cam.get_right(), vis_enc.get_left(), color=GRAY, buff=0.1, stroke_width=2)
        a2 = Arrow(tac.get_right(), tac_enc.get_left(), color=GRAY, buff=0.1, stroke_width=2)
        a3 = Arrow(lang.get_right(), lang_enc.get_left(), color=GRAY, buff=0.1, stroke_width=2)

        self.play(
            FadeIn(vis_enc), FadeIn(tac_enc), FadeIn(lang_enc),
            GrowArrow(a1), GrowArrow(a2), GrowArrow(a3),
        )

        # Token shapes
        vis_dim = Text("[B, 256, 768]", font_size=13, color=BLUE).next_to(vis_enc, DOWN, buff=0.08)
        tac_dim = Text("[B, 64, 768]", font_size=13, color=ORANGE).next_to(tac_enc, DOWN, buff=0.08)
        lang_dim = Text("[B, 32, 768]", font_size=13, color=PURPLE).next_to(lang_enc, DOWN, buff=0.08)
        self.play(FadeIn(vis_dim), FadeIn(tac_dim), FadeIn(lang_dim))

        # Fusion transformer
        fuse = RoundedRectangle(
            width=3.0, height=4.5, corner_radius=0.2,
            color=TEAL, fill_opacity=0.15, stroke_width=3,
        ).shift(RIGHT * 1.5 + DOWN * 0.2)
        fuse_title = Text("Transformer\nBackbone", font_size=20, color=TEAL, weight=BOLD).move_to(fuse.get_top() + DOWN * 0.5)
        fuse_detail = Text("[B, 352, 768]\nSelf-Attention\nacross all tokens", font_size=15, color=WHITE_T).move_to(fuse)

        a4 = Arrow(vis_enc.get_right(), fuse.get_left() + UP * 1.2, color=BLUE, buff=0.1, stroke_width=2)
        a5 = Arrow(tac_enc.get_right(), fuse.get_left(), color=ORANGE, buff=0.1, stroke_width=2)
        a6 = Arrow(lang_enc.get_right(), fuse.get_left() + DOWN * 1.2, color=PURPLE, buff=0.1, stroke_width=2)

        self.play(
            FadeIn(fuse), Write(fuse_title), FadeIn(fuse_detail),
            GrowArrow(a4), GrowArrow(a5), GrowArrow(a6),
        )

        # Action head
        action = self._block("Action\nHead", GREEN, RIGHT * 5.0, y_top - 2.0, w=2.2)
        a7 = Arrow(fuse.get_right(), action.get_left(), color=TEAL, buff=0.1, stroke_width=2)
        act_dim = Text("[B, 7]\nΔpose + grip", font_size=14, color=GREEN).next_to(action, DOWN, buff=0.1)

        self.play(FadeIn(action), GrowArrow(a7), FadeIn(act_dim))

        # Key insight
        insight = Text(
            "Tactile tokens attend to visual & language tokens — the model learns when touch matters",
            font_size=18, color=YELLOW,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(insight))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _block(self, text, color, x_offset, y, w=2.0):
        r = RoundedRectangle(width=w, height=1.0, corner_radius=0.1, color=color, fill_opacity=0.15, stroke_width=2)
        t = Text(text, font_size=16, color=color).move_to(r)
        grp = VGroup(r, t)
        grp.move_to(x_offset + UP * y)
        return grp


# ════════════════════════════════════════════════════════════════════
# SCENE 12 — Hand-Object Dynamics
# ════════════════════════════════════════════════════════════════════
class S12_HandObject(Scene):
    def construct(self):
        heading = Text("Dynamic Hand-Object Modeling via Touch", font_size=38, color=PINK).to_edge(UP, buff=0.4)
        cite = Text("Zhang, Li, Luo et al., 2021 — MIT / Shanghai Jiao Tong", font_size=18, color=GRAY)
        cite.next_to(heading, DOWN, buff=0.15)
        self.play(Write(heading), FadeIn(cite))

        # Paper figure
        try:
            fig_img = ImageMobject(FIGURES + "hand_object_fig.png")
            fig_img.height = 3.0
            fig_img.shift(UP * 0.0)
            fig_border = SurroundingRectangle(fig_img, color=PINK, buff=0.05, stroke_width=1)
            fig_label = Text("Tactile glove captures hand-object interactions during daily tasks", font_size=16, color=GRAY)
            fig_label.next_to(fig_img, DOWN, buff=0.15)
            self.play(FadeIn(fig_img), Create(fig_border), Write(fig_label))
        except Exception:
            pass

        self.wait(1.5)

        facts = VGroup(
            Text("• High-res tactile glove captures full hand-object contact", font_size=18, color=WHITE_T),
            Text("• Predictive model + contrastive learning from tactile sequences", font_size=18, color=WHITE_T),
            Text("• Reconstructs 3D trajectories of hand AND object from touch alone", font_size=18, color=WHITE_T),
            Text("• Applications: activity learning, imitation learning, robotics", font_size=18, color=TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        facts.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(facts), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 13 — Limitations
# ════════════════════════════════════════════════════════════════════
class S13_Limitations(Scene):
    def construct(self):
        heading = Text("Current Limitations", font_size=48, color=RED).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        limitations = [
            ("Sensor Fragility", "Optical sensors (GelSight) degrade with use;\ngel membranes tear, markers wear off", BLUE),
            ("No Standard Interface", "Every sensor has different output format;\nno USB-like standard for tactile data", PURPLE),
            ("Sparse Coverage", "Fingertip-only sensing misses palm,\nwrist, and arm contacts", ORANGE),
            ("Sim-to-Real Gap", "Simulated tactile signals don't perfectly\nmatch real sensor noise and drift", TEAL),
            ("Data Scarcity", "Far less tactile training data than\nvision (460K vs billions of images)", PINK),
            ("Latency vs Resolution", "High-res sensors are slow (~30Hz);\nfast sensors are low-res", GREEN),
        ]

        items = VGroup()
        for title, desc, color in limitations:
            title_t = Text(title, font_size=22, color=color, weight=BOLD)
            desc_t = Text(desc.split('\n')[0], font_size=17, color=GRAY)
            row = VGroup(title_t, desc_t).arrange(RIGHT, buff=0.5, aligned_edge=UP)
            items.add(row)

        items.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        items.next_to(heading, DOWN, buff=0.5).to_edge(LEFT, buff=1.0)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 14 — Frontier Research
# ════════════════════════════════════════════════════════════════════
class S14_Frontier(Scene):
    def construct(self):
        heading = Text("Frontier Research Directions", font_size=42, color=GREEN).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        topics = [
            ("Foundation Models for Touch", "Sparsh, T3 show SSL pre-training\nworks across sensor types", PURPLE, LEFT * 3.5),
            ("Tactile Simulation at Scale", "TacSL + Isaac Sim enable\nmillions of simulated touches", PINK, RIGHT * 3.5),
            ("Whole-Body Tactile Skins", "Cover entire robot surface\nnot just fingertips (RoboSkin)", ORANGE, LEFT * 3.5),
            ("Tactile-VLA Co-training", "Train VLAs with touch from\nthe start, not as an add-on", TEAL, RIGHT * 3.5),
            ("Cross-Modal Supervision", "Use vision labels to train\ntactile-only models (zero-shot)", BLUE, LEFT * 3.5),
            ("Dexterous In-Hand Manipulation", "Multi-finger + tactile = human-level\nobject reorientation", YELLOW, RIGHT * 3.5),
        ]

        cards = VGroup()
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
            grp = VGroup(card, content).move_to(x_pos + DOWN * (row * 1.8 + 1.0))
            cards.add(grp)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in cards], lag_ratio=0.15),
            run_time=3,
        )
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ════════════════════════════════════════════════════════════════════
# SCENE 15 — Summary
# ════════════════════════════════════════════════════════════════════
class S15_Summary(Scene):
    def construct(self):
        heading = Text("Tactile Sensing: The Big Picture", font_size=48, color=TEAL).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        points = [
            "Touch gives robots what vision can't: contact forces, slip, geometry",
            "3D-ViTac: fuse vision + touch in 3D point clouds → diffusion policy",
            "Sparsh: self-supervised foundation model for tactile representations",
            "TANDEM: active exploration — robot decides where to touch",
            "TacSL: GPU-accelerated tactile simulation for scalable training",
            "Fusion trend: all modalities → tokens → one big transformer",
            "Still early — data, sensors, and standards are the bottleneck",
        ]

        items = VGroup()
        for p in points:
            items.add(Text(f"→  {p}", font_size=21, color=WHITE_T))
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        items.next_to(heading, DOWN, buff=0.6).to_edge(LEFT, buff=0.8)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.5)
        self.wait(1)

        thanks = Text("Thanks for watching!", font_size=36, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(thanks, shift=UP * 0.3))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
