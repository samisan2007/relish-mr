"""Interactive SAM3 / SAM 3.1 image tester.

    python ui.py     ->  opens http://127.0.0.1:7860

Models load on selection. SAM 3.1 runs in a disposable GPU Docker worker.
"""

import gradio as gr
import gc
import torch

from geometry import mask_geometry
from sam3_runner import Sam3Runner
from sam31_runner import MODEL_SAM31_COMPILED, SAM31_CHOICES, Sam31Runner
from viz import draw_instances

runner = None


def run(image, prompt, threshold, show_masks, show_boxes, model_choice="SAM3"):
    global runner
    if image is None or not prompt.strip():
        return None, "Provide an image and a prompt."

    try:
        if model_choice in SAM31_CHOICES:
            runner = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            active = Sam31Runner(compile_model=model_choice == MODEL_SAM31_COMPILED)
        else:
            if runner is None:
                runner = Sam3Runner()
            active = runner
        result = active.segment(image, prompt.strip(), threshold=threshold)
    except Exception as error:
        return None, f"{model_choice} failed: {error}"
    if not result.instances:
        return image, f'No "{prompt}" found ({result.inference_ms:.0f}ms). Try lowering the threshold.'

    lines = [f'{len(result.instances)} found in {result.inference_ms:.0f}ms', ""]
    for i, inst in enumerate(result.instances):
        geo = mask_geometry(inst.mask)
        lines.append(f"#{i}  score {inst.score:.2f}  "
                     f"centre ({geo.centroid_xy[0]:.0f}, {geo.centroid_xy[1]:.0f})  "
                     f"diameter {geo.diameter_px:.0f}px")
    annotated = draw_instances(image, result.instances, show_masks=show_masks, show_boxes=show_boxes)
    return annotated, "\n".join(lines)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Prompt", placeholder="meatball"),
        gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="Confidence threshold"),
        gr.Checkbox(value=True, label="Segmentation masks"),
        gr.Checkbox(value=True, label="Bounding boxes"),
        gr.Dropdown(choices=["SAM3", *SAM31_CHOICES], value="SAM3", label="Model"),
    ],
    outputs=[gr.Image(label="Segmentation"), gr.Textbox(label="Results", lines=10)],
    title="Relish image tester",
    description="Type a noun phrase. First use of a model loads it; SAM 3.1 compiled takes minutes.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
