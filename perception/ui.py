"""Interactive SAM 3 tester: upload an image, type a prompt, see the mask.

    python ui.py     ->  opens http://127.0.0.1:7860

Model loads once at startup, so each prompt after that takes ~0.5s.
"""

import gradio as gr

from geometry import mask_geometry
from sam3_runner import Sam3Runner
from viz import draw_instances

print("Loading SAM 3...")
runner = Sam3Runner()
print(f"Ready on {runner.device} ({runner.load_seconds:.1f}s)")


def run(image, prompt, threshold, show_masks, show_boxes):
    if image is None or not prompt.strip():
        return None, "Provide an image and a prompt."

    result = runner.segment(image, prompt.strip(), threshold=threshold)
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
    ],
    outputs=[gr.Image(label="Segmentation"), gr.Textbox(label="Results", lines=10)],
    title="SAM 3 tester",
    description="Type any noun phrase. Masks, boxes, centroids and pixel diameters.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
