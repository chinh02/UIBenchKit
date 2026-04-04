"""
LayoutCoder Structure Analysis
==============================

Extracts hierarchical layout structure from mask/screenshot images
and converts structures back to HTML code.

Core pipeline:
  Image -> recursive_cut_draw() -> structure JSON -> json_to_html_css() -> HTML

Ported from LayoutCoder's:
  - utils/code_gen/layout_extract.py (recursive cutting + structure extraction)
  - utils/code_gen/struct2code2mask_utils.py (structure -> HTML conversion)
"""

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup

from .utils import nested_dict, set_value, get_value, numbers_to_portions


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Point:
    x: int
    y: int

    def move(self, point):
        return Point(self.x + point.x, self.y + point.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Line:
    """A separation line between layout regions."""
    start: Point
    end: Point

    def val(self):
        return self.start.x, self.start.y, self.end.x, self.end.y

    def direction(self):
        return "y" if self.start.x == self.end.x else "x"

    def color(self):
        return (0, 255, 0) if self.direction() == "x" else (255, 0, 0)

    def __hash__(self):
        return hash((self.start, self.end))


@dataclass
class AbsImage:
    """An image region with its absolute position in the full page."""
    img: Image.Image
    abs_p: Point
    path: List


# ============================================================
# Separation Line Detection
# ============================================================

def soft_separation_lines(img, bbox=None, var_thresh=60, diff_thresh=5,
                          diff_portion=0.3, sliding_window=30):
    """
    Detect horizontal separation lines in an image using a sliding window approach.
    Finds rows that are visually uniform (low variance) with sharp boundaries.
    """
    img_array = np.array(img.convert("L"))
    img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    offset = 0 if bbox is None else bbox[1]
    lines = []

    for i in range(1 + sliding_window, len(img_array) - 1):
        upper = img_array[i - sliding_window - 1]
        window = img_array[i - sliding_window: i]
        lower = img_array[i]
        is_blank = np.var(window) < var_thresh
        is_border_top = np.mean(np.abs(upper - window[0]) > diff_thresh) > diff_portion
        is_border_bottom = np.mean(np.abs(lower - window[-1]) > diff_thresh) > diff_portion
        if is_blank and (is_border_top or is_border_bottom):
            line = i if is_border_bottom else i - sliding_window
            lines.append(line + offset)

    return sorted(lines)


# ============================================================
# Image Cutting
# ============================================================

def transform2line(line, line_direct, img_size):
    """Convert a scalar position to a Line object."""
    if line_direct == "x":
        return Line(Point(0, line), Point(img_size[0], line))
    else:
        return Line(Point(line, 0), Point(line, img_size[1]))


def transform2absolute(line, absolute_point: Point):
    """Transform line coordinates from relative to absolute."""
    return Line(line.start.move(absolute_point), line.end.move(absolute_point))


def cut_img(image, var_thresh=60, diff_thresh=5, diff_portion=0.3,
            line_direct="x", verbose=False):
    """
    Cut an image along detected separation lines in the given direction.

    Args:
        image: AbsImage to cut
        line_direct: "x" for horizontal cuts, "y" for vertical cuts
        diff_portion: threshold for edge detection (0.9 = strict, for masks)
    """
    assert line_direct in ["x", "y"], "line_direct must be 'x' or 'y'"

    abs_img, abs_p = image.img, image.abs_p
    # Rotate 90 degrees for vertical cutting
    img = abs_img if line_direct == "x" else abs_img.rotate(-90, expand=True)
    img_array = np.array(img.convert("L"))
    lines = soft_separation_lines(img, None, var_thresh, diff_thresh, diff_portion,
                                  sliding_window=5)
    if not lines:
        return [], []

    lines = sorted(list(set([0] + lines + [img_array.shape[0]])))

    cut_imgs = []
    for i in range(1, len(lines)):
        cut = img.crop((0, lines[i - 1], img_array.shape[1], lines[i]))
        if cut.size[1] <= 10 or cut.size[0] <= 10:
            continue
        elif np.array(cut.convert("L")).mean() >= 200:
            continue
        # Rotate back if needed
        cut = cut if line_direct == "x" else cut.rotate(90, expand=True)
        x = 0 if line_direct == "x" else lines[i - 1]
        y = 0 if line_direct == "y" else lines[i - 1]
        cut_imgs.append(AbsImage(cut, Point(x, y) + abs_p, deepcopy(image.path)))

    for i, cut in enumerate(cut_imgs):
        cut_imgs[i].path += ["value", i]

    if len(cut_imgs) == 1:
        return [], []

    # Transform lines to absolute coordinates
    adjusted_lines = [transform2absolute(transform2line(line, line_direct, abs_img.size), abs_p)
                      for line in lines]
    return cut_imgs, adjusted_lines


# ============================================================
# Recursive Structure Extraction
# ============================================================

def flatten(items, only_list=False):
    """Yield items from any nested iterable."""
    for x in items:
        instance = List if only_list else Iterable
        if isinstance(x, instance) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def draw_sep_lines(img, lines, verbose=True):
    """Draw separation lines on an image."""
    draw = ImageDraw.Draw(img)
    for line in lines:
        draw.line(line.val(), fill=line.color(), width=1)
    if verbose:
        img.show()
    return img


def draw_bbox(img, data: Iterable, verbose=True):
    """Draw bounding boxes with IDs on an image."""
    draw = ImageDraw.Draw(img)
    for ele in data:
        position = ele["position"]
        x1, y1 = position["column_min"], position["row_min"]
        x2, y2 = position["column_max"], position["row_max"]
        draw.rectangle(((x1, y1), (x2, y2)), outline="black", fill=None, width=3)
        _id, depth = ele["id"], ele["depth"]
        content = f"id: {_id}, depth: {depth}"
        try:
            font = ImageFont.load_default(img.size[0] / 70)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((x1 + 5, y1 + 5), content, "black", font)
    if verbose:
        img.show()
    return img


def tag_and_get_atomic_components(structure_data):
    """Tag each atomic component with a sequential ID and return them as a flat list."""

    def process_structure(structure, atomic_id_list, atomic_id=1, depth=0):
        if structure['type'] == 'atomic':
            atomic_info = {
                'id': atomic_id,
                'position': structure['position'],
                'depth': depth,
            }
            atomic_id_list.append(atomic_info)
            atomic_id += 1
        else:
            if 'value' in structure:
                for item in structure['value']:
                    atomic_id = process_structure(item, atomic_id_list, atomic_id, depth + 1)
        return atomic_id

    atomic_components = []
    process_structure(structure_data, atomic_components)
    return atomic_components


def recursive_cut_draw(image_path, depth=3):
    """
    Recursively split an image into layout blocks and extract hierarchical structure.

    Args:
        image_path: Path to the input image (mask or screenshot)
        depth: Maximum recursion depth (2-3 for screenshots, 5 for masks)

    Returns:
        (result_img, {"structure": dict, "page_size": (w, h)})
    """
    origin_img = Image.open(image_path)
    abs_img = AbsImage(origin_img, Point(0, 0), [])

    total_lines = []
    inverse_count = 0
    line_direct = "x"  # Start with horizontal cuts

    structure = nested_dict()
    img_list = [abs_img]

    while inverse_count < depth:
        next_list = []
        for i, img in enumerate(img_list):
            cut_imgs, lines = cut_img(img, verbose=False, line_direct=line_direct,
                                      diff_portion=0.9)
            if not cut_imgs:
                # Try the other direction
                line_direct = "x" if line_direct == "y" else "y"
                cut_imgs, lines = cut_img(img, verbose=False, line_direct=line_direct,
                                          diff_portion=0.9)
                if not cut_imgs:
                    continue

            # Build structure
            if len(cut_imgs) > 0:
                current_type = "column" if line_direct == "x" else "row"
                h_w = 1 if line_direct == "x" else 0

                portions = []
                for cut in cut_imgs:
                    portion_part = cut.img.size[h_w]
                    portions.append({
                        "abs_pos": cut.abs_p,
                        "portion": portion_part,
                        "size": cut.img.size
                    })
                portions = numbers_to_portions(portions)

                child_structure = {
                    "type": current_type,
                    "value": [
                        {
                            "type": "atomic",
                            "portion": p["portion"],
                            "value": "     ",
                            "position": {
                                "column_min": p["abs_pos"].x,
                                "row_min": p["abs_pos"].y,
                                "column_max": p["abs_pos"].x + p["size"][0],
                                "row_max": p["abs_pos"].y + p["size"][1],
                            }
                        }
                        for p in portions
                    ]
                }
                # If only one element, promote it
                if len(child_structure["value"]) == 1:
                    child_structure = child_structure["value"]

                if len(cut_imgs[0].path) <= 2:
                    structure = child_structure
                else:
                    portion = get_value(structure, cut_imgs[0].path[:-2] + ["portion"])
                    set_value(structure, cut_imgs[0].path[:-2], child_structure)
                    set_value(structure, cut_imgs[0].path[:-2] + ["portion"], portion)

            next_list += cut_imgs
            total_lines += lines

        # Alternate direction each round
        line_direct = "x" if line_direct == "y" else "y"
        img_list = deepcopy(next_list)
        inverse_count += 1

    # Generate visualization
    atomic_components = tag_and_get_atomic_components(structure)
    result_img = draw_sep_lines(origin_img, list(set(flatten(total_lines))), verbose=False)
    result_img = draw_bbox(result_img, atomic_components, verbose=False)

    return result_img, {
        "structure": structure,
        "page_size": origin_img.size
    }


def mask2json(input_root, output_root, name):
    """
    Extract layout structure from a mask image.

    Args:
        input_root: Directory containing {name}_mask.png
        output_root: Directory to save {name}_sep.png
        name: Image base name

    Returns:
        {"structure": dict, "page_size": (w, h)}
    """
    input_path = os.path.join(input_root, f"{name}_mask.png")
    os.makedirs(output_root, exist_ok=True)
    sep_path = os.path.join(output_root, f"{name}_sep.png")

    result_img, data = recursive_cut_draw(input_path, depth=5)
    result_img.save(sep_path)
    return data


# ============================================================
# Structure -> HTML Conversion
# ============================================================

def json_to_html_css(structure):
    """Convert a hierarchical layout structure to HTML with flex CSS."""

    def process_node(node, parent_type='row', root=False):
        html = ''
        if node['type'] == 'row':
            html += f'<div class="row{" root" if root else ""}" style="flex: {node.get("portion", 1)};">\n'
            for child in node['value']:
                html += process_node(child, parent_type='row')
            html += '</div>\n'
        elif node['type'] == 'column':
            html += f'<div class="column{" root" if root else ""}" style="flex: {node.get("portion", 1)};">\n'
            for child in node['value']:
                html += process_node(child, parent_type='column')
            html += '</div>\n'
        elif node['type'] == 'atomic':
            html += f'<div class="atomic" style="flex: {node["portion"]};">{node["code"]}</div>\n'
        return html

    return process_node(structure, root=True)


def add_html_template(data, border=True, margin="2px", bg_color="white", ratio=None):
    """Wrap layout HTML in a full HTML template with CSS for flex layout."""
    border = "border: 1px solid white;" if border else "border: none;"
    margin = f"margin: {margin};"
    bg_color = f"background-color: {bg_color};" if bg_color else ""
    root_ratio = f"width: 100%; aspect-ratio: {ratio};" if ratio else "width: 100vw; height: 100vh;"

    return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap" rel="stylesheet">
            <script src="https://cdn.tailwindcss.com"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
            <title>VAN UI2Code</title>
            <style>
            body {{
              font-family: 'Noto Sans SC', sans-serif;
            }}
            </style>
            <title>Random Structure</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}
                .root {{
                    margin: 0;
                    padding: 0;
                    {root_ratio}
                }}
                .row {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: row;
                    width: 100%;
                }}
                .column {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                }}
                .atomic {{
                    padding: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    {border}
                    {margin}
                    {bg_color}
                }}
            </style>
        </head>
        <body>
            {data}
        </body>
        </html>
        """


def prettify_html(data):
    """Prettify HTML using BeautifulSoup."""
    return BeautifulSoup(data, 'html.parser').prettify()
