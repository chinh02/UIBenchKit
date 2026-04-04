"""
UICopilot Bbox Tree Utilities
==============================

Functions for parsing, manipulating, and rendering bbox-annotated HTML trees.
Ported from uicopilot/scripts/train/utils.py.

Heavy dependencies (torch, transformers) are imported lazily inside functions
that need them, so that the module can be imported without GPU/ML libraries.
"""

import re
import io
import base64

# Precision for bbox coordinate rounding
BBOX_PRECISION = 3


def move_to_device(data, device):
    """Recursively move tensors to the specified device."""
    import torch
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def BboxTree2Html(node, style=False, size=(1, 1)):
    """Convert a bbox tree dict to an HTML string."""
    if isinstance(node, str):
        return node
    elif not node:
        return ''
    dom_type = node['type']
    child_doms = [BboxTree2Html(cnode, style, size) for cnode in node['children']]
    if style:
        if node['type'] == 'input':
            tree = f"<{dom_type} style='{node.get('style', '')}' value='{''.join(child_doms)}'></{dom_type}>"
        elif node['type'] == 'img':
            tree = f"<{dom_type} style='{node.get('style', '')}' src='{child_doms[0] if child_doms else ''}'></{dom_type}>"
        else:
            tree = f"<{dom_type} style='{node.get('style', '')}'>{''.join(child_doms)}</{dom_type}>"
    else:
        bbox = node['bbox']
        tree = (
            f"<{dom_type} bbox=["
            f"{round(bbox[0] / size[0], BBOX_PRECISION)},"
            f"{round(bbox[1] / size[1], BBOX_PRECISION)},"
            f"{round(bbox[2] / size[0], BBOX_PRECISION)},"
            f"{round(bbox[3] / size[1], BBOX_PRECISION)}"
            f"]>{''.join(child_doms)}</{dom_type}>"
        )
    return tree


def BboxTree2StyleList(node, index='', skip_leaf=True):
    """Extract a flat list of nodes with bbox and style info."""
    if skip_leaf and not len(node['children']):
        return []
    bs_list = [{
        'type': node['type'],
        'bbox': node['bbox'],
        'index': index,
        'style': node['style'].strip() if ('style' in node and node['style']) else '',
        'children': [{
            'type': x['type'],
            'bbox': x['bbox'],
            'style': x['style'].strip() if ('style' in x and x['style']) else ''
        } for x in node['children'] if isinstance(x, dict)]
    }]
    for idx, cnode in enumerate(node['children']):
        if isinstance(cnode, dict):
            bs_list += BboxTree2StyleList(
                cnode, f"{index}{'-' if index else ''}{idx}", skip_leaf
            )
    return bs_list


def Html2BboxTree(html, size=(1, 1)):
    """Parse bbox-annotated HTML string into a tree dict."""
    root_node = None
    index = None

    while len(html):
        html = html.replace('<s>', '').strip()

        match_bot = re.search(r'^<([a-zA-Z0-9]+)\s*([^>]*)\s*>', html)
        match_eot = re.search(r'^</([a-zA-Z0-9]+)\s*>', html)

        if match_bot:
            dom_type, bbox_str = match_bot.groups()
            bbox = [float(x) for x in bbox_str.split('[')[1].split(']')[0].split(',')]
            bbox[0] = int(bbox[0] * size[0])
            bbox[1] = int(bbox[1] * size[1])
            bbox[2] = int(bbox[2] * size[0])
            bbox[3] = int(bbox[3] * size[1])
            html = html[match_bot.end():]
            node = {
                'type': dom_type,
                'bbox': bbox,
                'children': []
            }
            if not root_node:
                root_node = node
                index = []
            else:
                target = root_node
                for i in index:
                    target = target['children'][i]
                target['children'].append(node)
                index.append(len(target['children']) - 1)

        elif match_eot:
            dom_type, = match_eot.groups()
            html = html[match_eot.end():]
            target = root_node
            for i in index:
                target = target['children'][i]
            if target['type'] == dom_type and len(index):
                index.pop()
        else:
            break

    return root_node


def smart_tokenizer_and_embedding_resize(model, tokenizer, special_tokens_dict):
    """Resize tokenizer and embeddings for special tokens."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def add_special_tokens(model, tokenizer):
    """Add the special tokens required by the Pix2Struct bbox model."""
    from transformers import AddedToken
    smart_tokenizer_and_embedding_resize(model, tokenizer, {
        'bos_token': AddedToken('<s>', rstrip=False, lstrip=False, single_word=False, normalized=True),
        'additional_special_tokens': [
            AddedToken('<dom>', rstrip=False, lstrip=False, single_word=False, normalized=True),
            AddedToken('</dom>', rstrip=False, lstrip=False, single_word=False, normalized=True),
            AddedToken('<css>', rstrip=False, lstrip=False, single_word=False, normalized=True),
            AddedToken('</css>', rstrip=False, lstrip=False, single_word=False, normalized=True),
        ]
    })


def pil_to_base64(image):
    """Convert a PIL Image to a base64-encoded string."""
    from PIL import Image
    buffered = io.BytesIO()
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
