from typing import Union
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from skimage.metrics import structural_similarity as ssim
import os
from PIL import Image, ImageDraw, ImageEnhance 
from tqdm.auto import tqdm
import time
import re
import base64
import io
from openai import OpenAI, AzureOpenAI
import numpy as np
import google.generativeai as genai
import json
import anthropic

# Import model classes from the new models package for backward compatibility
from models import Bot, FakeBot, Gemini, GPT4, Claude, QwenVL, BedrockBot

def take_screenshot(driver, filename):
    # save_full_page_screenshot is Firefox-only, use get_screenshot_as_file for Chrome
    try:
        driver.save_full_page_screenshot(filename)
    except AttributeError:
        # Chrome doesn't have save_full_page_screenshot, use regular screenshot
        driver.save_screenshot(filename)

def get_driver(file=None, headless=True, string=None, window_size=(1920, 1080)):
    assert file or string, "You must provide a file or a string"
    
    # Try Chrome first (better Docker support), fall back to Firefox
    try:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService
        
        options = ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        
        # Check for custom chromedriver path (Docker)
        chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")
        chrome_bin = os.environ.get("CHROME_BIN")
        
        if chrome_bin:
            options.binary_location = chrome_bin
        
        if chromedriver_path:
            service = ChromeService(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)
            
    except Exception as e:
        # Fall back to Firefox
        options = Options()
        if headless:
            options.add_argument("-headless")
        driver = webdriver.Firefox(options=options)
        driver.set_window_size(window_size[0], window_size[1])

    if not string:
        driver.get("file:///" + os.getcwd() + "/" + file)
    else:
        string = base64.b64encode(string.encode('utf-8')).decode()
        driver.get("data:text/html;base64," + string)

    return driver


from playwright.sync_api import sync_playwright
import os
import base64

def take_screenshot_pw(page, filename=None):
    # Takes a full-page screenshot with Playwright
    if filename:
        page.screenshot(path=filename, full_page=True)
    else:
        return page.screenshot(full_page=True)  # Returns the screenshot as bytes if no filename is provided

def get_driver_pw(file=None, headless=True, string=None, window_size=(1920, 1080)):
    assert file or string, "You must provide a file or a string"
   
    p = sync_playwright().start()  # Start Playwright context manually
    browser = p.chromium.launch(headless=headless)
    page = browser.new_page()

    # If the user provides a file, load it, else load the HTML string
    if file:
        page.goto("file://" + os.getcwd() + "/" + file)
    else:
        string = base64.b64encode(string.encode('utf-8')).decode()
        page.goto("data:text/html;base64," + string)
    
    # Set the window size
    page.set_viewport_size({"width": window_size[0], "height": window_size[1]})
    
    return page, browser  # Return the page and browser objects


# Try to load placeholder from multiple possible locations
PLACEHOLDER_URL = None
placeholder_paths = [
    './placeholder.png',
    './scripts/placeholder.png',
    './Tool/static/placeholder.png',
    os.path.join(os.path.dirname(__file__), 'placeholder.png'),
    os.path.join(os.path.dirname(__file__), 'scripts', 'placeholder.png'),
    os.path.join(os.path.dirname(__file__), 'Tool', 'static', 'placeholder.png')
]

for placeholder_path in placeholder_paths:
    if os.path.exists(placeholder_path):
        with open(placeholder_path, 'rb') as image_file:
            # Read the image as a binary stream
            img_data = image_file.read()
            # Convert the image to base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            # Create a base64 URL (assuming it's a PNG image)
            PLACEHOLDER_URL = f"data:image/png;base64,{img_base64}"
        break

if PLACEHOLDER_URL is None:
    # Fallback: create a simple 1x1 transparent PNG
    import base64
    transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    PLACEHOLDER_URL = f"data:image/png;base64,{base64.b64encode(transparent_png).decode('utf-8')}"

def get_placeholder(html):
    html_with_base64 = html.replace("placeholder.png", PLACEHOLDER_URL)
    return html_with_base64


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import base64
from tqdm.auto import tqdm
import os
from PIL import Image, ImageDraw, ImageChops

def num_of_nodes(driver, area="body", element=None):
    # number of nodes in body
    element = driver.find_element(By.TAG_NAME, area) if not element else element
    script = """
    function get_number_of_nodes(base) {
        var count = 0;
        var queue = [];
        queue.push(base);
        while (queue.length > 0) {
            var node = queue.shift();
            count += 1;
            var children = node.children;
            for (var i = 0; i < children.length; i++) {
                queue.push(children[i]);
            }
        }
        return count;
    }
    return get_number_of_nodes(arguments[0]);
    """
    return driver.execute_script(script, element)

measure_time = {
    "script": 0,
    "screenshot": 0,
    "comparison": 0,
    "open image": 0,
    "hash": 0,
}


import hashlib
import mmap

def compute_hash(image_path):
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        # Use memory-mapped file for efficient reading
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            hash_md5.update(mm)
    return hash_md5.hexdigest()

def are_different_fast(img1_path, img2_path):
    # a extremely fast algorithm to determine if two images are different,
    # only compare the size and the hash of the image
    return compute_hash(img1_path) != compute_hash(img2_path)

str2base64 = lambda s: base64.b64encode(s.encode('utf-8')).decode()

import time

def simplify_graphic(driver, element, progress_bar=None, img_name={"origin": "origin.png", "after": "after.png"}):
    """utility for simplify_html, simplify the html by removing elements that are not visible in the screenshot"""
    children = element.find_elements(By.XPATH, "./*")
    deletable = True
    # check childern
    if len(children) > 0:
        for child in children:
            deletable *= simplify_graphic(driver, child, progress_bar=progress_bar, img_name=img_name)
    # check itself
    
    if deletable:
        original_html = driver.execute_script("return arguments[0].outerHTML;", element)

        tick = time.time()
        driver.execute_script("""
            var element = arguments[0];
            var attrs = element.attributes;
            while(attrs.length > 0) {
                element.removeAttribute(attrs[0].name);
            }
            element.innerHTML = '';""", element)
        measure_time["script"] += time.time() - tick
        tick = time.time()
        driver.save_full_page_screenshot(img_name["after"])
        measure_time["screenshot"] += time.time() - tick
        tick = time.time()
        deletable = not are_different_fast(img_name["origin"], img_name["after"])
        measure_time["comparison"] += time.time() - tick

        if not deletable:
            # be careful with children vs child_node and assining outer html to element without parent
            driver.execute_script("arguments[0].outerHTML = arguments[1];", element, original_html)
        else:
            driver.execute_script("arguments[0].innerHTML = 'MockElement!';", element)
            # set visible to false
            driver.execute_script("arguments[0].style.display = 'none';", element)
    if progress_bar:
        progress_bar.update(1)

    return deletable
            
def simplify_html(fname, save_name, pbar=True, area="html", headless=True):
    """simplify the html file and save the result to save_name, return the compression rate of the html file after simplification"""
    # copy the fname as save_name
    
    driver = get_driver(file=fname, headless=headless)
    print("driver initialized")
    original_nodes = num_of_nodes(driver, area)
    bar = tqdm(total=original_nodes) if pbar else None
    compression_rate = 1
    driver.save_full_page_screenshot(f"{fname}_origin.png")
    try:
        simplify_graphic(driver, driver.find_element(By.TAG_NAME, area), progress_bar=bar, img_name={"origin": f"{fname}_origin.png", "after": f"{fname}_after.png"})
        elements = driver.find_elements(By.XPATH, "//*[text()='MockElement!']")

        # Iterate over the elements and remove them from the DOM
        for element in elements:
            driver.execute_script("""
                var elem = arguments[0];
                elem.parentNode.removeChild(elem);
            """, element)
        
        compression_rate = num_of_nodes(driver, area) / original_nodes
        with open(save_name, "w", encoding="utf-8") as f:
            f.write(driver.execute_script("return document.documentElement.outerHTML;"))
    except Exception as e:
        print(e, fname)
    # remove images
    driver.quit()

    os.remove(f"{fname}_origin.png")
    os.remove(f"{fname}_after.png")
    return compression_rate

# Function to encode the image in base64
def encode_image(image):
    """
    Encode an image to base64 string.
    
    Args:
        image: Either a file path (str) or a PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    if type(image) == str:
        try: 
            with open(image, "rb") as image_file:
                encoding = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(e)
            with open(image, "r", encoding="utf-8") as image_file:
                encoding = base64.b64encode(image_file.read()).decode('utf-8')
        return encoding
    else:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


# Note: Bot classes (Bot, Gemini, GPT4, QwenVL, Claude, BedrockBot, FakeBot) 
# have been moved to the 'models' package. They are imported at the top of this file
# for backward compatibility.


from abc import ABC, abstractmethod
import random

class ImgNode(ABC):
    # self.img: the image of the node
    # self.bbox: the bounding box of the node
    # self.children: the children of the node

    @abstractmethod
    def get_img(self):
        pass


class ImgSegmentation(ImgNode):
    def __init__(self, img: Union[str, Image.Image], bbox=None, children=None, max_depth=None, var_thresh=50, diff_thresh=45, diff_portion=0.9, window_size=50) -> None:
        if type(img) == str:
            img = Image.open(img)
        self.img = img
        # (left, top, right, bottom)
        self.bbox = (0, 0, img.size[0], img.size[1]) if not bbox else bbox
        self.children = children if children else []
        self.var_thresh = var_thresh
        self.diff_thresh = diff_thresh
        self.diff_portion = diff_portion
        self.window_size = window_size
        
        if max_depth:
            self.init_tree(max_depth)
        self.depth = self.get_depth()

    def init_tree(self, max_depth):
        def _init_tree(node, max_depth, cur_depth=0):
            if cur_depth == max_depth:
                return
            cuts = node.cut_img_bbox(node.img, node.bbox, line_direct="x")
            
            if len(cuts) == 0:
                cuts = node.cut_img_bbox(node.img, node.bbox, line_direct="y")

            # print(cuts)
            for cut in cuts:
                node.children.append(ImgSegmentation(node.img, cut, [], None, self.var_thresh, self.diff_thresh, self.diff_portion, self.window_size))

            for child in node.children:
                _init_tree(child, max_depth, cur_depth + 1)

        _init_tree(self, max_depth)

    def get_img(self, cut_out=False, outline=(0, 255, 0)):
        if cut_out:
            return self.img.crop(self.bbox)
        else:
            img_draw = self.img.copy()
            draw = ImageDraw.Draw(img_draw)
            draw.rectangle(self.bbox, outline=outline, width=5)
            return img_draw
    
    def display_tree(self, save_path=None):
        # draw a tree structure on the image, for each tree level, draw a different color
        def _display_tree(node, draw, color=(255, 0, 0), width=5):
            # deep copy the image
            draw.rectangle(node.bbox, outline=color, width=width)
            for child in node.children:
                # _display_tree(child, draw, color=tuple([int(random.random() * 255) for i in range(3)]), width=max(1, width))
                _display_tree(child, draw, color=color, width=max(1, width))

        img_draw = self.img.copy()
        draw = ImageDraw.Draw(img_draw)
        for child in self.children:
            _display_tree(child, draw)
        if save_path:
            img_draw.save(save_path)
        else:
            img_draw.show()

    def get_depth(self):
        def _get_depth(node):
            if node.children == []:
                return 1
            return 1 + max([_get_depth(child) for child in node.children])
        return _get_depth(self)
    
    def is_leaf(self):
        return self.children == []
    
    def to_json(self, path=None):
        '''
        [
            { "bbox": [left, top, right, bottom],
                "level": the level of the node,},
            { "bbox": [left, top, right, bottom],
            "level": the level of the node,}
            ...
        ]
        '''
        # use bfs to traverse the tree
        res = []
        queue = [(self, 0)]
        while queue:
            node, level = queue.pop(0)
            res.append({"bbox": node.bbox, "level": level})
            for child in node.children:
                queue.append((child, level + 1))
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=4)
        return res
    
    def to_json_tree(self, path=None):
        '''
        {
            "bbox": [left, top, right, bottom],
            "children": [
                {
                    "bbox": [left, top, right, bottom],
                    "children": [ ... ]
                },
                ...
            ]
        }
        '''
        def _to_json_tree(node):
            res = {"bbox": node.bbox, "children": []}
            for child in node.children:
                res["children"].append(_to_json_tree(child))
            return res
        res = _to_json_tree(self)
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=4)
        return res

    def cut_img_bbox(self, img, bbox,  line_direct="x", verbose=False, save_cut=False):
        """cut the the area of interest specified by bbox (left, top, right, bottom), return a list of bboxes of the cut image."""
        
        diff_thresh = self.diff_thresh
        diff_portion = self.diff_portion
        var_thresh = self.var_thresh
        sliding_window = self.window_size

        # def soft_separation_lines(img, bbox=None, var_thresh=None, diff_thresh=None, diff_portion=None, sliding_window=None):
        #     """return separation lines (relative to whole image) in the area of interest specified by bbox (left, top, right, bottom). 
        #     Good at identifying blanks and boarders, but not explicit lines. 
        #     Assume the image is already rotated if necessary, all lines are in x direction.
        #     Boundary lines are included."""
        #     img_array = np.array(img.convert("L"))
        #     img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
        #     offset = 0 if bbox is None else bbox[1]
        #     lines = []
        #     for i in range(1 + sliding_window, len(img_array) - 1):
        #         upper = img_array[i-sliding_window-1]
        #         window = img_array[i-sliding_window:i]
        #         lower = img_array[i]
        #         is_blank = np.var(window) < var_thresh
        #         # content width is larger than 33% of the width
        #         is_boarder_top = np.mean(np.abs(upper - window[0]) > diff_thresh) > diff_portion
        #         is_boarder_bottom = np.mean(np.abs(lower - window[-1]) > diff_thresh) > diff_portion
        #         if is_blank and (is_boarder_top or is_boarder_bottom):
        #             line = i if is_boarder_bottom else i - sliding_window
        #             lines.append(line + offset)
        #     return sorted(lines)
        def soft_separation_lines(img, bbox=None, var_thresh=None, diff_thresh=None, diff_portion=None, sliding_window=None):
            """return separation lines (relative to whole image) in the area of interest specified by bbox (left, top, right, bottom). 
            Good at identifying blanks and boarders, but not explicit lines. 
            Assume the image is already rotated if necessary, all lines are in x direction.
            Boundary lines are included."""
            img_array = np.array(img.convert("L"))
            img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            # import matplotlib.pyplot as plt
            # # show the image array
            # plt.imshow(img_array, cmap="gray")
            # plt.show()

            offset = 0 if bbox is None else bbox[1]
            lines = []
            for i in range(2*sliding_window, len(img_array) - sliding_window):
                upper = img_array[i-2*sliding_window:i-sliding_window]
                window = img_array[i-sliding_window:i]
                lower = img_array[i:i+sliding_window]
                is_blank = np.var(window) < var_thresh
                # content width is larger than 33% of the width
                is_boarder_top = np.var(upper) > var_thresh
                is_boarder_bottom = np.var(lower) > var_thresh
                # print(i, "is_blank", is_blank, "is_boarder_top", is_boarder_top, "is_boarder_bottom", is_boarder_bottom)
                if is_blank and (is_boarder_top or is_boarder_bottom):
                    line = (i + i - sliding_window) // 2
                    lines.append(line + offset)

            # print(sorted(lines))
            return sorted(lines)

        def hard_separation_lines(img, bbox=None, var_thresh=None, diff_thresh=None, diff_portion=None):
            """return separation lines (relative to whole image) in the area of interest specified by bbox (left, top, right, bottom). 
            Good at identifying explicit lines (backgorund color change). 
            Assume the image is already rotated if necessary, all lines are in x direction
            Boundary lines are included."""
            img_array = np.array(img.convert("L"))
            # img.convert("L").show()
            img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            offset = 0 if bbox is None else bbox[1]
            prev_row = None
            prev_row_idx = None
            lines = []

            # loop through the image array
            for i in range(len(img_array)):
                row = img_array[i]
                # if the row is too uniform, it's probably a line
                if np.var(img_array[i]) < var_thresh:
                    # print("row", i, "var", np.var(img_array[i]))
                    if prev_row is not None:
                        # the portion of two rows differ more that diff_thresh is larger than diff_portion
                        # print("prev_row", prev_row_idx, "diff", np.mean(np.abs(row - prev_row) > diff_thresh))
                        if np.mean(np.abs(row - prev_row) > diff_thresh) > diff_portion:
                            lines.append(i + offset)
                            # print("line", i)
                    prev_row = row
                    prev_row_idx = i
            # print(sorted(lines))
            return lines

        def new_bbox_after_rotate90(img, bbox, counterclockwise=True):
            """return the new coordinate of the bbox after rotating 90 degree, based on the original image."""
            if counterclockwise:
                # the top right corner of the original image becomes the origin of the coordinate after rotating 90 degree
                top_right = (img.size[0], 0)
                # change the origin
                bbox = (bbox[0] - top_right[0], bbox[1] - top_right[1], bbox[2] - top_right[0], bbox[3] - top_right[1])
                # rotate the bbox 90 degree counterclockwise (x direction change sign)
                bbox = (bbox[1], -bbox[2], bbox[3], -bbox[0])
            else:
                # the bottom left corner of the original image becomes the origin of the coordinate after rotating 90 degree
                bottom_left = (0, img.size[1])
                # change the origin
                bbox = (bbox[0] - bottom_left[0], bbox[1] - bottom_left[1], bbox[2] - bottom_left[0], bbox[3] - bottom_left[1])
                # rotate the bbox 90 degree clockwise (y direction change sign)
                bbox = (-bbox[3], bbox[0], -bbox[1], bbox[2])
            return bbox
        
        assert line_direct in ["x", "y"], "line_direct must be 'x' or 'y'"
        img = ImageEnhance.Sharpness(img).enhance(6)
        bbox = bbox if line_direct == "x" else new_bbox_after_rotate90(img, bbox, counterclockwise=True) # based on the original image
        img = img if line_direct == "x" else img.rotate(90, expand=True)
        lines = []
        # img.show()
        lines = soft_separation_lines(img, bbox, var_thresh, diff_thresh, diff_portion, sliding_window)
        lines += hard_separation_lines(img, bbox, var_thresh, diff_thresh, diff_portion)
        # print(hash(str(np.array(img).data)), bbox, var_thresh, diff_thresh, diff_portion, sliding_window, lines)
        if lines == []:
            return []
        lines = sorted(list(set([bbox[1],] + lines + [bbox[3],]))) # account for the beginning and the end of the image
        # list of images cut by the lines
        cut_imgs = []
        for i in range(1, len(lines)):
            cut = img.crop((bbox[0], lines[i-1], bbox[2], lines[i]))
            # if empty or too small, skip
            if cut.size[1] < sliding_window:
                continue
            elif np.array(cut.convert("L")).var() < var_thresh:
                continue
            cut = (bbox[0], lines[i-1], bbox[2], lines[i])  # (left, top, right, bottom)
            cut = cut if line_direct == "x" else new_bbox_after_rotate90(img, cut, counterclockwise=False)
            cut_imgs.append(cut)

        # if all other images are blank, this remaining image is the same as the original image
        if len(cut_imgs) == 1:
            return []
        if verbose:
            img = img if line_direct == "x" else img.rotate(-90, expand=True)
            draw = ImageDraw.Draw(img)
            for cut in cut_imgs:
                draw.rectangle(cut, outline=(0, 255, 0), width=5)
                draw.line(cut, fill=(0, 255, 0), width=5)
            img.show()
        if save_cut:
            img.save("cut.png")
        
        return cut_imgs
    
from threading import Thread
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import bs4


class DCGenTrace():
    def __init__(self, img_seg, bot, prompt):
        self.img = img_seg.img
        self.bbox = img_seg.bbox
        self.children = []
        self.bot = bot
        self.prompt = prompt
        self.code = None

    def get_img(self, cut_out=False, outline=(255, 0, 0)):
        if cut_out:
            return self.img.crop(self.bbox)
        else:
            img_draw = self.img.copy()
            draw = ImageDraw.Draw(img_draw)
            # shift one pixel to the right and down to make the outline visible
            draw.rectangle(self.bbox, outline=outline, width=5)
            return img_draw

    def display_tree(self, node_size=(5, 5)):
        def _plot_node(ax, node, position, parent_position=None, color='r'):
            # Display the node's image
            img = np.array(node.get_img())
            ax.imshow(img, extent=(position[0] - node_size[0]/2, position[0] + node_size[0]/2,
                                   position[1] - node_size[1]/2, position[1] + node_size[1]/2))

            # Draw a rectangle around the node's image
            ax.add_patch(patches.Rectangle((position[0] - node_size[0]/2, position[1] - node_size[1]/2),
                                           node_size[0], node_size[1], fill=False, edgecolor=color, linewidth=2))

            # Connect parent to child with a line
            if parent_position:
                ax.plot([parent_position[0], position[0]], [parent_position[1], position[1]], color=color, linewidth=2)
            
            # Recursive plotting for children
            num_children = len(node.children)
            if num_children > 0:
                for i, child in enumerate(node.children):
                    # Calculate child position
                    child_x = position[0] + (i - (num_children - 1) / 2) * node_size[0] * 2
                    child_y = position[1] - node_size[1] * 3
                    _plot_node(ax, child, (child_x, child_y), position, color=tuple([int(random.random() * 255) / 255.0 for _ in range(3)]))

        # Setup the plot
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.axis('off')

        # Start plotting from the root node
        _plot_node(ax, self, (0, 0))
        plt.savefig("tree.png")

    def generate_code(self, recursive=False, cut_out=False, multi_thread=True):
        if self.is_leaf() or not recursive:
            self.code = self.bot.try_ask(self.prompt, encode_image(self.get_img(cut_out=cut_out)))
            pure_code = re.findall(r"```html([^`]+)```", self.code)
            if pure_code:
                self.code = pure_code[0]
        else:
            code_parts = []  
            if multi_thread:
                threads = []
                for child in self.children:
                    t = Thread(target=child.generate_code, kwargs={"recursive": True, "cut_out": cut_out})
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
            else:
                for child in self.children:
                    child.generate_code(recursive=True, cut_out=cut_out, multi_thread=False)

            for child in self.children:
                code_parts.append(child.code)
                if child.code is None:
                    print("Warning: Child code is None")

            code_parts = '\n=============\n'.join(code_parts)
            self.code = self.bot.try_ask(self.prompt + code_parts, encode_image(self.get_img(cut_out=cut_out)))
            pure_code = re.findall(r"```html([^`]+)```", self.code)
            if pure_code:
                self.code = pure_code[0]
        return self.code
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def get_num_of_nodes(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + sum([child.get_num_of_nodes() for child in self.children])
        
    def to_json(self, path=None):
        '''
        [
            { 
            "bbox": [left, top, right, bottom],
            "code": the code of the node,
            "level": the level of the node,
            },
            { 
            "bbox": [left, top, right, bottom],
            "code": the code of the node,
            "level": the level of the node
            },
            ...
        ]
        '''
        def _to_json(node, level):
            res = []
            res.append({"bbox": node.bbox, "code": node.code, "level": level, "prompt": node.prompt})
            for child in node.children:
                res += _to_json(child, level + 1)
            return res
        res = _to_json(self, 0)

        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=4)
        return res



    @classmethod
    def from_img_seg(cls, img_seg, bot, prompt_leaf, prompt_node, prompt_root=None):
        if not prompt_root:
            prompt_root = prompt_node
        def _from_img_seg(img_seg, entry_point=False):
            if img_seg.is_leaf() and not entry_point:
                return DCGenTrace(img_seg, bot, prompt_leaf)
            elif not entry_point:
                trace = DCGenTrace(img_seg, bot, prompt_node)
                for child in img_seg.children:
                    trace.children.append(_from_img_seg(child))
                return trace
            else:
                trace = DCGenTrace(img_seg, bot, prompt_root)
                for child in img_seg.children:
                    trace.children.append(_from_img_seg(child))
                return trace
            
        return _from_img_seg(img_seg, entry_point=True)
    

from concurrent.futures import ThreadPoolExecutor
class DCGenGrid:
    def __init__(self, img_seg, prompt_seg, prompt_refine):
        self.img_seg_tree = self.assign_seg_tree_id(img_seg.to_json_tree())
        self.img = img_seg.img
        self.prompt_seg = prompt_seg
        self.prompt_refine = prompt_refine
        self.html_template = self.get_html_template()
        self.code = None
        self.raw_code = None

    def generate_code(self, bot, multi_thread=True):
        """generate the complete html code for the image"""
        # print("Generating code for the image...")
        code_dict = self.generate_code_dict(bot, multi_thread)
        # print("Substituting code in the HTML template...")
        self.raw_code = self.code_substitution(self.html_template, code_dict)
        # print("Refining the code...")
        code = bot.try_ask(self.prompt_refine.replace("[CODE]", self.raw_code), encode_image(self.img), num_generations=1)
        pure_code = re.findall(r"```html([^`]+)```", code)
        if pure_code:
            code = pure_code[0]
        # print("Optimizing the code...")
        self.code = bot.optimize([code, self.raw_code], self.img, showimg=False)
        return self.code

    def _generate_code_dict(self, bot):
        """generate code for all the leaf nodes in the bounding box tree, return a dictionary: {'id': 'code'}"""
        code_dict = {}
        def _generate_code(node):
            if node["children"] == []:
                bbox = node["bbox"]
                cropped_img = self.img.crop(bbox)
                code = bot.try_ask(self.prompt_seg, encode_image(cropped_img), num_generations=2).replace("```html", "").replace("```", "")
                code_dict[node["id"]] = code
            else:
                for child in node["children"]:
                    _generate_code(child)

        _generate_code(self.img_seg_tree)
        return code_dict
    
    
    def _generate_code_dict_parallel(self, bot):
        """Generate code for all the leaf nodes in the bounding box tree, return a dictionary: {'id': 'code'}"""
        code_dict = {}

        def _generate_code(node):
            if node["children"] == []:
                bbox = node["bbox"]
                cropped_img = self.img.crop(bbox)
                # print(f"Generating code for node {node['id']} with bbox {bbox}")
                generated_code = bot.try_ask(self.prompt_seg, encode_image(cropped_img), num_generations=2).replace("```html", "").replace("```", "")
                code_dict[node["id"]] = generated_code
            else:
                for child in node["children"]:
                    _generate_code(child)

        # Using ThreadPoolExecutor to handle parallelism
        with ThreadPoolExecutor() as executor:
            # Traverse the tree and submit tasks to the thread pool
            futures = []
            def submit_task(node):
                if node["children"] == []:
                    futures.append(executor.submit(_generate_code, node))
                else:
                    for child in node["children"]:
                        submit_task(child)
            
            submit_task(self.img_seg_tree)
            
            # Wait for all futures to complete
            for future in futures:
                future.result()  # This will block until the thread finishes

        return code_dict
        
    def generate_code_dict(self, bot, parallel=True):
        """generate code for all the leaf nodes in the bounding box tree, return a dictionary: {'id': 'code'}"""
        if parallel:
            return self._generate_code_dict_parallel(bot)
        else:
            return self._generate_code_dict(bot)
    
    def get_html_template(self, output_file=None, verbose=False):
        """
        Generates an HTML file with nested containers based on the bounding box tree.

        :param bbox_tree: Dictionary representing the bounding box tree.
        :param output_file: The name of the output HTML file.
        """
        bbox_tree = self.img_seg_tree
        # HTML and CSS templates
        # the container class is used to create grid and position the boxes
        # include the tailwind css in the head tag
        html_template_start = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Bounding Boxes Layout</title>
            <style>
                body, html {
                    margin: 0;
                    padding: 0;
                    width: 100vw;
                    height: 100vh;
                }
                .container { 
                    position: relative;
                    width: 100%;
                    height: 100%;
                    max-width: 100% !important;
                    max-height: 100% !important;
                    box-sizing: border-box;
                    min-width: [ROOT_WIDTH]px;
                    min-height: [ROOT_HEIGHT]px;

                }
                .box {
                    position: absolute;
                    box-sizing: border-box;
                    overflow: hidden;
                }
                .box > .container {
                    display: grid;
                    width: 100%;
                    height: 100%;
                }

            </style>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container">
        """

        html_template_end = """
            </div>
        </body>
        </html>
        """

        # Function to recursively generate HTML
        def process_bbox(node, parent_width, parent_height, parent_left, parent_top):
            """
            Recursively processes the bounding box tree and returns HTML string.

            :param node: Current bounding box node.
            :param parent_width: Width of the parent container.
            :param parent_height: Height of the parent container.
            :param parent_left: Left position of the parent container.
            :param parent_top: Top position of the parent container.
            :return: HTML string for the current node and its children.
            """
            bbox = node['bbox']
            children = node.get('children', [])
            id = node['id']

            # Calculate relative positions and sizes
            left = (bbox[0] - parent_left) / parent_width * 100
            top = (bbox[1] - parent_top) / parent_height * 100
            width = (bbox[2] - bbox[0]) / parent_width * 100
            height = (bbox[3] - bbox[1]) / parent_height * 100
            color = ''
            if verbose:
                color = f"background-color: #{random.randint(0, 0xFFFFFF):06x}; "
            # Start the box div
            html = f'''
                <div id="{id}" class="box" style="left: {left}%; top: {top}%; width: {width}%; height: {height}%; {color}">
            '''

            if children:
                # If there are children, add a nested container
                html += '''
                    <div class="container">
                '''
                # Get the current box's width and height in pixels for child calculations
                current_width = bbox[2] - bbox[0]
                current_height = bbox[3] - bbox[1]
                for child in children:
                    html += process_bbox(child, current_width, current_height, bbox[0], bbox[1])
                html += '''
                    </div>
                '''
            
            # Close the box div
            html += '''
                </div>
            '''
            return html

        # Start processing from the root
        root_bbox = bbox_tree['bbox']
        root_children = bbox_tree.get('children', [])
        root_width = root_bbox[2] - root_bbox[0]
        root_height = root_bbox[3] - root_bbox[1]
        root_x = root_bbox[0]
        root_y = root_bbox[1]

        # Initialize HTML content
        html_content = html_template_start.replace("[ROOT_WIDTH]", str(root_width)).replace("[ROOT_HEIGHT]", str(root_height))

        # Process each top-level child
        for child in root_children:
            html_content += process_bbox(child, root_width, root_height, root_x, root_y)

        # Close HTML tags
        html_content += html_template_end

        # prettify the HTML content
        soup = bs4.BeautifulSoup(html_content, 'html.parser')
        html_content = soup.prettify()

        if verbose:
            output_file = "verbose.html"

        # Write to the output file
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        return html_content

    @staticmethod
    def assign_seg_tree_id(img_seg_tree):
        """assign each node a unique id"""
        def assign_id(node, id):
            node["id"] = id
            for child in node.get("children", []):
                id = assign_id(child, id+1)
            return id
        assign_id(img_seg_tree, 0)
        return img_seg_tree
    
    @staticmethod
    def code_substitution(html, code_dict, output_file=None):
        """substitute the containers in the html template with the corresponding generated code in code_dict"""
        soup = bs4.BeautifulSoup(html, 'html.parser')
        for id, code in code_dict.items():
            code = code.replace("```html", "").replace("```", "")
            div = soup.find(id=id)
            # replace the inner html of the div
            if div:
                div.append(bs4.BeautifulSoup(code, 'html.parser'))
        result = soup.prettify()
        if output_file:
            with open(output_file, "w", encoding="utf8", errors="ignore") as f:
                f.write(result)
        return result