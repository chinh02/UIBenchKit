"""
LatCoder Prompts
================

LLM prompts for code generation, assembly, and refinement.
"""

PROMPT_GENERATE = """You are an expert Tailwind developer.

Based on the reference screenshot of a specific section of a webpage (such as the header, footer, card, etc.) provided by the user, build a single-page app using Tailwind, HTML, and JS. Please follow the detailed requirements below to ensure the generated code is accurate:

### Basic Requirements:
                         
1. **Rigid Requirements**
   - You are provided with the following unmodifiable HTML framework:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body>
    <!-- Your task is to fill this area -->
</body>
</html>
```
   - Your task is to generate a code block that starts with a <div> tag and ends with a </div> tag, and embed it within the <body> tag of the above - mentioned framework.
   - Do not deliberately center the content. Arrange the elements according to their original layout and positions.
   - The generated code should not have fixed width and height settings.
   - Ensure that the proportions of images in the code are preserved.
   - Both the margin and padding in the code should be set to 0.
   - Make sure that the generated code does not conflict with outer <div> elements in terms of layout and style.
   - The final return should be the complete HTML code, that is, including the above - mentioned framework and the code you generated and embedded into the <body> of the framework.

2. **Appearance and Layout Consistency:**
   - Ensure the app looks exactly like the screenshot, including the position, hierarchy, and content of all elements.
   - The generated HTML elements and Tailwind classes should match those in the screenshot, ensuring that text, colors, fonts, padding, margins, borders, and other styles are perfectly aligned.

3. **Content Consistency:**
   - Use the exact text from the screenshot, ensuring the content of every element matches the image.
   - For images, use placeholder images from https://placehold.co and include a detailed description in the alt text for AI-generated images.

4. **No Comments or Placeholders:**
   - Do not add comments like "<!-- Add other navigation links as needed -->" or "<!-- ... other news items ... -->". Write the full, complete code for each element.

5. **Libraries to Use:**
   - Use the following libraries:
     - Google Fonts: Use the relevant fonts from the screenshot.
                         

### Process Steps:

1. **Analyze the Section:**
   Based on the provided screenshot, analyze a specific section of the webpage (such as the header, footer, card, form, etc.). Break down all the elements in this section (e.g., text, images, buttons, etc.) and understand their relative positions and hierarchy.

2. **Generate HTML Code:**
   Based on the analysis from Step 1, generate a complete HTML code snippet representing that specific section, ensuring all elements, positions, and styles match the screenshot.

3. **Text Content Comparison:**
   Compare the generated HTML with the screenshot's text content to ensure accuracy. If there are any discrepancies or missing content, make corrections.

4. **Color Comparison:**
   Compare the text color and background color in the generated HTML with those in the screenshot. If they don't match, adjust the Tailwind classes and styles to reflect the correct colors.

5. **Background and Other Style Comparison:**
   Ensure the background colors, borders, padding, margins, and other styles in the generated HTML accurately reflect the design shown in the screenshot.

6. **Final Integration:**
   After reviewing and refining the previous steps, ensure that the generated HTML code is complete and perfectly matches the specific section of the screenshot.

### Code Format:

Please return the complete HTML code"""


PROMPT_ASSEMBLE = """You are an experienced front-end developer tasked with assembling multiple webpage module codes into a complete webpage.

# CONTEXT #
I will provide a screenshot of a webpage, the location information for each module, and the corresponding module code. 
Your task is to assemble these modules into a complete webpage code based on their positions.

# OBJECTIVE #
Generate a complete HTML file that ensures the layout, style, and content of each module match the original webpage.

# RESPONSE #
You need to return the final assembled complete HTML code, for example:
```html
code
```

# steps #
Please follow the steps below:

**step1**: Analyze the webpage screenshot and the position information of each module.
- Based on the screenshot and the module position data, understand the relative placement and layout of each module.
- The position of each module is defined by two coordinates: the top-left corner [x1, y1] and the bottom-right corner [x2, y2]. x1 < x2 and y1 < y2. The coordinate values range from 0 to 1, representing the ratio relative to the width and height of the image.

**step2**: Assemble the HTML code of each module based on its position.
- Use the provided module code to stitch the modules together in the correct order and position.
- Ensure that the modules do not overlap and that the layout is correct.

**step3**: Review and fix the assembled webpage.
- Compare the generated webpage with the screenshot to ensure the content, layout, and styles match exactly.
- If any issues such as misalignment, overlapping, or missing content are found, fix them.

**step4**: Generate the final HTML code.
- Based on the checks and fixes from step 3, generate the final complete webpage code that closely matches the screenshot.

# Notes #
- There should be no overlap between modules.
- For image modules, use placeholder images from https://placehold.co and provide a detailed description in the alt text for AI-based image generation.
- Pay attention to details such as background color, text color, font size, padding, margins, borders, and other visual elements.
- **Do not omit any module's code**. Every module, regardless of its size or complexity, must be included in the final HTML code. Ensure that each module's functionality and layout are represented fully in the assembled page.
                 
# Libraries #
- You may use Google Fonts to match the fonts in the screenshot.
- Use the Font Awesome icon library: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>
"""


PROMPT_REFINE = """You are an expert front-end developer tasked with refining a generated webpage to better match the original design.

# CONTEXT #
I will provide:
1. The original design image (Image 1)
2. A screenshot of the current generated webpage (Image 2)
3. The current HTML code
4. Text content extracted from the original design

# OBJECTIVE #
Analyze the differences between the design and the generated webpage, then refine the HTML code to achieve a closer match.

# RESPONSE #
Return the refined complete HTML code:
```html
refined code
```

# REFINEMENT FOCUS #
1. **Layout Accuracy**: Fix any misaligned or incorrectly positioned elements
2. **Content Match**: Ensure all text content matches the original
3. **Style Consistency**: Match colors, fonts, spacing, and visual effects
4. **Missing Elements**: Add any elements present in design but missing in output

# Notes #
- Preserve all functional elements and interactions
- Use the same library stack (Tailwind, FontAwesome)
- Include all content, no omissions or placeholders"""


PROMPT_GET_TEXT = """You are a text extraction specialist.

# TASK #
Extract all visible text content from the provided webpage design image.

# OUTPUT FORMAT #
Return the extracted text as a JSON object with sections:
```json
{
    "navigation": ["text items in navigation"],
    "headings": ["heading texts"],
    "body_text": ["paragraph and body text"],
    "buttons": ["button labels"],
    "other": ["any other text content"]
}
```

Be thorough and capture ALL visible text exactly as it appears."""
