import math
import torch

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from oauthlib.uri_validate import segment
from sympy.polys.subresultants_qq_zz import rotate_r

from .utility.utility import tensor2pil, pil2tensor, conditioning_set_values
class FrameSegments:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frames = -1
        self.segment_rows = []
        self.row_count = 0

    def __getitem__(self, key):
        return self.segment_rows[key]

    def __str__(self):
        return f"FrameSegments: {self.row_count} rows"

    def add_row(self):
        row = FrameSegmentRow(self.row_count, self)
        self.row_count += 1
        self.segment_rows.append(row)
        return row

class FrameSegmentRow:
    def __init__(self, row, segments):
        self.cols = []
        self.row = row
        self.col_count = 0
        self.segments = segments

    def __getitem__(self, key):
        return self.cols[key]

    def __str__(self):
        return f"Row {self.row}, Columns: {self.col_count}"

    def add_col(self, segment):
        segment.row = self.row
        segment.col = self.col_count
        segment.segments = self.segments
        self.col_count += 1
        self.cols.append(segment)
        return segment

class FrameSegment:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center_x = self.x + self.width/2
        self.center_y = self.y + self.height/2
        self.row = -1
        self.col = -1
        self.segments = None

    def __str__(self):
        return f"Segment: {self.row}, {self.col} ({self.x}, {self.y}) ({self.width}, {self.height})"

    def is_first_row(self):
        return self.row == 0

    def is_last_row(self):
        return self.row == self.segments.row_count - 1

    def is_first_col(self):
        return self.col == 0

    def is_last_col(self):
        return self.col == self.segments[self.row].col_count - 1



class FrameSegmenter:

    RETURN_TYPES = ("FRAME_SEGMENTS", "MASK")
    RETURN_NAMES = ("frame_segments", "seam_mask")
    FUNCTION = "frame_segmenter"
    CATEGORY = "AharaNodes/frame_segmenter"
    DESCRIPTION = """
"""

    @classmethod
    def INPUT_TYPES(s):
        inputs =  {
            "required": {
                "row_count": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
                "col_count_1": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
                "col_count_2": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
                "col_count_3": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
                "seam_width": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            }
        }

        return inputs

    def frame_segmenter(self, row_count, col_count_1, col_count_2, col_count_3, seam_width, frames, frame_width, frame_height):
        col_count = [col_count_1, col_count_2, col_count_3]
        if not self.validate(row_count, col_count, frame_width, frame_height, seam_width):
            return (False, False, )

        segments = self.get_segments(row_count, col_count, frame_width, frame_height)
        segments.frames = frames
        seam_mask = self.get_seam_mask(seam_width, segments)

        return (segments, seam_mask, )

    def validate(self, row_count, col_count, frame_width, frame_height, seam_width):
        if frame_height % row_count != 0:
            print(f"frame_height {frame_height} not a multiple of row_count {row_count} ")
            return False

        for col in col_count:
            if frame_width % col !=0:
                print(f"frame_width {frame_width} not a multiple of col_count {col} ")
                return False

        if seam_width % 2 != 0:
            print(f"seam_width {seam_width} not a multiple of 2")
            return False

        return True

    def get_segments(self, row_count, col_count, frame_width, frame_height):
        height_diff = frame_height // row_count
        segments = FrameSegments(frame_width, frame_height)
        y = 0
        for row_index in range(row_count):
            x = 0
            row = segments.add_row()
            col = col_count[row_index]
            width_diff = frame_width // col
            for c in range(col):
                row.add_col(FrameSegment(x,y, width_diff, height_diff))
                x += width_diff
            y += height_diff

        return segments

    def get_seam_mask(self, seam_width, segments):
        # Define the number of images in the batch
        color = "white"

        image = Image.new("RGB", (segments.frame_width, segments.frame_height), "black")
        draw = ImageDraw.Draw(image)

        for row in segments:
            for segment in row:
                # draw top
                if not segment.is_first_row():
                    left_up = (segment.x , segment.y - seam_width // 2)
                    right_down = (segment.x + segment.width , segment.y + seam_width // 2)
                    two_points = [left_up, right_down]
                    draw.rectangle(two_points, fill=color)
                # draw right
                if not segment.is_last_col():
                    left_up = (segment.x + segment.width - seam_width // 2, segment.y)
                    right_down = (segment.x + segment.width + seam_width // 2, segment.y + segment.height)
                    two_points = [left_up, right_down]
                    draw.rectangle(two_points, fill=color)
                # draw bot
                if not segment.is_last_row():
                    left_up = (segment.x , segment.y + segment.height - seam_width // 2)
                    right_down = (segment.x + segment.width , segment.y + segment.height + seam_width // 2)
                    two_points = [left_up, right_down]
                    draw.rectangle(two_points, fill=color)
                # draw left
                if not segment.is_first_col():
                    left_up = (segment.x - seam_width // 2, segment.y)
                    right_down = (segment.x + seam_width // 2, segment.y + segment.height)
                    two_points = [left_up, right_down]
                    draw.rectangle(two_points, fill=color)

        image = pil2tensor(image)
        mask = image[:, :, :, 0]

        out = [mask for x in range(segments.frames)]
        outstack = torch.cat(out, dim=0)
        return outstack

class FrameSegmenterIndexer:

    RETURN_TYPES = ("MASK", "MASK", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("mask", "mask_inverted", "POS", "NEG")
    FUNCTION = "indexer"
    CATEGORY = "AharaNodes/frame_segmenter"
    DESCRIPTION = """
Isolates conditioning to a specific segment of a frame and additional returns masks to use in composotion of that segment in a final frame.  
Segment type supports 4 modes, 3x3, 3x2, 3x1 and 3x1 + 1x1. Defaults to width x height (wide orientation), use swap_dimensions to height x width (tall orientation).
segment_row and segment_column provide indexes in to the final grid.
If conditioning is supplied, use that, otherwise generate from clip + text prompt. Both, either, or neither of [pos|neg] can be provided.
"""

    @classmethod
    def INPUT_TYPES(s):
        inputs =  {
            "required": {
                "segments": ("FRAME_SEGMENTS", ),
                "segment_row": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "segment_col": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "pos_text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "neg_text": ("STRING", {"multiline": True, "dynamicPrompts": True})
            }, "optional": {
                "clip": ("CLIP", ),
                "pos": ("CONDITIONING", ), 
                "neg": ("CONDITIONING", ), 
            }
        }

        return inputs

    def indexer(self, segments, segment_row, segment_col, clip=None, pos_text="", neg_text="", pos=None, neg=None):
        if not self.validate(clip, pos, neg):
            return (False, False, False, False,)

        current_segment = segments[segment_row][segment_col]
        print(current_segment)

        mask, mask_inverted = self.create_shape_mask(segments, current_segment)
        if pos is None:
            pos = self.create_conditioning(clip, pos_text)
        if neg is None:
            neg = self.create_conditioning(clip, neg_text)
        c_pos, c_neg = self.constrain_conditioning(pos, neg, current_segment)

        return (mask, mask_inverted, c_pos, c_neg, )

    def validate(self, clip, pos, neg):
        if (pos is None or neg is None) and clip is None:
            print("No clip provided and need to condition pos or neg.")
            return False

        return True

    def create_shape_mask(self, segments, current_segment):
        # Define the number of images in the batch
        color = "white"

        image = Image.new("RGB", (segments.frame_width, segments.frame_height), "black")
        draw = ImageDraw.Draw(image)
        left_up = (current_segment.x, current_segment.y)
        right_down = (current_segment.x + current_segment.width, current_segment.y + current_segment.height)
        two_points = [left_up, right_down]
        draw.rectangle(two_points, fill=color)
        image = pil2tensor(image)
        mask = image[:, :, :, 0]

        out = [mask for x in range(segments.frames)]
        outstack = torch.cat(out, dim=0)
        return (outstack, 1.0 - outstack)

    def create_conditioning(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]


    def constrain_conditioning(self, pos, neg, current_segment):
        c_pos = conditioning_set_values(pos, {"area": (current_segment.height // 8, current_segment.width // 8, current_segment.y // 8, current_segment.x // 8),
                                                 "strength": 1,
                                                 "set_area_to_bounds": False})
        c_neg = conditioning_set_values(neg, {"area": (current_segment.height // 8, current_segment.width // 8, current_segment.y // 8, current_segment.x // 8),
                                                 "strength": 1,
                                                 "set_area_to_bounds": False})
        return (c_pos, c_neg)