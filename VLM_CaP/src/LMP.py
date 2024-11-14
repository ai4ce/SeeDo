import copy
import openai
import shapely
import ast
import astunparse
from time import sleep
import numpy as np
from shapely.geometry import *
from shapely.affinity import *

# from openai.error import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from src.env import COLORS

# openai keys
from src.key import mykey, projectkey


class LMP:

    def __init__(self, client, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg
        self._client = client

        self._base_prompt = self._cfg["prompt_text"]

        self._stop_tokens = list(self._cfg["stop"])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""

    def clear_exec_hist(self):
        self.exec_hist = ""

    def build_prompt(self, query, context=""):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (
                f"from utils import {', '.join(self._variable_vars.keys())}"
            )
        else:
            variable_vars_imports_str = ""
        prompt = self._base_prompt.replace(
            "{variable_vars_imports}", variable_vars_imports_str
        )

        if self._cfg["maintain_session"]:
            prompt += f"\n{self.exec_hist}"

        if context != "":
            prompt += f"\n{context}"

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f"\n{use_query}"

        return prompt, use_query

    def __call__(self, query, context="", **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)
        
        message = [
            {"role": "user", "content": prompt},
        ]

        while True:
            try:
                code_str = self._client.chat.completions.create(
                    messages = message,
                    stop=self._stop_tokens,
                    temperature=self._cfg["temperature"],
                    model=self._cfg["engine"],
                    max_tokens=self._cfg["max_tokens"],
                ).choices[0].message.content.strip()
                break
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                print(f"OpenAI API got err {e}")
                print("Retrying after 10s.")
                sleep(10)

        if self._cfg["include_context"] and context != "":
            to_exec = f"{context}\n{code_str}"
            to_log = f"{context}\n{use_query}\n{code_str}"
        else:
            to_exec = code_str
            to_log = f"{use_query}\n{to_exec}"

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f"LMP {self._name} exec:\n\n{to_log_pretty}\n")

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg["debug_mode"]:
            print("to_exec: ", to_exec)
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f"\n{to_exec}"

        if self._cfg["maintain_session"]:
            self._variable_vars.update(lvars)

        if self._cfg["has_return"]:
            return lvars[self._cfg["return_val_name"]]


class LMPFGen:

    def __init__(self, client, cfg, fixed_vars, variable_vars):
        self._cfg = cfg
        self._client = client

        self._stop_tokens = list(self._cfg["stop"])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg["prompt_text"]

    def create_f_from_sig(
        self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False
    ):
        print(f"Creating function: {f_sig}")

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        # prompt = f"{self._base_prompt}\n{use_query}"
        message = [
            {"role": "user", "content": f"{self._base_prompt}\n{use_query}"}
        ]

        while True:
            try:
                f_src = self._client.chat.completions.create(
                    messages = message,
                    stop=self._stop_tokens,
                    temperature=self._cfg["temperature"],
                    model=self._cfg["engine"],
                    max_tokens=self._cfg["max_tokens"],
                ).choices[0].message.content.strip()
                break
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                print(f"OpenAI API got err {e}")
                print("Retrying after 10s.")
                sleep(10)

        if fix_bugs:
            f_src = openai.CodeEdit.create(
                model="code-davinci-002",
                input="# " + f_src,
                temperature=0,
                instruction="Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.",
            )["choices"][0]["text"].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(
            f"{use_query}\n{f_src}", PythonLexer(), TerminalFormatter()
        )
        print(f"LMP FGEN created:\n\n{to_print}\n")

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(
        self, code_str, other_vars=None, fix_bugs=False, return_src=False
    ):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts(
                [self._fixed_vars, self._variable_vars, new_fs, other_vars]
            )
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(
                    f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True
                )

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts(
                        [self._fixed_vars, self._variable_vars, new_fs, other_vars]
                    )
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    print("savely executing code: ", code_str)
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    # print("custom_gvars: ", custom_gvars)
    # print("------")
    # print("local vars: ", lvars)
    # print("------")
    exec(code_str, custom_gvars, lvars)


class LMP_wrapper:

    def __init__(self, env, cfg, render=False):
        self.env = env
        self._cfg = cfg
        self.object_names = list(self._cfg["env"]["init_objs"])

        self._min_xy = np.array(self._cfg["env"]["coords"]["bottom_left"])
        self._max_xy = np.array(self._cfg["env"]["coords"]["top_right"])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg["env"]["coords"]["table_z"]
        self.render = render

    def is_obj_visible(self, obj_name):
        return obj_name in self.object_names

    def get_obj_names(self):
        return self.object_names[::]

    def denormalize_xy(self, pos_normalized):
        return pos_normalized * self._range_xy + self._min_xy

    def get_corner_positions(self):
        unit_square = box(0, 0, 1, 1)
        normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
        corners = np.array(
            ([self.denormalize_xy(corner) for corner in normalized_corners])
        )
        return corners

    def get_side_positions(self):
        side_xs = np.array([0, 0.5, 0.5, 1])
        side_ys = np.array([0.5, 0, 1, 0.5])
        normalized_side_positions = np.c_[side_xs, side_ys]
        side_positions = np.array(
            ([self.denormalize_xy(corner) for corner in normalized_side_positions])
        )
        return side_positions

    def get_obj_pos(self, obj_name):
        # return the xy position of the object in robot base frame
        return self.env.get_obj_pos(obj_name)[:2]

    def get_obj_position_np(self, obj_name):
        return self.get_pos(obj_name)

    def get_bbox(self, obj_name):
        # return the axis-aligned object bounding box in robot base frame (not in pixels)
        # the format is (min_x, min_y, max_x, max_y)
        bbox = self.env.get_bounding_box(obj_name)
        return bbox

    def get_color(self, obj_name):
        for color, rgb in COLORS.items():
            if color in obj_name:
                return rgb

    def pick_place(self, pick_pos, place_pos):
        pick_pos_xyz = np.r_[pick_pos, [self._table_z]]
        place_pos_xyz = np.r_[place_pos, [self._table_z]]
        pass

    def put_first_on_second(self, arg1, arg2):
        # put the object with obj_name on top of target
        # target can either be another object name, or it can be an x-y position in robot base frame
        pick_pos = self.get_obj_pos(arg1) if isinstance(arg1, str) else arg1
        place_pos = self.get_obj_pos(arg2) if isinstance(arg2, str) else arg2
        self.env.step(action={"pick": pick_pos, "place": place_pos})

    def get_robot_pos(self):
        # return robot end-effector xy position in robot base frame
        return self.env.get_ee_pos()

    def goto_pos(self, position_xy):
        # move the robot end-effector to the desired xy position while maintaining same z
        ee_xyz = self.env.get_ee_pos()
        position_xyz = np.concatenate([position_xy, ee_xyz[-1]])
        while np.linalg.norm(position_xyz - ee_xyz) > 0.01:
            self.env.movep(position_xyz)
            self.env.step_sim_and_render()
            ee_xyz = self.env.get_ee_pos()

    def follow_traj(self, traj):
        for pos in traj:
            self.goto_pos(pos)

    def get_corner_positions(self):
        normalized_corners = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
        return np.array(
            ([self.denormalize_xy(corner) for corner in normalized_corners])
        )

    def get_side_positions(self):
        normalized_sides = np.array([[0.5, 1], [1, 0.5], [0.5, 0], [0, 0.5]])
        return np.array(([self.denormalize_xy(side) for side in normalized_sides]))

    def get_corner_name(self, pos):
        corner_positions = self.get_corner_positions()
        corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        return [
            "top left corner",
            "top right corner",
            "bottom left corner",
            "botom right corner",
        ][corner_idx]

    def get_side_name(self, pos):
        side_positions = self.get_side_positions()
        side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
        return ["top side", "right side", "bottom side", "left side"][side_idx]
