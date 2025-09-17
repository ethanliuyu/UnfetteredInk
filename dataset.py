import math
import os
import pickle
import torch
import lmdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cairosvg
from PIL import Image
import io

content_reference_json = "../cr_mapping.json"
import random
import torchvision.transforms as transforms


def simulate_in_coon_handwritten(merged_data, epsilon=0.05, h_initial=50, connect_prob=0.5):
    """
    模中手写，**部分**笔画连接并拟合线性段。

    参数：
    - merged_data (dict): 包含 'command', 'para' 字段的字典。
    - epsilon (float): 误差阈值，用于控制拟合精度。
    - h_initial (float): 初始控制点偏移量。
    - connect_prob (float): 两笔画相邻时执行连接操作的概率，取值范围 0~1。

    返回:
    - dict: 处理后的包含 'command', 'para' 字段的字典。
    """
    commands = merged_data['command']
    paras = merged_data['para']

    new_commands = []
    new_paras = []

    last_point = None  # 上一个点的坐标

    for i, (cmd, para) in enumerate(zip(commands, paras)):
        if cmd == 'M':
            # 只有在满足概率条件时才执行笔画连接
            if last_point is not None and random.random() < connect_prob:
                # 连接上一个点 (p0) 和当前 'M' 点 (p2)
                p0 = last_point
                p2 = para

                # 计算 p1（控制点）
                x0, y0 = p0
                x2, y2 = p2
                x_m = (x0 + x2) / 2
                y_m = (y0 + y2) / 2

                dx = x2 - x0
                dy = y2 - y0
                d = math.sqrt(dx ** 2 + dy ** 2)

                if d == 0:
                    u_x, u_y = 0, 0
                else:
                    u_x = -dy / d
                    u_y = dx / d

                h = d / 2
                # 调整 h 以确保 p1 在 [1, 255] 范围内
                while True:
                    x1 = x_m + h * u_x
                    y1 = y_m + h * u_y
                    if 1 <= x1 <= 255 * 2 and 1 <= y1 <= 255 * 2:
                        break
                    h = h / 2
                    if h < 1e-3:
                        # 如果 h 变得过小，设置为中点
                        x1, y1 = x_m, y_m
                        break

                # 将控制点坐标取整数
                p1 = (int(round(x1)), int(round(y1)))

                # 定义二次贝塞尔曲线点计算函数
                def bezier_quadratic(t, p0, p1, p2):
                    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
                    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
                    return (x, y)

                # 计算两点之间的欧氏距离
                def distance(p1, p2):
                    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

                # 递归拟合函数
                def fit_linear_segments(p0, p1, p2, epsilon):
                    segments = []

                    def recurse(p0, p1, p2):
                        B_mid = bezier_quadratic(0.5, p0, p1, p2)
                        L_mid = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
                        e = distance(B_mid, L_mid)

                        if e < epsilon:
                            # 将坐标取整后添加
                            seg_start = (int(round(p0[0])), int(round(p0[1])))
                            seg_end = (int(round(p2[0])), int(round(p2[1])))
                            segments.append((seg_start, seg_end))
                        else:
                            # 分割曲线为两部分
                            p0_a = p0
                            p1_a = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                            p2_a = B_mid

                            p0_b = B_mid
                            p1_b = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                            p2_b = p2

                            recurse(p0_a, p1_a, p2_a)
                            recurse(p0_b, p1_b, p2_b)

                    recurse(p0, p1, p2)
                    return segments

                # 获取拟合后的线段
                segments = fit_linear_segments(p0, p1, p2, epsilon)

                # 将线段转换为 'L' 命令并添加到新列表中
                for seg_start, seg_end in segments:
                    new_commands.append('L')
                    new_paras.append([seg_end[0], seg_end[1]])
            else:
                # 不连接，保持原来的 'M' 命令
                new_commands.append(cmd)
                new_paras.append(para)

            # 更新上一个点为当前点
            last_point = para
        else:
            # 直接添加 'L' 命令
            new_commands.append(cmd)
            new_paras.append(para)
            last_point = para

    # 返回处理后的字典
    return {
        'command': new_commands,
        'para': new_paras,
    }


def felax_line(command, para):
    """
    处理命令和参数列表，应用类似贝塞尔曲线的线性变换。

    参数：
    - command (list of str): 包含 'M' 或 'L' 命令的列表。
    - para (list of list/tuple): 包含坐标对 [x, y] 的列表。

    返回:
    - list of list: 修改后的 para 列表，坐标已更新。
    """

    def bezier_curve(P0, P1, t):
        """
        计算线性贝塞尔曲线上的点，并将结果四舍五入为整数。

        参数：
        - P0 (list/tuple): 起点坐标 [x0, y0]。
        - P1 (list/tuple): 终点坐标 [x1, y1]。
        - t (float): 参数 t，控制点的位置。

        返回:
        - list: 新的坐标 [x, y]，取整后。
        """
        x = (1 - t) * P0[0] + t * P1[0]
        y = (1 - t) * P0[1] + t * P1[1]

        return [int(round(x)), int(round(y))]

    # 定义可选的 t 值
    t_values = [0.8, 0.9, 1.0, 1.1, 1.2]

    # 初始化 P0 为 None
    P0 = None

    for i in range(len(command)):
        cmd = command[i]
        para_i = para[i]

        if cmd == "M":
            # 记录当前点作为 P0
            P0 = para_i.copy()  # 使用 copy() 防止修改原始数据
        elif cmd == "L":
            if P0 is None:
                raise ValueError(f"'L' 命令出现在第 {i} 个位置，但之前没有 'M' 命令。")

            P1 = para_i.copy()
            # 随机选择一个 t 值
            t = random.choice(t_values)
            # 计算新的 P0
            new_P0 = bezier_curve(P0, P1, t)
            # 添加抖动 [-5, 5] 到每个坐标
            jitter_x = random.randint(-5, 5)
            jitter_y = random.randint(-5, 5)

            # 可选：确保坐标在有效范围内
            new_P0[0] = max(1, min(512, new_P0[0]))
            new_P0[1] = max(1, min(256, new_P0[1]))

            new_P0[0] += jitter_x
            new_P0[1] += jitter_y
            # 更新 para 中的当前点
            para[i] = new_P0
            # 更新 P0 为新的点
            P0 = new_P0
        else:
            raise ValueError(f"不支持的命令 '{cmd}' 在索引 {i}。")

    return para


def svg_path_to_image(path_str, image_size=(256, 256)):
    # 创建SVG内容，指定宽高
    svg_data = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg xmlns="http://www.w3.org/2000/svg" width="{image_size[0]}" height="{image_size[1]}" viewBox="0 0 {image_size[0]} {image_size[1]}">
    <g transform="matrix(1 0 0 1 0 0)">
        <path d="{path_str}" fill="none" stroke="black" stroke-width="5"/>
    </g>
    </svg>
    '''
    # 使用cairosvg将SVG转换为PNG
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

    image = Image.open(io.BytesIO(png_data)).convert("RGBA")  # 使用 RGBA 以保留透明度
    background = Image.new("RGB", image.size, (255, 255, 255))  # 创建白色背景
    image_l = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB").convert("L")

    return np.array(image_l)

 # ---------------------------
# Stroke-wise uniform offset
# ---------------------------
def apply_stroke_offset(command, para, offset_range=(-5, 5)):
    """Apply a uniform random translation to every stroke.

    A stroke is the sequence of drawing commands between two consecutive
    'M' commands (inclusive of the starting 'M'). For each stroke we
    sample \(\Delta_x,\Delta_y\) uniformly from ``offset_range`` and add
    this offset to **all** coordinates in that stroke, simulating layout
    disturbances that occur during handwriting.
    """
    dx, dy = 0, 0
    new_para = []
    for cmd, p in zip(command, para):
        # When a new stroke starts, sample a new offset
        if cmd == 'M':
            dx = random.randint(offset_range[0], offset_range[1])
            dy = random.randint(offset_range[0], offset_range[1])
        # Apply the current stroke offset
        new_x = max(1, min(256, p[0] + dx))
        new_y = max(1, min(256, p[1] + dy))
        new_para.append([new_x, new_y])
    return new_para


def apply_stroke_jitter(para, jitter_range=(-5, 5)):
    """Add independent random jitter to every coordinate.

    Each coordinate \((x,y)\) is shifted by a random integer offset
    \(\Delta_x,\Delta_y \in\) ``jitter_range`` to simulate handwriting
    tremor. Coordinates are clipped to the drawing canvas (1–256).
    """
    new_para = []
    for x, y in para:
        jx = random.randint(jitter_range[0], jitter_range[1])
        jy = random.randint(jitter_range[0], jitter_range[1])
        new_x = max(1, min(256, x + jx))
        new_y = max(1, min(256, y + jy))
        new_para.append([new_x, new_y])
    return new_para

# ---------------------------
# Global translation offset
# ---------------------------

def apply_translation(para, offset_range=(-5, 5)):
    """Apply the same random translation to all coordinates.

    A single \(\Delta_x,\Delta_y\) is sampled from *offset_range* and added
    to every coordinate, simulating global character translation.
    """
    dx = random.randint(offset_range[0], offset_range[1])
    dy = random.randint(offset_range[0], offset_range[1])
    translated = []
    for x, y in para:
        new_x = max(1, min(256, x + dx))
        new_y = max(1, min(256, y + dy))
        translated.append([new_x, new_y])
    return translated


# ---------------------------
# Uniform resampling of SVG sequences
# ---------------------------
# NOTE: numpy 已在文件顶部以 `np` 引入，这里无需再次导入

def _uniform_resample_para(para, target_len=256):
    """Resample a list of coordinates to *target_len* uniformly.

    When up-sampling we linearly interpolate between neighboring points;
    when down-sampling we pick equally spaced positions and still use
    interpolation so the output length is exactly *target_len*.
    Coordinates are rounded to int at the end.
    """
    if len(para) == 0:
        return para

    if len(para) == target_len:
        # Ensure int type
        return [[int(round(x)), int(round(y))] for x, y in para]

    pts = np.array(para, dtype=float)
    n = len(pts)
    idxs = np.linspace(0, n - 1, target_len)
    resampled = []
    for idx in idxs:
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, n - 1)
        t = idx - i0
        pt = (1 - t) * pts[i0] + t * pts[i1]
        resampled.append([int(round(pt[0])), int(round(pt[1]))])

    return resampled


def uniform_resample_svg(svg_dict, target_len=256):
    """Return a new svg_dict uniformly resampled to *target_len* points."""
    new_para = _uniform_resample_para(svg_dict["para"], target_len)
    new_cmd = ["M"] + ["L"] * (target_len - 1) if target_len > 0 else []
    return {"command": new_cmd, "para": new_para}


# 通用重采样接口: 支持单个 dict 或 dict 列表
def resample_svg(data, target_len=256):
    """Uniformly resample SVG drawing parameters to *target_len*.

    Parameters
    ----------
    data : dict | list[dict]
        SVG 描述，或包含多个 SVG 的列表。每个 dict 需要包含 `command` 与 `para` 字段。
    target_len : int, optional
        期望的序列长度，默认 256。

    Returns
    -------
    Same type as *data*
        重采样后的结果，保持输入的结构 (dict 或 list)。
    """
    def _process_single(svg):
        """Process single svg dict according to new rule."""
        cmds = svg.get("command", [])
        paras = svg["para"]

        # 1) Build coordinate list with sentinel 0 0 before every 'M'
        coords = []
        fixed_idx = []  # indices that should stay untouched (sentinel & first pt)
        for i, (c, p) in enumerate(zip(cmds, paras)):
            if c == "M":
                # insert sentinel
                fixed_idx.append(len(coords))  # sentinel index
                coords.append([0, 0])
                # insert starting coordinate
                fixed_idx.append(len(coords))
                coords.append(p)
            else:  # 'L'
                coords.append(p)

        if not cmds:  # if commands missing, assume already coord list
            coords = paras.copy()
            fixed_idx = []

        # 2) Separate fixed and variable part
        fixed_count = len(fixed_idx)
        var_coords = [coords[i] for i in range(len(coords)) if i not in fixed_idx]

        # Determine new variable length
        target_var_len = max(0, target_len - fixed_count)

        # Resample variable coordinates to target_var_len (or truncate/pad)
        if target_var_len == 0:
            resampled_var = []
        else:
            if len(var_coords) == 0:
                resampled_var = []
            else:
                resampled_var = _uniform_resample_para(var_coords, target_var_len)

        # Reconstruct full list preserving order
        result_para = []
        var_iter = iter(resampled_var)
        for idx in range(len(coords)):
            if idx in fixed_idx:
                result_para.append(coords[idx])
            else:
                try:
                    result_para.append(next(var_iter))
                except StopIteration:
                    break  # in case target_var_len < original variable positions

        # If we still have remaining resampled_var (when target_var_len > variable_positions)
        # append them at the end
        result_para.extend(list(var_iter))

        # If length mismatch, pad/truncate
        if len(result_para) < target_len:
            # pad with last coordinate
            pad_pt = result_para[-1] if result_para else [0, 0]
            while len(result_para) < target_len:
                result_para.append(pad_pt)
        else:
            result_para = result_para[:target_len]

        return result_para

    if isinstance(data, dict):
        return _process_single(data)
    elif isinstance(data, list):
        return [_process_single(d) for d in data]
    else:
        raise TypeError("data must be a dict or list of dicts")


class handSvgDataset(Dataset):
    def __init__(self, lmdbPath):

        self.sqeLen = 256  # 每个字符包含绘图参数的数量

        print("load lmdb....\n")

        self.clssNum = 0

        # lmdbPaht="../getFont/lmdbdata/lmdb"
        self.env = self.load_lmdb(lmdbPath)

        self.className, self.class_to_idx, self.allFile, self.class_to_files = self._make_dataset()

        self.apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])



    def load_lmdb(self, lmdb_path):
        """
        load_lmdb
        """
        lmdb_path = os.path.join(lmdb_path)
        env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        return env

    def _make_dataset(self):
        # 从数据库中获取类名列表
        with self.env.begin() as txn:
            key = "className"
            value_bytes = txn.get(key.encode('utf-8'))

            if value_bytes is not None:
                className = pickle.loads(value_bytes)

        self.clssNum = len(className)
        print('number of fonts: ', self.clssNum)

        # 创建类名到索引的映射
        class_to_idx = {cls_name: i for i, cls_name in enumerate(className)}
        print(class_to_idx)

        allFile = []

        # 初始化类名到文件列表的字典
        class_to_files = {cls_name: [] for cls_name in className}

        # 遍历数据库中的所有文件，分类存储
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    key_str = key.decode('utf-8')
                    if key_str != "className":
                        allFile.append(key_str)

                        # 提取类名（假设类名是下划线前的部分）
                        class_name = key_str.split('_')[0]

                        # 将文件添加到对应类的列表中
                        if class_name in class_to_files:
                            class_to_files[class_name].append(key_str)
                        else:
                            # 如果类名不在class_to_files中，可能是新的类名
                            class_to_files[class_name] = [key_str]

        print("sample num", len(allFile))

        # 打印每个类的文件数量（可选）
        for cls, files in class_to_files.items():
            print(f"Class '{cls}' has {len(files)} files.")

        # 返回类名列表、类名到索引的映射、所有文件列表以及类名到文件列表的字典
        return className, class_to_idx, allFile, class_to_files



    def bulidInputData(self, input_dict, className=None):

        # 伸缩笔画
        input_dict["para"] = felax_line(input_dict["command"], input_dict["para"])

        # Stroke Offset: 为每一笔画施加统一的随机平移扰动
        input_dict["para"] = apply_stroke_offset(input_dict["command"], input_dict["para"])



        # 将笔画连接
        processed = simulate_in_coon_handwritten(input_dict)

        # Strokes Jitter: 对每个坐标随机抖动 (在连接后再抖动)
        processed["para"] = apply_stroke_jitter(processed["para"])

        # Translation: 全局随机平移（放在抖动之后）
        processed["para"] = apply_translation(processed["para"])

        return processed

    def getStyle(self, className, num=1):
        num = 3
        files = self.class_to_files.get(className, []).copy()
        key = random.sample(files, k=num)

        image_size = (256, 256)
        stacked_images = np.zeros((image_size[0], image_size[1], num), dtype=np.uint8)
        styleSeq_list = []

        with self.env.begin() as txn:
            for i, k in enumerate(key):
                InputS = pickle.loads(txn.get(k.encode('utf-8')))
                styleSeq_list.append(InputS)

                # rasterise for visual reference
                paths = "".join(f"{cmd} {p[0]} {p[1]} " for cmd, p in zip(InputS['command'], InputS['para']))
                img = svg_path_to_image(paths, image_size)
                stacked_images[:, :, i] = img

        return styleSeq_list, stacked_images

    def __len__(self):

        return len(self.allFile)

    def __getitem__(self, index):
        #try:
            key = self.allFile[index]

            with self.env.begin() as txn:
                value_bytes = txn.get(key.encode('utf-8'))

                if value_bytes is not None:
                    path = pickle.loads(value_bytes)

            split_parts = key.split("_")
            charName = split_parts[1]  # 获取当前是那个字符
            className = split_parts[0]  # 获取该字符是那个字体类别
            classIndex = self.class_to_idx[className]  # 转换为类别的索引

            # 创建输入数据
            input_data = self.bulidInputData(path, className)

            # 获取风格参考字符（返回列表和图像）
            styleSeq_list, stacked_images = self.getStyle(className, num=3)

            target=resample_svg(path, target_len=256)

            input_data = resample_svg(input_data, target_len=256)

            styleSeq_list = resample_svg(styleSeq_list, target_len=256)

            # -------------
            # Convert to tensors so that the DataLoader collate function
            # produces batched tensors instead of lists.
            #   input_data:  (256, 2)
            #   target:      (256, 2)
            #   styleSeq:    (num_style, 256, 2)  -> here num_style = 3
            # -------------
            # 归一化到 [-1,1] 区间，缓解损失量纲过大问题
            norm = 127.5
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            styles_tensor = torch.tensor(styleSeq_list, dtype=torch.float32)

            input_tensor  = (input_tensor  - norm) / norm
            target_tensor = (target_tensor - norm) / norm
            styles_tensor = (styles_tensor - norm) / norm

            # Return tensors for model consumption
            return (
                input_tensor,
                target_tensor,
                styles_tensor,
                self.apply_transform(stacked_images),
            )

        # except Exception as e:
        #     print(f"An error occurred while processing data item {index}: {e}")
        #     #处理错误，例如返回一个默认的数据项或None
        #     #return default_data_item
        #     #或者
        #     new_idx = random.randint(0, len(self.allFile) - 1)
        #     return self.__getitem__(new_idx)


if __name__ == '__main__':
    refPath = "./cr_mapping.json"
    lmdbPath = "./DataPreparation/LMDB/lmdb"
    dataset = handSvgDataset(lmdbPath)
    from torch.utils.data.dataloader import default_collate


    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, data in enumerate(dataloader):
        # Do something with the data
        # print(path)
        # print(target)
        inputs, targets, styles, images = data
        print(inputs.shape)
        print("#" * 10, len(inputs[0]))
        print("images batch:", images.shape)
        print(styles)

        break
