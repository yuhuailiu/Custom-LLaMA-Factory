"""
独立的代码评测脚本
可以直接拿到其他工程中使用，无需依赖其他模块
预处理逻辑与原评测工程完全一致

包含的关键指标：
    - 相似度: 基于编辑距离的整体相似度（0-100）
    - 代码召回率: ground truth中有多少被生成（0-100）
    - 代码采纳率: 生成代码中有多少是正确的（0-100）
    - F1_score: F1分数（0-1）
    - 首行命中率 ~ 前5行命中率: 前N行是否连续命中（0或100）
    - 生成代码行数: 生成代码的变更行数

使用方法:
    1. 批量JSONL评测:
        python code_eval_standalone.py --input data.jsonl --output result.xlsx
        
    2. 在代码中调用:
        from code_eval_standalone import eval_jsonl
        eval_jsonl('input.jsonl', 'output.xlsx')

依赖:
    - pandas
    - difflib (Python标准库)
    - re (Python标准库)
    - json (Python标准库)
"""

import re
import json
import difflib
import argparse
from difflib import unified_diff
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ==================== 预处理函数（与原脚本一致） ====================

def remove_comments_and_docstrigns(source: str) -> str:
    """
    移除代码中的注释和文档字符串
    支持单行注释 (//) 和多行注释 (/* */)
    保留字符串中的注释标记
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "
        else:
            return s
    
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != '':
            temp.append(x)
    return '\n'.join(temp)


def format_arkTs_code(arkts_code: str) -> List[str]:
    """
    格式化代码：去除每行首尾空白，移除空行
    返回非空行的列表
    """
    response_lines = arkts_code.split('\n')
    for i in range(len(response_lines)):
        response_lines[i] = response_lines[i].strip()
    
    response_lines_no_empty = []
    for r in response_lines:
        if len(r) > 0:
            response_lines_no_empty.append(r)
    return response_lines_no_empty


def extract_code(text: str, language: str = None) -> Tuple[str, int]:
    """
    从Markdown代码块中提取代码
    """
    if language:
        pattern = re.compile(rf'```{language}([\s\S]*?)```', re.IGNORECASE | re.DOTALL)
    else:
        pattern = re.compile(r'```(?:arkts|cpp|c|java|python|typescript|ts|javascript|js)?\s*([\s\S]*?)```',
                              re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return "".join(matches), 0
    return text, 1


# ==================== Diff相关函数（与原脚本一致） ====================

def extract_differences(diff_string: str) -> Tuple[str, int]:
    """
    从unified diff输出中提取差异内容
    返回: (差异文本, 变更行数)
    """
    lines = diff_string.strip().split('\n')
    changes = []
    i = 0

    while i < len(lines):
        if lines[i].startswith('@@') or lines[i].startswith('---'):
            i += 1
            continue
        if lines[i].strip() == '+' or lines[i].strip() == '-':
            i += 1
            continue

        if lines[i].startswith('-'):
            old_block = []
            while i < len(lines) and lines[i].startswith('-'):
                old_block.append(lines[i][1:].strip())
                i += 1
            
            new_block = []
            while i < len(lines) and lines[i].startswith('+'):
                new_block.append(lines[i][1:].strip())
                i += 1
            
            paired_length = min(len(old_block), len(new_block))
            for idx in range(paired_length):
                changes.append({
                    'type': 'modified',
                    'old_line': old_block[idx],
                    'new_line': new_block[idx]
                })
            for idx in range(paired_length, len(old_block)):
                changes.append({
                    'type': 'deleted',
                    'old_line': old_block[idx]
                })
            for idx in range(paired_length, len(new_block)):
                changes.append({
                    'type': 'added',
                    'new_line': new_block[idx]
                })
        elif lines[i].startswith('+'):
            changes.append({
                'type': 'added',
                'new_line': lines[i][1:].strip()
            })
            i += 1
        else:
            i += 1
    
    add_lines = []
    delete_lines = []
    for change in changes:
        if change['type'] == 'modified' or change['type'] == 'added':
            add_line = "+ " + change['new_line']
            if add_line.strip() != "+":
                add_lines.append(add_line)
        elif change['type'] == 'deleted':
            if not change['old_line'].strip():
                continue
            delete_line = "- " + change['old_line']
            delete_lines.append(delete_line)
    
    diff_text = "\n".join(delete_lines) + "\n" + "\n".join(add_lines) + "\n"
    return diff_text, len(add_lines) + len(delete_lines)


def find_line_numbers(code_short: str, code_long_add: str, code_long_delete: str) -> Dict:
    """
    为diff代码添加行号映射
    """
    lines_long_add = code_long_add.split('\n')
    lines_long_del = code_long_delete.split('\n')
    lines_short_del_lst = [line.strip()[2:] for line in code_short.split('\n') if line.strip().startswith('-')]
    lines_short_add_lst = [line.strip()[2:] for line in code_short.split('\n') if line.strip().startswith('+')]

    line_map = {}
    del_lst = []

    del_short_lines_count = defaultdict(int)
    add_short_lines_count = defaultdict(int)
    for line in lines_short_del_lst:
        del_short_lines_count[line] += 1
    for line in lines_short_add_lst:
        add_short_lines_count[line] += 1
    
    for i, line in enumerate(lines_long_del):
        line_content = line.strip()
        if line_content and line_content in del_short_lines_count:
            code_content = f"- {line_content}"
            if code_content not in line_map:
                line_map[code_content] = []
            line_map[code_content].append(i + 1)
            if del_short_lines_count.get(line_content, 0) > 0:
                del_lst.append(i + 1)
                del_short_lines_count[line_content] -= 1
    
    for i, line in enumerate(lines_long_add):
        line_content = line.strip()
        if line_content and line_content in add_short_lines_count:
            num_del = sum(1 for d in del_lst if d < (i + 1))
            adjusted_line = i + 1 + num_del
            code_content = f"+ {line_content}"
            if code_content not in line_map:
                line_map[code_content] = []
            line_map[code_content].append(adjusted_line)
    return line_map


def modify_code_with_line_numbers(diff_code_diff: str, line_map: Dict) -> str:
    """
    为diff代码添加行号
    """
    lines_short_del_lst = [line.strip() for line in diff_code_diff.split('\n') if line.strip().startswith('-')]
    lines_short_add_lst = [line.strip() for line in diff_code_diff.split('\n') if line.strip().startswith('+')]
    result = []

    previous_line_number = 0
    for line in lines_short_del_lst:
        if line in line_map:
            index_lst = line_map[line]
            index_lst.sort()
            for index in index_lst:
                if index > previous_line_number:
                    result.append(f"{index} {line}")
                    previous_line_number = index
                    break
        else:
            # 如果找不到行号，直接添加
            result.append(line)
    
    previous_line_number = 0
    for line in lines_short_add_lst:
        if line in line_map:
            index_lst = line_map[line]
            index_lst.sort()
            for index in index_lst:
                if index > previous_line_number:
                    result.append(f"{index} {line}")
                    previous_line_number = index
                    break
        else:
            # 如果找不到行号，直接添加
            result.append(line)
    return "\n".join(result)


# ==================== 编辑距离相关（与原脚本一致） ====================

def edit_distance(str1: str, str2: str) -> int:
    """
    计算两个字符串的编辑距离（Levenshtein距离）
    """
    m, n = len(str1), len(str2)
    dp = [0] * (n + 1)
    for i in range(m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            curr = dp[j]
            if i > 0 and str1[i - 1] == str2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(dp[j - 1], dp[j], prev) + 1
            prev = curr
    return dp[n]


def compute_similarity(expected_code: str, generated_code: str) -> float:
    """
    基于编辑距离计算相似度（0-100）
    """
    if not expected_code and not generated_code:
        return 100.0
    if not expected_code or not generated_code:
        return 0.0
    
    edit_dist = edit_distance(expected_code, generated_code)
    max_len = max(len(expected_code), len(generated_code))
    similarity = 1 - edit_dist / max_len
    return round(similarity, 4) * 100


def num_lines_difference(expected_code: str, generated_code: str) -> int:
    """
    计算两段代码的行数差异的绝对值
    """
    expected_lines = len(expected_code.split('\n'))
    generated_lines = len(generated_code.split('\n'))
    return abs(expected_lines - generated_lines)


# ==================== 行级别匹配（与原脚本一致） ====================

def count_hit_lines(lines1: List[str], lines2: List[str]) -> int:
    """
    使用difflib计算两个代码行列表之间的重复行数
    """
    diff = difflib.Differ()
    diffs = list(diff.compare(lines1, lines2))
    
    duplicate_count = 0
    for diff_line in diffs:
        if re.match(r'^[-+\?] ', diff_line):
            continue
        duplicate_count += 1
    
    return duplicate_count


# ==================== F1分数 ====================

def f1_score(precision: float, recall: float) -> float:
    """
    计算F1分数
    """
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


# ==================== 核心评测函数 ====================

def process_item(item: Dict, 
                 generated_key: str = 'First_Chunk',
                 ground_truth_key: str = 'real_ground_truth',
                 user_excerpt_key: str = 'User_Excerpt',
                 tag_key: str = 'tag',
                 eval_fim: bool = True) -> Dict:
    """
    处理单条数据
    
    参数:
        item: 数据字典
        generated_key: 生成代码的字段名
        ground_truth_key: ground truth代码的字段名
        user_excerpt_key: 原始代码片段的字段名（仅eval_fim=True时使用）
        tag_key: 标签字段名
        eval_fim: 是否使用FIM模式（需要User_Excerpt和diff处理）
                  - True: 使用原脚本的FIM diff预处理逻辑
                  - False: 直接比较generated和ground_truth，无需diff处理
    """
    # 获取生成代码和ground truth
    generated = str(item.get(generated_key, ''))
    generated = generated.replace("<|editable_region_start|>\n", "").replace("\n<|editable_region_end|>", "")
    expected = str(item.get(ground_truth_key, ''))
    
    # 预处理：根据模式处理生成代码和ground truth
    if eval_fim:
        # FIM模式：直接移除注释和文档字符串
        generated = remove_comments_and_docstrigns(generated).replace('\r\n', '\n')
        expected = remove_comments_and_docstrigns(expected).replace('\r\n', '\n')
    else:
        # 非FIM模式：先提取代码块，再移除注释和文档字符串
        generated, _ = extract_code(generated)
        generated = remove_comments_and_docstrigns(generated).replace('\r\n', '\n')
        expected, _ = extract_code(expected)
        expected = remove_comments_and_docstrigns(expected).replace('\r\n', '\n')
    
    if not eval_fim:
        # ========== 非FIM模式：使用diff预处理 ==========
        # 提取原始可编辑区域
        user_excerpt = str(item.get(user_excerpt_key, ''))
        origin_left_index = user_excerpt.find("<|editable_region_start|>")
        origin_right_index = user_excerpt.find("<|editable_region_end|>")
        
        if origin_left_index >= 0 and origin_right_index >= 0:
            origin_editable_region = remove_comments_and_docstrigns(
                user_excerpt[origin_left_index + len("<|editable_region_start|>"):origin_right_index]
            ).replace('<|user_cursor_is_here|>', '') + "\n"
        else:
            # 如果没有editable region标记，使用整个user_excerpt
            origin_editable_region = remove_comments_and_docstrigns(user_excerpt).replace('<|user_cursor_is_here|>', '') + "\n"
        
        # 处理特殊标记
        generated_for_diff = generated + '\n'
        expected_for_diff = expected.replace(
            '<|editable_region_start|>', '').replace(
            '<|editable_region_end|>', '').replace(
            '<|start_of_file|>', '').replace(
            '<|end_of_file|>', '') + '\n'
        
        # 计算diff
        diff_gen = unified_diff(
            origin_editable_region.splitlines(keepends=True), 
            generated_for_diff.splitlines(keepends=True), 
            fromfile='origin_editable_region', 
            tofile='generated', 
            n=100, 
            lineterm=""
        )
        diff_text_gen = ''.join(diff_gen)
        generated_code_diff, gen_num_of_newlines = extract_differences(diff_text_gen)
        
        diff_exp = unified_diff(
            origin_editable_region.splitlines(keepends=True), 
            expected_for_diff.splitlines(keepends=True), 
            fromfile='origin_editable_region', 
            tofile='expected', 
            n=100, 
            lineterm=""
        )
        diff_text_exp = ''.join(diff_exp)
        expected_code_diff, exp_num_of_newlines = extract_differences(diff_text_exp)
        
        # 对于预测场景或编辑场景，添加行号
        tag = item.get(tag_key, '')
        if tag and (str(tag).startswith('预测场景') or "编辑" in str(tag)):
            try:
                line_map_gen = find_line_numbers(generated_code_diff, generated_for_diff, origin_editable_region)
                line_map_truth = find_line_numbers(
                    expected_code_diff, 
                    expected_for_diff[1:] if expected_for_diff.startswith('\n') else expected_for_diff, 
                    origin_editable_region
                )
                generated_code_diff = modify_code_with_line_numbers(generated_code_diff, line_map_gen)
                expected_code_diff = modify_code_with_line_numbers(expected_code_diff, line_map_truth)
            except Exception:
                pass  # 如果行号处理失败，继续使用原始diff
        
        # 格式化diff后的代码
        if expected_code_diff is not None and generated_code_diff is not None:
            expected_data = format_arkTs_code(expected_code_diff)
            generated_data = format_arkTs_code(generated_code_diff)
        else:
            expected_data = []
            generated_data = []
    else:
        # ========== FIM模式：直接比较generated和ground_truth ==========
        # 格式化代码
        generated_data = format_arkTs_code(generated)
        expected_data = format_arkTs_code(expected)
        
        # 非FIM模式下，生成代码行数就是格式化后的行数
        gen_num_of_newlines = len(generated_data)
        exp_num_of_newlines = len(expected_data)
    
    # 重新组合为字符串
    post_expected_code = '\n'.join(expected_data)
    post_generated_code = '\n'.join(generated_data)
    
    # 初始化指标
    first_line_hit = 0
    second_line_hit = 0
    third_line_hit = 0
    fourth_line_hit = 0
    fifth_line_hit = 0
    hit_line_count = 0
    hit50 = 0
    line_hit = 0
    line_hit1 = 0
    similarity = 0
    
    if len(generated_data) > 0 and len(expected_data) > 0:
        # 计算相似度
        similarity = compute_similarity(post_expected_code, post_generated_code)
        
        # 计算1-5行命中率（与原脚本逻辑一致）
        if generated_data[0] == expected_data[0]:
            first_line_hit = 100
        if len(generated_data) > 2 and len(expected_data) > 2 and generated_data[1] == expected_data[1] and first_line_hit == 100:
            second_line_hit = 100
        if len(generated_data) > 3 and len(expected_data) > 3 and generated_data[2] == expected_data[2] and first_line_hit == 100 and second_line_hit == 100:
            third_line_hit = 100
        if len(generated_data) > 4 and len(expected_data) > 4 and generated_data[3] == expected_data[3] and first_line_hit == 100 and second_line_hit == 100 and third_line_hit == 100:
            fourth_line_hit = 100
        if len(generated_data) > 5 and len(expected_data) > 5 and generated_data[4] == expected_data[4] and first_line_hit == 100 and second_line_hit == 100 and third_line_hit == 100 and fourth_line_hit == 100:
            fifth_line_hit = 100
        
        # 计算命中行数
        hit_line_count = count_hit_lines(expected_data, generated_data)
        line_hit = 100.0 * hit_line_count / len(expected_data)
        line_hit1 = 100.0 * hit_line_count / len(generated_data)
        
        if line_hit >= 50:
            hit50 = 100
    
    # 构建结果
    result = {
        '相似度': round(similarity, 2),
        '代码召回率': round(line_hit, 2),
        '代码采纳率': round(line_hit1, 2),
        'F1_score': round(f1_score(line_hit / 100.0, line_hit1 / 100.0), 4),
        '首行命中率': first_line_hit,
        '前2行命中率': second_line_hit,
        '前3行命中率': third_line_hit,
        '前4行命中率': fourth_line_hit,
        '前5行命中率': fifth_line_hit,
        '生成代码行数': gen_num_of_newlines,
        'GT代码变更数': exp_num_of_newlines,
        '行数绝对值差': num_lines_difference(post_expected_code, post_generated_code),
        'hit50': hit50,
        'post_ground_truth': post_expected_code,
        'post_generated_code': post_generated_code
    }
    
    return result


def eval_jsonl(input_path: str, 
               output_path: str,
               generated_key: str = 'First_Chunk',
               ground_truth_key: str = 'real_ground_truth',
               user_excerpt_key: str = 'User_Excerpt',
               tag_key: str = 'tag',
               eval_fim: bool = True) -> Tuple[Dict, Dict, List[Dict]]:
    """
    批量评测JSONL文件
    
    参数:
        input_path: 输入JSONL文件路径
        output_path: 输出Excel文件路径
        generated_key: 生成代码的字段名
        ground_truth_key: ground truth代码的字段名
        user_excerpt_key: 原始代码片段的字段名（仅eval_fim=True时使用）
        tag_key: 标签字段名
        eval_fim: 是否使用FIM模式
                  - True (默认): 使用FIM diff预处理（需要User_Excerpt字段）
                  - False: 直接比较generated和ground_truth，无需diff处理
    
    返回:
        (总体平均值字典, 按tag分组的统计字典, 详细结果列表)
    """
    try:
        import pandas as pd
    except ImportError:
        print("错误: 批量评测需要安装pandas库")
        print("请运行: pip install pandas openpyxl")
        return {}, {}, []
    
    # 读取数据并预处理
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # 预处理已移至process_item中
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"跳过无效JSON行: {e}")
                continue
    
    print(f"读取到 {len(data)} 条数据")
    print(f"评测模式: {'FIM模式（直接比较）' if eval_fim else '非FIM模式（使用diff预处理）'}")
    
    # 评测每条数据
    eval_results = []
    for idx, item in enumerate(data):
        metrics = process_item(
            item, 
            generated_key=generated_key,
            ground_truth_key=ground_truth_key,
            user_excerpt_key=user_excerpt_key,
            tag_key=tag_key,
            eval_fim=eval_fim
        )
        
        # 合并结果
        result = {
            'index': idx,
            generated_key: item.get(generated_key, ''),
            ground_truth_key: item.get(ground_truth_key, ''),
            tag_key: item.get(tag_key, ''),
            **metrics
        }
        
        # 保留原始数据中的其他字段
        for key in item:
            if key not in result:
                result[key] = item[key]
        
        eval_results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"已评测 {idx + 1}/{len(data)} 条数据")
    
    # 计算总体平均值
    columns_to_average = ['相似度', '代码召回率', '代码采纳率', 'F1_score',
                          '首行命中率', '前2行命中率', '前3行命中率', '前4行命中率', '前5行命中率',
                          '生成代码行数', 'GT代码变更数', '行数绝对值差']
    
    df = pd.DataFrame(eval_results)
    
    average_values = {}
    for col in columns_to_average:
        if col in df.columns:
            average_values[col] = round(df[col].mean(), 2)
        else:
            average_values[col] = 0
    
    print("\n总体评测结果:")
    print(f"  相似度: {average_values.get('相似度', 0):.2f}")
    print(f"  代码召回率: {average_values.get('代码召回率', 0):.2f}%")
    print(f"  代码采纳率: {average_values.get('代码采纳率', 0):.2f}%")
    print(f"  F1_score: {average_values.get('F1_score', 0):.4f}")
    print(f"  首行命中率: {average_values.get('首行命中率', 0):.2f}%")
    print(f"  前2行命中率: {average_values.get('前2行命中率', 0):.2f}%")
    print(f"  前3行命中率: {average_values.get('前3行命中率', 0):.2f}%")
    print(f"  前4行命中率: {average_values.get('前4行命中率', 0):.2f}%")
    print(f"  前5行命中率: {average_values.get('前5行命中率', 0):.2f}%")
    print(f"  生成代码行数: {average_values.get('生成代码行数', 0):.2f}")
    
    # 按tag分组统计
    tagging_values = {}
    if tag_key in df.columns:
        unique_tags = df[tag_key].dropna().unique()
        print(f"\n发现 {len(unique_tags)} 种不同的tag")
        
        for tag in unique_tags:
            if not tag or tag == '':
                continue
            
            tag_data = df[df[tag_key] == tag]
            tag_stats = {'tag': tag, 'count': len(tag_data)}
            
            for col in columns_to_average:
                if col in tag_data.columns:
                    tag_stats[col] = round(tag_data[col].mean(), 2)
            
            tagging_values[tag] = tag_stats
            print(f"  Tag '{tag}': {len(tag_data)} 条, 相似度: {tag_stats.get('相似度', 0):.2f}, 召回率: {tag_stats.get('代码召回率', 0):.2f}%")
    
    # 保存Excel
    if output_path:
        df_summary = pd.DataFrame([average_values])
        df_detail = pd.DataFrame(eval_results)
        
        with pd.ExcelWriter(output_path) as writer:
            df_summary.to_excel(writer, sheet_name='summary', index=False)
            df_detail.to_excel(writer, sheet_name='detail', index=False)
            
            if tagging_values:
                tag_data_list = list(tagging_values.values())
                df_tagging = pd.DataFrame(tag_data_list)
                df_tagging.to_excel(writer, sheet_name='tagging', index=False)
        
        print(f"\n结果已保存到: {output_path}")
        
        # 同时保存简化的JSONL
        simplified_jsonl_path = output_path.replace('.xlsx', '_simplified.jsonl')
        simplified_columns = [generated_key, ground_truth_key, 'post_generated_code', 'post_ground_truth',
                              tag_key, '相似度', '代码召回率', '代码采纳率', 'F1_score',
                              '首行命中率', '前2行命中率', '前3行命中率', '前4行命中率', '前5行命中率',
                              '生成代码行数']
        
        with open(simplified_jsonl_path, 'w', encoding='utf-8') as f:
            for item in eval_results:
                simplified_item = {k: item.get(k) for k in simplified_columns if k in item}
                f.write(json.dumps(simplified_item, ensure_ascii=False) + '\n')
        
        print(f"简化结果已保存到: {simplified_jsonl_path}")
    
    return average_values, tagging_values, eval_results


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(description='独立的代码评测脚本')
    parser.add_argument('--input', '-i', type=str, help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出Excel文件路径')
    parser.add_argument('--generated-key', type=str, default='First_Chunk',
                        help='生成代码的字段名 (默认: First_Chunk)')
    parser.add_argument('--ground-truth-key', type=str, default='real_ground_truth',
                        help='ground truth代码的字段名 (默认: real_ground_truth)')
    parser.add_argument('--user-excerpt-key', type=str, default='User_Excerpt',
                        help='原始代码片段的字段名 (默认: User_Excerpt, 仅FIM模式需要)')
    parser.add_argument('--tag-key', type=str, default='tag',
                        help='标签字段名 (默认: tag)')
    parser.add_argument('--eval-fim', action='store_true', default=False,  
                        help='使用FIM模式（不需要User_Excerpt字段进行diff预处理，默认关闭）')
    parser.add_argument('--demo', action='store_true', help='运行演示示例')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
    
    if args.input:
        if not args.output:
            args.output = args.input.replace('.jsonl', '_eval.xlsx')
        
        eval_jsonl(
            args.input,
            args.output,
            generated_key=args.generated_key,
            ground_truth_key=args.ground_truth_key,
            user_excerpt_key=args.user_excerpt_key,
            tag_key=args.tag_key,
            eval_fim=args.eval_fim
        )
    else:
        parser.print_help()
        print("\n示例用法:")
        print("  python code_eval_standalone.py --demo")
        print("  python code_eval_standalone.py -i input.jsonl -o output.xlsx")
        print("  python code_eval_standalone.py -i input.jsonl -o output.xlsx --no-eval-fim  # 直接比较模式")
        print("  python code_eval_standalone.py -i input.jsonl --generated-key 'model_output' --ground-truth-key 'answer'")

if __name__ == "__main__":
    eval_jsonl(
        input_path="/data/k8s/lyh/output_models/qwen_7b_full_pretrain_MIX30_gAcc_1_1106_psm200_plus_zeta_sft_1127_eval_test/adoptation_eval_step_135.jsonl",
        output_path="/data/k8s/lyh/output_models/qwen_7b_full_pretrain_MIX30_gAcc_1_1106_psm200_plus_zeta_sft_1127_eval_test/adoptation_eval_step_135.xlsx",
        generated_key="generated_response",
        ground_truth_key="real_ground_truth",
        user_excerpt_key="User_Excerpt",
        tag_key="tag",
        eval_fim=False
    )