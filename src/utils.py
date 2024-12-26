import torch
import numpy as np
import random
import re
from torch.nn.utils.rnn import pad_sequence

def generate_random_seeds(num_seeds=10, seed_range=(0, 10000)):
    """ランダムにシードを生成"""
    return [random.randint(*seed_range) for _ in range(num_seeds)]

def set_random_seed(seed):
    """ランダムシードを固定するための関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_sections(report_text):
    """
    診断書からFINDINGS、IMPRESSION、もしくは最終セクションを抽出します。
    条件:
    1. FINDINGSとIMPRESSION両方が含まれる場合 -> 両方を抜き出す
    2. FINDINGSだけが含まれる場合 -> FINDINGSを抜き出す
    3. IMPRESSIONが含まれる場合 -> IMPRESSIONを抜き出す
    4. FINDINGSとIMPRESSION両方が含まれない場合 -> 最後の段落を抜き出す

    Args:
        report_text (str): 診断書のテキスト

    Returns:
        str: 抽出されたセクション
    """

    # FINDINGSセクションを探す正規表現
    findings_pattern = re.compile(r'FINDINGS:(.*?)(?:IMPRESSION:|FINAL REPORT|$)', re.DOTALL)
    findings_match = findings_pattern.search(report_text)
    findings = findings_match.group(1).strip() if findings_match else None

    # IMPRESSIONセクションを探す正規表現
    impression_pattern = re.compile(r'IMPRESSION:(.*?)(?:FINAL REPORT|$)', re.DOTALL)
    impression_match = impression_pattern.search(report_text)
    impression = impression_match.group(1).strip() if impression_match else None

    # IMPRESSION内にFINDINGSが含まれているかを確認
    if impression:
        # IMPRESSION セクション内に FINDINGS セクションが含まれているかどうかを確認
        impression_findings_pattern = re.compile(r'FINDINGS:', re.DOTALL)
        impression_contains_findings = impression_findings_pattern.search(impression)
        if impression_contains_findings:
            # IMPRESSION セクションに FINDINGS が含まれている場合は FINDINGS を重複抽出しない
            impression = re.sub(r'FINDINGS:(.*?)(?:IMPRESSION:|FINAL REPORT|$)', '', impression, flags=re.DOTALL).strip()

    # セクションを条件に基づいて返す
    if findings and impression:
        # FINDINGSとIMPRESSIONが両方存在する場合は両方を返す
        return f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
    elif findings:
        # FINDINGSのみが存在する場合はFINDINGSを返す
        return f"FINDINGS:\n{findings}"
    elif impression:
        # IMPRESSIONのみが存在する場合はIMPRESSIONを返す
        return f"IMPRESSION:\n{impression}"
    else:
        # FINDINGSとIMPRESSIONがどちらも存在しない場合は、最後の段落を抜き出す
        paragraphs = report_text.strip().split('\n \n')
        # 空でない段落だけを取り出す
        valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        # 最後の段落を取得
        last_paragraph = valid_paragraphs[-1] if valid_paragraphs else ""
        return last_paragraph
    
# テンソルのリストをパディングする
def pad_tensors(tensor_list):
    return pad_sequence(tensor_list, batch_first=True, padding_value=0)