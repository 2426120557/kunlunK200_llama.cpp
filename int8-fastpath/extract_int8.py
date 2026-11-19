#!/usr/bin/env python3
"""
从完整调优 patch 中按 hunk 索引抽取 "int8 快路径(非 MoE)" 的核心改动,
生成可 git apply 的精简 patch。

用法:
  python3 extract_int8.py backend backend-full.patch  backend-int8.patch
  python3 extract_int8.py vendor  vendor-full.patch   vendor-int8.patch

抽取范围见同目录 README.md。hunk 索引基于 2026-11-20 备份的 patch,
若 patch 内容变化需重新核对索引。抽取后务必人工审阅 + 编译验证。
"""
import re
import sys

# 每个文件要保留的 hunk 索引(0-based, 按 patch 中 @@ 出现顺序)
BACKEND = {
    "ggml/src/ggml-kunlun/ggml-kunlun.cpp": [0, 2, 11, 13, 14],
}
VENDOR = {
    "include/kunlun/k200.h": [1, 2, 3],
    "src/k200.cpp": [4, 5, 6, 7, 8, 9],
}


def split_sections(text):
    """按 'diff --git' 切分,返回 {path: section_text}"""
    parts = re.split(r'(?=^diff --git )', text, flags=re.M)
    out = {}
    for p in parts:
        if not p.startswith('diff --git'):
            continue
        m = re.match(r'diff --git a/(\S+) b/(\S+)', p)
        if not m:
            continue
        out[m.group(2)] = p
    return out


def select_hunks(section, indices):
    """保留 section 中指定索引的 hunk,头部(diff/index/---/+++)原样保留"""
    split = re.split(r'(?=^@@)', section, flags=re.M)
    head = split[0]
    hunks = split[1:]
    kept = [hunks[i] for i in indices if i < len(hunks)]
    return head + ''.join(kept)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    which, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    sel = VENDOR if which == 'vendor' else BACKEND
    text = open(src, encoding='utf-8').read()
    sections = split_sections(text)
    result = []
    for path, indices in sel.items():
        if path not in sections:
            print(f"WARN: {path} not found, skipped", file=sys.stderr)
            continue
        result.append(select_hunks(sections[path], indices))
    open(dst, 'w', encoding='utf-8').write(''.join(result))
    nhunks = sum(len(re.findall(r'^@@', s, flags=re.M)) for s in result)
    print(f"wrote {dst}: {len(result)} files, {nhunks} hunks")


if __name__ == '__main__':
    main()
