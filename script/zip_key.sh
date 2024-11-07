#!/bin/bash

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "用法: $0 <根目录> <模式> [关键字]"
    echo "模式: zip (压缩) 或 unzip (解压)"
    exit 1
fi

# 设置根目录和模式
ROOT_DIR="$1"
MODE="$2"
# 设置可选的关键字
KEYWORD="$3"

# 检查根目录是否存在
if [ ! -d "$ROOT_DIR" ]; then
    echo "错误: 目录 '$ROOT_DIR' 不存在"
    exit 1
fi

# 检查模式是否有效
if [ "$MODE" != "zip" ] && [ "$MODE" != "unzip" ]; then
    echo "错误: 模式必须是 'zip' 或 'unzip'"
    exit 1
fi

# 创建一个临时文件来存储操作信息
temp_file=$(mktemp)

if [ "$MODE" = "zip" ]; then
    # 压缩模式
    if [ -z "$KEYWORD" ]; then
        # 如果没有关键字,压缩根目录下的所有文件夹
        find "$ROOT_DIR" -maxdepth 1 -type d ! -path "$ROOT_DIR" | while read dir; do
            dirname=$(basename "$dir")
            echo "正在压缩: $dir"
            cd "$ROOT_DIR"
            zip -r "${dirname}.zip" "$dirname"
            echo "文件夹: $dir -> 压缩包: ${ROOT_DIR}/${dirname}.zip" >> "$temp_file"
            cd - > /dev/null
        done
    else
        # 如果有关键字,只压缩包含关键字的文件夹
        find "$ROOT_DIR" -maxdepth 1 -type d -name "*${KEYWORD}*" | while read dir; do
            dirname=$(basename "$dir")
            echo "正在压缩: $dir"
            cd "$ROOT_DIR" 
            zip -r "${dirname}.zip" "$dirname"
            echo "文件夹: $dir -> 压缩包: ${ROOT_DIR}/${dirname}.zip" >> "$temp_file"
            cd - > /dev/null
        done
    fi
    echo "压缩完成"
else
    # 解压模式
    if [ -z "$KEYWORD" ]; then
        # 如果没有关键字,解压根目录下的所有zip文件
        find "$ROOT_DIR" -maxdepth 1 -name "*.zip" | while read zipfile; do
            filename=$(basename "$zipfile")
            basename="${filename%.*}"
            echo "正在解压: $zipfile"
            cd "$ROOT_DIR"
            # 检查zip文件是否存在
            if [ ! -f "$filename" ]; then
                echo "错误: zip文件 '$filename' 不存在"
                continue
            fi
            
            # 创建临时目录并解压
            temp_dir=$(mktemp -d)
            unzip -q "$filename" -d "$temp_dir"
            
            # 检查解压后的内容
            if [ -d "$temp_dir/$basename" ]; then
                # 如果存在同名文件夹，直接移动
                mv "$temp_dir/$basename" "./"
            else
                # 否则创建目标文件夹并移动所有内容
                mkdir -p "$basename"
                mv "$temp_dir"/* "$basename/"
            fi
            
            # 清理临时目录
            rm -rf "$temp_dir"
            echo "解压: $zipfile -> ${ROOT_DIR}/${basename}" >> "$temp_file"
            cd - > /dev/null
        done
    else
        # 如果有关键字,只解压包含关键字的zip文件
        find "$ROOT_DIR" -maxdepth 1 -name "*${KEYWORD}*.zip" | while read zipfile; do
            filename=$(basename "$zipfile")
            basename="${filename%.*}"
            echo "正在解压: $zipfile"
            cd "$ROOT_DIR"
            # 检查zip文件是否存在
            if [ ! -f "$filename" ]; then
                echo "错误: zip文件 '$filename' 不存在"
                continue
            fi
            
            # 创建临时目录并解压
            temp_dir=$(mktemp -d)
            unzip -q "$filename" -d "$temp_dir"
            
            # 检查解压后的内容
            if [ -d "$temp_dir/$basename" ]; then
                # 如果存在同名文件夹，直接移动
                mv "$temp_dir/$basename" "./"
            else
                # 否则创建目标文件夹并移动所有内容
                mkdir -p "$basename"
                mv "$temp_dir"/* "$basename/"
            fi
            
            # 清理临时目录
            rm -rf "$temp_dir"
            echo "解压: $zipfile -> ${ROOT_DIR}/${basename}" >> "$temp_file"
            cd - > /dev/null
        done
    fi
    echo "解压完成"
fi

echo "操作记录:"
cat "$temp_file"
rm "$temp_file"