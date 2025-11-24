#!/bin/bash
# 
# 示例脚本：从star文件提取slice indices，从particle txt提取对应行，提取颗粒ID，从tbl提取对应行
# 
# 使用方法：
#   bash analyze_results.sh <star_file> <particle_txt> <tbl_file> <output_dir>
# 
# 或者修改下面的变量后运行：
#   bash analyze_results.sh

# 设置变量（可以根据实际情况修改）
TomoPanda_ROOT="/fast/AG_Kudryashev/MuWang/github/TomoPANDA-pick"
STAR_FILE="${1:-particles.star}"              # 输入的star文件
PARTICLE_TXT="${2:-particles.txt}"             # 输入的颗粒路径txt文件
INPUT_TBL="${3:-all_particles.tbl}"            # 输入的Dynamo tbl文件
OUTPUT_DIR="${4:-output}"                       # 输出目录

# 检查文件是否存在
if [ ! -f "$STAR_FILE" ]; then
    echo "Error: Star file not found: $STAR_FILE"
    exit 1
fi

if [ ! -f "$PARTICLE_TXT" ]; then
    echo "Error: Particle txt file not found: $PARTICLE_TXT"
    exit 1
fi

if [ ! -f "$INPUT_TBL" ]; then
    echo "Error: Input tbl file not found: $INPUT_TBL"
    exit 1
fi

# 显示使用的文件
echo "=========================================="
echo "Processing star, particle txt, and tbl files"
echo "=========================================="
echo "Star file:          $STAR_FILE"
echo "Particle txt file:  $PARTICLE_TXT"
echo "Input tbl file:     $INPUT_TBL"
echo "Output directory:   $OUTPUT_DIR"
echo ""

# 执行处理操作
python $TomoPanda_ROOT/utils/2d_projection/analyze_results.py \
    --process-star-txt-tbl \
    -s "$STAR_FILE" \
    -t "$PARTICLE_TXT" \
    --tbl "$INPUT_TBL" \
    --output-dir "$OUTPUT_DIR"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Success! All files saved to:"
    echo "  $OUTPUT_DIR/"
    echo ""
    echo "Generated files:"
    echo "  - index.txt"
    echo "  - $(basename "$PARTICLE_TXT" | sed 's/\(.*\)\.\(.*\)/\1.processed.\2/')"
    echo "  - $(basename "$INPUT_TBL" | sed 's/\(.*\)\.\(.*\)/\1.processed.\2/')"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Failed to process files"
    echo "=========================================="
    exit 1
fi

