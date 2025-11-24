#!/bin/bash
# 
# 示例脚本：从txt文件提取颗粒ID，然后从tbl文件提取对应的行
# 
# 使用方法：
#   bash extract_tbl_example.sh
# 
# 或者修改下面的变量后运行：
#   bash extract_tbl_example.sh <particles_txt> <input_tbl> <output_tbl>

# 设置变量（可以根据实际情况修改）
PARTICLES_TXT="${1:-particles.txt}"           # 输入的颗粒路径txt文件
INPUT_TBL="${2:-all_particles.tbl}"           # 输入的Dynamo tbl文件
OUTPUT_TBL="${3:-filtered_particles.tbl}"     # 输出的tbl文件

# 检查文件是否存在
if [ ! -f "$PARTICLES_TXT" ]; then
    echo "Error: Particles txt file not found: $PARTICLES_TXT"
    exit 1
fi

if [ ! -f "$INPUT_TBL" ]; then
    echo "Error: Input tbl file not found: $INPUT_TBL"
    exit 1
fi

# 显示使用的文件
echo "=========================================="
echo "Extracting particles from Dynamo tbl file"
echo "=========================================="
echo "Input particles txt: $PARTICLES_TXT"
echo "Input tbl file:      $INPUT_TBL"
echo "Output tbl file:     $OUTPUT_TBL"
echo ""

# 执行提取操作
python utils/2d_projection/analyze_results.py \
    --extract-tbl-by-txt \
    -t "$PARTICLES_TXT" \
    --tbl "$INPUT_TBL" \
    --output-tbl "$OUTPUT_TBL"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Success! Filtered tbl file saved to:"
    echo "  $OUTPUT_TBL"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Failed to extract particles"
    echo "=========================================="
    exit 1
fi

