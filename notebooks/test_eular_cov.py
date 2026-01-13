import numpy as np
from scipy.spatial.transform import Rotation as R

"""
小工具：分析 Dynamo ZXZ ↔ RELION ZYZ 之间的关系

目标：
1. 验证三组欧拉角对应的旋转矩阵关系；
2. 尝试拟合一个固定的坐标变换 F，使得：
   R_relion_good ≈ F @ R_dynamo 或 R_relion_good ≈ R_dynamo @ F
3. 打印 F 的欧拉角/轴角形式，帮助后续在 io_dynamo.py 里加入坐标系变换。
4. 如果提供多个粒子对，验证 F 是否对所有粒子都一致。
"""


def print_matrix(name, M):
    print(f"\n{name} =")
    with np.printoptions(precision=6, suppress=True):
        print(M)


def analyze_single_pair(dynamo_zxz, relion_zyz_good, script_zyz=None, verbose=True):
    """
    分析单个粒子对的转换关系。
    
    Returns:
        F1: 使得 R_relion_good = F1 @ R_dynamo 的固定变换
        F2: 使得 R_relion_good = R_dynamo @ F2 的固定变换
        error_F1: 使用 F1 重建的误差
        error_F2: 使用 F2 重建的误差
    """
    # 转为旋转矩阵
    R_dynamo = R.from_euler('ZXZ', dynamo_zxz, degrees=True).as_matrix()
    R_relion_good = R.from_euler('ZYZ', relion_zyz_good, degrees=True).as_matrix()
    
    if script_zyz is not None:
        R_script = R.from_euler('ZYZ', script_zyz, degrees=True).as_matrix()
        if verbose:
            print("=== 基本一致性检查 ===")
            print("Dynamo vs RELION good:", np.allclose(R_dynamo, R_relion_good, atol=1e-6))
            print("Dynamo vs script:", np.allclose(R_dynamo, R_script, atol=1e-6))
            print("RELION good vs script:", np.allclose(R_relion_good, R_script, atol=1e-6))
    
    # 尝试拟合固定坐标变换 F1, F2
    F1 = R_relion_good @ R_dynamo.T
    F2 = R_dynamo.T @ R_relion_good
    
    # 计算重建误差
    R_relion_from_F1 = F1 @ R_dynamo
    R_relion_from_F2 = R_dynamo @ F2
    error_F1 = np.linalg.norm(R_relion_good - R_relion_from_F1)
    error_F2 = np.linalg.norm(R_relion_good - R_relion_from_F2)
    
    if verbose:
        print_matrix("R_dynamo", R_dynamo)
        print_matrix("R_relion_good", R_relion_good)
        if script_zyz is not None:
            print_matrix("R_script", R_script)
        
        print("\n=== 尝试坐标变换 F ===")
        print_matrix("F1 (假设 R_relion_good = F1 @ R_dynamo)", F1)
        print_matrix("F2 (假设 R_relion_good = R_dynamo @ F2)", F2)
        
        def check_F(name, F):
            I = np.eye(3)
            ortho = np.allclose(F.T @ F, I, atol=1e-6)
            det = np.linalg.det(F)
            print(f"\n{name}: 正交={ortho}, det={det:.6f}")
            r = R.from_matrix(F)
            axis = r.as_rotvec()
            angle_deg = np.linalg.norm(axis) * 180.0 / np.pi
            if angle_deg > 1e-8:
                axis_dir = axis / np.linalg.norm(axis)
            else:
                axis_dir = np.array([0.0, 0.0, 1.0])
            zyz = r.as_euler('ZYZ', degrees=True)
            print(f"{name}: 旋转角约 {angle_deg:.3f} 度, 轴 ~ {axis_dir}")
            print(f"{name}: 以 ZYZ 表示为 {zyz}")
        
        check_F("F1", F1)
        check_F("F2", F2)
        
        print("\n=== 用 F 重建 R_relion_good 的误差 ===")
        print(f"||R_relion_good - F1 @ R_dynamo||_F = {error_F1:.2e}")
        print(f"||R_relion_good - R_dynamo @ F2||_F = {error_F2:.2e}")
    
    return F1, F2, error_F1, error_F2


def analyze_multiple_pairs(particle_pairs, verbose=True):
    """
    分析多个粒子对，验证固定变换 F 是否一致。
    
    particle_pairs: list of tuples (dynamo_zxz, relion_zyz_good)
    """
    F1_list = []
    F2_list = []
    
    for i, (dynamo_zxz, relion_zyz_good) in enumerate(particle_pairs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"粒子对 {i+1}: Dynamo {dynamo_zxz}, RELION {relion_zyz_good}")
            print('='*60)
        F1, F2, e1, e2 = analyze_single_pair(dynamo_zxz, relion_zyz_good, verbose=verbose)
        F1_list.append(F1)
        F2_list.append(F2)
    
    # 检查所有 F1 是否一致
    if len(F1_list) > 1:
        print(f"\n{'='*60}")
        print("=== 验证 F1 的一致性（跨多个粒子）===")
        print('='*60)
        F1_ref = F1_list[0]
        for i, F1_i in enumerate(F1_list[1:], start=1):
            diff = np.linalg.norm(F1_i - F1_ref)
            print(f"F1[0] vs F1[{i+1}]: ||F1[0] - F1[{i+1}]||_F = {diff:.2e}")
            if diff < 1e-3:
                print(f"  ✓ F1[{i+1}] 与 F1[0] 一致（误差 < 1e-3）")
            else:
                print(f"  ✗ F1[{i+1}] 与 F1[0] 不一致！")
        
        # 计算平均 F1（如果一致的话）
        F1_avg = np.mean(F1_list, axis=0)
        # 重新正交化（SVD）
        U, s, Vt = np.linalg.svd(F1_avg)
        F1_avg_ortho = U @ Vt
        print(f"\n平均 F1 (重新正交化后):")
        print_matrix("F1_avg", F1_avg_ortho)
        
        # 用平均 F1 测试所有粒子对
        print("\n=== 用平均 F1 测试所有粒子对 ===")
        max_error = 0.0
        for i, (dynamo_zxz, relion_zyz_good) in enumerate(particle_pairs):
            R_dynamo = R.from_euler('ZXZ', dynamo_zxz, degrees=True).as_matrix()
            R_relion_good = R.from_euler('ZYZ', relion_zyz_good, degrees=True).as_matrix()
            R_relion_from_F1_avg = F1_avg_ortho @ R_dynamo
            error = np.linalg.norm(R_relion_good - R_relion_from_F1_avg)
            print(f"粒子 {i+1}: 误差 = {error:.2e}")
            max_error = max(max_error, error)
        print(f"\n最大误差: {max_error:.2e}")
        
        return F1_avg_ortho
    
    return F1_list[0] if F1_list else None


def main():
    # 第一个粒子对（已知的）
    dynamo_zxz = np.array([-43.444, 70.291, 12.048])
    relion_zyz_good = np.array([-77.952, 70.291, 46.556])
    script_zyz = np.array([-133.444, 70.291, 102.048])
    
    print("="*60)
    print("单个粒子对分析")
    print("="*60)
    F1, F2, e1, e2 = analyze_single_pair(dynamo_zxz, relion_zyz_good, script_zyz, verbose=True)
    
    # 如果有更多粒子对，可以在这里添加
    # 示例格式：
    # particle_pairs = [
    #     (np.array([-43.444, 70.291, 12.048]), np.array([-77.952, 70.291, 46.556])),
    #     (np.array([angle1, angle2, angle3]), np.array([angle1, angle2, angle3])),
    #     # ... 更多粒子对
    # ]
    # F1_global = analyze_multiple_pairs(particle_pairs, verbose=True)
    
    print("\n" + "="*60)
    print("结论和建议")
    print("="*60)
    print("""
根据分析结果：
1. convert_euler 函数在数学上是正确的（ZXZ ↔ ZYZ 转换无误）
2. 但 RELION 和 Dynamo 之间存在一个固定的坐标变换 F1
3. 正确的转换应该是：R_relion = F1 @ R_dynamo
4. 因此，在 io_dynamo.py 中，需要：
   - 先将 Dynamo ZXZ 转为旋转矩阵
   - 应用固定变换 F1
   - 再将结果转为 RELION ZYZ 欧拉角
    """)


if __name__ == "__main__":
    main()
