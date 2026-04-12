#!/usr/bin/env python3
"""
验证自定义 Policy 配置的测试脚本

使用方法：
    python scripts/test_custom_policy.py --config g1_my_rl
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def test_policy_config(config_name: str):
    """测试 policy 配置是否正确"""
    
    print("=" * 60)
    print(f"Testing Policy Configuration: {config_name}")
    print("=" * 60)
    
    # 1. 测试配置加载
    print("\n[1/6] Loading configuration...")
    try:
        from robojudo.config.config_manager import ConfigManager
        config_manager = ConfigManager(config_name=config_name)
        cfg = config_manager.get_cfg()
        print("✓ Configuration loaded successfully")
        print(f"  - Robot: {cfg.robot}")
        print(f"  - Pipeline: {cfg.pipeline_type}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # 2. 测试 Policy 类是否可导入
    print("\n[2/6] Checking policy class...")
    try:
        import robojudo.policy
        policy_type = cfg.policy.policy_type
        policy_class = getattr(robojudo.policy, policy_type, None)
        if policy_class is None:
            print(f"✗ Policy class '{policy_type}' not found")
            print(f"  Available policies: {dir(robojudo.policy)}")
            return False
        print(f"✓ Policy class '{policy_type}' found")
    except Exception as e:
        print(f"✗ Failed to import policy: {e}")
        return False
    
    # 3. 测试模型文件是否存在
    print("\n[3/6] Checking model file...")
    model_path = Path(cfg.policy.policy_file)
    if not model_path.exists():
        print(f"✗ Model file not found: {cfg.policy.policy_file}")
        print(f"  Please ensure your model file is at this location")
        return False
    print(f"✓ Model file exists: {cfg.policy.policy_file}")
    print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 4. 测试 Policy 初始化
    print("\n[4/6] Initializing policy...")
    try:
        policy = policy_class(cfg_policy=cfg.policy, device="cpu")
        print("✓ Policy initialized successfully")
        print(f"  - Observation DoFs: {policy.num_dofs}")
        print(f"  - Action DoFs: {policy.num_actions}")
        print(f"  - Frequency: {policy.freq} Hz")
        print(f"  - History length: {policy.history_length}")
    except Exception as e:
        print(f"✗ Failed to initialize policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试观测空间
    print("\n[5/6] Testing observation space...")
    try:
        # 创建模拟的环境数据
        class MockEnvData:
            def __init__(self, num_dofs):
                self.base_pos = np.array([0.0, 0.0, 0.79], dtype=np.float32)
                self.base_ang_vel = np.zeros(3)
                self.base_lin_vel = np.zeros(3)
                self.base_quat = np.array([1, 0, 0, 0])  # w, x, y, z
                self.dof_pos = np.zeros(num_dofs)
                self.dof_vel = np.zeros(num_dofs)
                self.imu_acc = np.zeros(3)
                self.imu_gyro = np.zeros(3)
        
        env_data = MockEnvData(policy.num_dofs)
        ctrl_data = {}  # 空控制数据
        
        obs, info = policy.get_observation(env_data, ctrl_data)
        print("✓ Observation generated successfully")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # 检查是否有 NaN 或 Inf
        if np.isnan(obs).any():
            print("  ⚠ Warning: Observation contains NaN values")
        if np.isinf(obs).any():
            print("  ⚠ Warning: Observation contains Inf values")
    except Exception as e:
        print(f"✗ Failed to generate observation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 测试动作推理
    print("\n[6/6] Testing action inference...")
    try:
        action = policy.get_action(obs)
        print("✓ Action generated successfully")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  - Action mean: {action.mean():.3f}")
        print(f"  - Action std: {action.std():.3f}")
        
        # 检查动作是否合理
        if np.abs(action).max() > 100:
            print("  ⚠ Warning: Actions seem very large (> 100)")
        if np.isnan(action).any():
            print("  ⚠ Warning: Action contains NaN values")
            return False
        if np.isinf(action).any():
            print("  ⚠ Warning: Action contains Inf values")
            return False
    except Exception as e:
        print(f"✗ Failed to generate action: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 总结
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nYour policy is ready to run!")
    print(f"\nRun with: python scripts/run_pipeline.py --config {config_name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test custom policy configuration"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Name of the config to test"
    )
    args = parser.parse_args()
    
    success = test_policy_config(args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
