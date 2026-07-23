#!/usr/bin/env python3
"""
Test script for Enhanced PiKV MoE implementation

This script tests the three optional implementations:
1. Dynamic Load Balancing
2. Async Execution
3. Communication Optimization
"""

import torch
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")

    try:
        from core.single.enhanced_pikv_moe import EnhancedPiKVMoE, create_enhanced_pikv_moe
        from core.single.enhanced_config import get_enhanced_config, create_optimization_presets
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of enhanced MoE"""
    print("\nTesting basic functionality...")

    try:
        from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

        model = create_enhanced_pikv_moe(
            enable_dynamic_balancing=False,
            enable_async_execution=False,
            enable_communication_optimization=False,
            enable_smartmoe=False,
        )

        x = torch.randn(4, 32, 512)
        query = torch.randn(4, 32, 512)

        with torch.no_grad():
            logits = model(x, query=query)

        assert logits.shape == (4, 32, 10000), f"Expected shape (4, 32, 10000), got {logits.shape}"
        print("✓ Basic functionality works")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_dynamic_load_balancing():
    """Test dynamic load balancing functionality"""
    print("\nTesting dynamic load balancing...")

    try:
        from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

        model = create_enhanced_pikv_moe(
            enable_dynamic_balancing=True,
            enable_async_execution=False,
            enable_communication_optimization=False,
            enable_smartmoe=False,
        )

        x = torch.randn(4, 32, 512)
        query = torch.randn(4, 32, 512)

        with torch.no_grad():
            logits = model(x, query=query)

        assert hasattr(model, 'load_balancer'), "Load balancer not found"
        assert model.enable_dynamic_balancing, "Dynamic balancing not enabled"

        metrics = model.get_performance_metrics()
        assert 'load_balancing' in metrics, "Load balancing metrics not found"

        print("✓ Dynamic load balancing works")
        return True

    except Exception as e:
        print(f"✗ Dynamic load balancing test failed: {e}")
        return False


def test_async_execution():
    """Test async execution functionality"""
    print("\nTesting async execution...")

    try:
        from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

        model = create_enhanced_pikv_moe(
            enable_dynamic_balancing=False,
            enable_async_execution=True,
            enable_communication_optimization=False,
            enable_smartmoe=False,
        )

        x = torch.randn(4, 32, 512)
        query = torch.randn(4, 32, 512)

        with torch.no_grad():
            logits = model(x, query=query)

        assert hasattr(model, 'async_manager'), "Async manager not found"
        assert model.enable_async_execution, "Async execution not enabled"

        print("✓ Async execution works")
        return True

    except Exception as e:
        print(f"✗ Async execution test failed: {e}")
        return False


def test_communication_optimization():
    """Test communication optimization functionality"""
    print("\nTesting communication optimization...")

    try:
        from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

        model = create_enhanced_pikv_moe(
            enable_dynamic_balancing=False,
            enable_async_execution=False,
            enable_communication_optimization=True,
            enable_smartmoe=False,
            world_size=2,
        )

        x = torch.randn(4, 32, 512)
        query = torch.randn(4, 32, 512)

        with torch.no_grad():
            logits = model(x, query=query)

        assert hasattr(model, 'communication_placer'), "Communication placer not found"
        assert model.enable_communication_optimization, "Communication optimization not enabled"

        metrics = model.get_performance_metrics()
        assert 'communication' in metrics, "Communication metrics not found"

        print("✓ Communication optimization works")
        return True

    except Exception as e:
        print(f"✗ Communication optimization test failed: {e}")
        return False


def test_combined_optimizations():
    """Test all optimizations combined"""
    print("\nTesting combined optimizations...")

    try:
        from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

        model = create_enhanced_pikv_moe(
            enable_dynamic_balancing=True,
            enable_async_execution=True,
            enable_communication_optimization=True,
            enable_smartmoe=False,
            world_size=2,
        )

        x = torch.randn(4, 32, 512)
        query = torch.randn(4, 32, 512)

        with torch.no_grad():
            logits = model(x, query=query)

        assert model.enable_dynamic_balancing, "Dynamic balancing not enabled"
        assert model.enable_async_execution, "Async execution not enabled"
        assert model.enable_communication_optimization, "Communication optimization not enabled"

        metrics = model.get_performance_metrics()
        assert 'load_balancing' in metrics, "Load balancing metrics not found"
        assert 'communication' in metrics, "Communication metrics not found"

        print("✓ Combined optimizations work")
        return True

    except Exception as e:
        print(f"✗ Combined optimizations test failed: {e}")
        return False


def test_configuration():
    """Test configuration functionality"""
    print("\nTesting configuration...")

    try:
        from core.single.enhanced_config import get_enhanced_config, create_optimization_presets

        config = get_enhanced_config(
            load_balancing_strategy='adaptive',
            execution_mode='async',
            communication_strategy='topology_aware',
        )

        assert config['load_balancing']['strategy'].value == 'adaptive'
        assert config['async_execution']['mode'].value == 'async'
        assert config['communication_optimization']['strategy'].value == 'topology_aware'

        presets = create_optimization_presets()
        assert 'balanced' in presets
        assert 'high_performance' in presets
        assert 'memory_efficient' in presets

        print("✓ Configuration works")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_optimization_toggles():
    """Test optimization enable/disable functionality"""
    print("\nTesting optimization toggles...")

    try:
        from core.single.enhanced_pikv_moe import create_enhanced_pikv_moe

        model = create_enhanced_pikv_moe(
            enable_dynamic_balancing=True,
            enable_async_execution=True,
            enable_communication_optimization=True,
            enable_smartmoe=False,
        )

        model.disable_dynamic_load_balancing()
        assert not model.enable_dynamic_balancing, "Failed to disable load balancing"

        model.disable_async_execution_mode()
        assert not model.enable_async_execution, "Failed to disable async execution"

        model.disable_communication_optimization()
        assert not model.enable_communication_optimization, "Failed to disable communication optimization"

        model.enable_dynamic_load_balancing()
        assert model.enable_dynamic_balancing, "Failed to enable load balancing"

        model.enable_async_execution_mode()
        assert model.enable_async_execution, "Failed to enable async execution"

        model.enable_communication_optimization()
        assert model.enable_communication_optimization, "Failed to enable communication optimization"

        print("✓ Optimization toggles work")
        return True

    except Exception as e:
        print(f"✗ Optimization toggles test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Enhanced PiKV MoE Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_basic_functionality,
        test_dynamic_load_balancing,
        test_async_execution,
        test_communication_optimization,
        test_combined_optimizations,
        test_configuration,
        test_optimization_toggles,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! Enhanced PiKV MoE is working correctly.")
        return True

    print("Some tests failed. Please check the implementation.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
