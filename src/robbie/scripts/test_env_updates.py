#!/usr/bin/env python3

"""
Test script to verify:
1. Camera images are updating properly
2. Collision detection triggers episode restart
"""

import rospy
import sys
import os

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from rl_environment import LineFollowingEnv


def test_camera_updates():
    """Test that camera images are being received and processed"""
    print("\n" + "="*60)
    print("TEST 1: Camera Update Test")
    print("="*60)
    
    env = LineFollowingEnv()
    
    print("Waiting for initial image...")
    rospy.sleep(2)
    
    if env.image_received:
        print("✓ Initial image received successfully")
    else:
        print("✗ No image received - check camera topic")
        return False
    
    # Test that state updates with new images
    initial_state = env.current_state.copy()
    print(f"Initial state sum: {initial_state.sum()}")
    
    # Execute a simple action
    print("\nExecuting forward motion...")
    next_state, reward, done, info = env.step(7)  # Middle action - forward
    
    print(f"Next state sum: {next_state.sum()}")
    print(f"Image received flag: {env.image_received}")
    
    if env.image_received:
        print("✓ Camera updated after step")
    else:
        print("✗ Camera did not update after step")
        return False
    
    env.close()
    return True


def test_collision_restart():
    """Test that collision properly ends episode and resets"""
    print("\n" + "="*60)
    print("TEST 2: Collision Restart Test")
    print("="*60)
    
    env = LineFollowingEnv()
    
    print("Waiting for initialization...")
    rospy.sleep(2)
    
    # Manually trigger collision
    print("Simulating collision...")
    env.collision_detected = True
    
    # Take a step
    next_state, reward, done, info = env.step(7)
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    if done and reward == -500.0:
        print("✓ Collision correctly triggered episode end with -500 penalty")
    else:
        print(f"✗ Collision handling incorrect - done={done}, reward={reward}")
        return False
    
    # Test reset
    print("\nTesting reset...")
    initial_state = env.reset()
    
    print(f"Timeout counter after reset: {env.timeout}")
    print(f"Collision flag after reset: {env.collision_detected}")
    print(f"Steps counter after reset: {env.steps}")
    
    if env.timeout == 0 and not env.collision_detected and env.steps == 0:
        print("✓ Reset properly cleared all counters")
    else:
        print("✗ Reset did not clear counters properly")
        return False
    
    env.close()
    return True


def test_timeout_termination():
    """Test that line timeout properly ends episode"""
    print("\n" + "="*60)
    print("TEST 3: Line Lost Timeout Test")
    print("="*60)
    
    env = LineFollowingEnv()
    
    print("Waiting for initialization...")
    rospy.sleep(2)
    
    # Manually set timeout to trigger termination
    print(f"Setting timeout to {env.max_timeout + 1}...")
    env.timeout = env.max_timeout + 1
    
    # Take a step
    next_state, reward, done, info = env.step(7)
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    if done and reward == -500.0 and info.get('termination_reason') == 'line_lost':
        print("✓ Timeout correctly triggered episode end with -500 penalty")
    else:
        print(f"✗ Timeout handling incorrect - done={done}, reward={reward}, info={info}")
        return False
    
    env.close()
    return True


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ENVIRONMENT UPDATE VERIFICATION TESTS")
    print("="*60)
    
    try:
        # Run tests
        test1_passed = test_camera_updates()
        rospy.sleep(1)
        
        test2_passed = test_collision_restart()
        rospy.sleep(1)
        
        test3_passed = test_timeout_termination()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Camera Updates:       {'✓ PASSED' if test1_passed else '✗ FAILED'}")
        print(f"Collision Restart:    {'✓ PASSED' if test2_passed else '✗ FAILED'}")
        print(f"Timeout Termination:  {'✓ PASSED' if test3_passed else '✗ FAILED'}")
        
        if test1_passed and test2_passed and test3_passed:
            print("\n🎉 All tests passed! Environment is ready for training.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
