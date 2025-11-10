#!/usr/bin/env python3

"""
Quick start script for RL line following
Handles environment setup and provides easy command interface
"""

import os
import sys
import subprocess
import argparse


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    missing = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
    
    try:
        import rospy
        print(f"✓ ROS Python")
    except ImportError:
        print("✗ ROS Python not found - make sure ROS is sourced")
        return False
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies satisfied!\n")
    return True


def train(args):
    """Start training"""
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    cmd = ["python3", "train_rl.py"]
    
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.render:
        cmd.append("--render")
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.checkpoint_dir:
        cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
    
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def test(args):
    """Run inference"""
    print("="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    if not args.checkpoint:
        # Find latest checkpoint
        checkpoint_dir = args.checkpoint_dir or "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort()
                args.checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Using latest checkpoint: {args.checkpoint}\n")
            else:
                print("No checkpoints found! Train a model first.")
                return
        else:
            print("No checkpoint directory found! Train a model first.")
            return
    
    cmd = ["python3", "run_inference.py", args.checkpoint]
    
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.continuous:
        cmd.append("--continuous")
    if args.no_render:
        cmd.append("--no-render")
    
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def visualize(args):
    """Run bin visualizer"""
    print("="*60)
    print("STARTING BIN VISUALIZER")
    print("="*60)
    print("This will show how camera images are processed into bins")
    print("Press 'q' to quit\n")
    
    subprocess.run(["python3", "visualize_bins.py"])


def info(args):
    """Show system information"""
    print("="*60)
    print("RL LINE FOLLOWING SYSTEM INFO")
    print("="*60)
    
    # Check for checkpoints
    checkpoint_dir = args.checkpoint_dir or "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        metrics = [f for f in os.listdir(checkpoint_dir) if f.startswith('metrics_')]
        
        print(f"\nCheckpoint directory: {checkpoint_dir}")
        print(f"  Model checkpoints: {len(checkpoints)}")
        print(f"  Metrics files: {len(metrics)}")
        
        if checkpoints:
            print(f"\nLatest checkpoint: {sorted(checkpoints)[-1]}")
            
            # Try to load and show info
            try:
                import torch
                latest = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
                checkpoint = torch.load(latest, map_location='cpu')
                print(f"  Episodes trained: {checkpoint.get('episode_count', 'unknown')}")
                print(f"  Training steps: {checkpoint.get('training_step', 'unknown')}")
                print(f"  Epsilon: {checkpoint.get('epsilon', 'unknown'):.4f}")
            except:
                pass
    else:
        print(f"\nNo checkpoint directory found at: {checkpoint_dir}")
    
    # Check ROS topics
    print("\n" + "="*60)
    print("Checking ROS environment...")
    
    try:
        result = subprocess.run(["rostopic", "list"], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            topics = result.stdout.strip().split('\n')
            camera_topics = [t for t in topics if 'camera' in t.lower() or 'image' in t.lower()]
            cmd_topics = [t for t in topics if 'cmd_vel' in t]
            collision_topics = [t for t in topics if 'isHit' in t or 'collision' in t.lower()]
            
            print(f"✓ ROS is running ({len(topics)} topics)")
            if camera_topics:
                print(f"  Camera topics: {', '.join(camera_topics[:3])}")
            if cmd_topics:
                print(f"  Command topics: {', '.join(cmd_topics)}")
            if collision_topics:
                print(f"  Collision topics: {', '.join(collision_topics)}")
        else:
            print("✗ ROS master not running")
            print("  Start simulation first!")
    except:
        print("✗ ROS not accessible")
        print("  Make sure ROS is sourced and simulation is running")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Quick start script for RL line following",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check dependencies and info:
    ./quickstart.py info
    
  Train for 500 episodes with rendering:
    ./quickstart.py train --episodes 500 --render
    
  Resume training from checkpoint:
    ./quickstart.py train --resume checkpoints/dqn_ep_250_*.pth
    
  Test latest trained model:
    ./quickstart.py test
    
  Test specific checkpoint:
    ./quickstart.py test --checkpoint checkpoints/dqn_ep_500_*.pth
    
  Visualize camera bins:
    ./quickstart.py visualize
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--episodes', type=int, help='Number of episodes')
    train_parser.add_argument('--render', action='store_true', help='Render during training')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run inference')
    test_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    test_parser.add_argument('--episodes', type=int, help='Number of test episodes')
    test_parser.add_argument('--continuous', action='store_true', help='Run continuously')
    test_parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    test_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize camera bins')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                            help='Checkpoint directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Always check dependencies first (except for info)
    if args.command != 'info':
        if not check_dependencies():
            sys.exit(1)
    
    # Run command
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'visualize':
        visualize(args)
    elif args.command == 'info':
        info(args)


if __name__ == '__main__':
    main()
