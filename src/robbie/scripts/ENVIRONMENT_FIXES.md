# Environment Update Fixes

## Issues Identified

### 1. Camera Not Updating Constantly
**Problem**: The environment was not waiting for new camera images between steps, causing the agent to potentially train on stale image data.

**Root Cause**: The `step()` function only waited a fixed 0.1 seconds, which might not be enough time for a new camera frame to arrive and be processed.

**Solution**: 
- Added `image_received` flag that gets set to `True` when a new camera image is processed
- Modified `step()` to reset this flag before executing action
- Added wait loop that waits for `image_received` to become `True` before proceeding
- Maximum wait time of 5 seconds with timeout warning if image doesn't arrive

### 2. Collision Not Restarting Training
**Problem**: When a collision occurred, the episode would end but the next episode wouldn't start fresh.

**Root Cause**: 
- The `collision_detected` flag was being reset in `_calculate_reward()` instead of in `reset()`
- The `timeout` counter was not being reset during episode reset
- No verification that a fresh image was received after reset

**Solution**:
- Moved collision flag reset to `reset()` method
- Added `timeout` counter reset in `reset()` method
- Added wait loop in `reset()` to ensure a fresh camera image is received before starting new episode
- Removed premature flag reset from `_calculate_reward()`

## Code Changes

### rl_environment.py

1. **Added image tracking flag** (line ~44):
```python
self.image_received = False  # Track if we've received at least one image
```

2. **Updated image_callback** (line ~110):
```python
def image_callback(self, msg):
    # ... existing code ...
    self.image_received = True  # Mark that we've received an image
```

3. **Updated step() method** (line ~205):
```python
def step(self, action_idx):
    # Mark that we need a new image
    self.image_received = False
    
    # Execute action
    linear_vel, angular_vel = self.action_space[action_idx]
    self._publish_velocity(linear_vel, angular_vel)
    
    # Wait for new camera image to arrive
    timeout_counter = 0
    max_wait = 50  # 5 seconds max wait (0.1s * 50)
    while not self.image_received and timeout_counter < max_wait:
        rospy.sleep(0.1)
        timeout_counter += 1
    
    if timeout_counter >= max_wait:
        rospy.logwarn("Timeout waiting for camera image!")
    
    # Get next state (updated by image callback)
    next_state = self.current_state.copy()
    # ... rest of method ...
```

4. **Updated reset() method** (line ~375):
```python
def reset(self, random_start=True):
    # ... existing position reset code ...
    
    # Reset episode tracking
    self.steps = 0
    self.episode_reward = 0
    self.collision_detected = False
    self.timeout = 0  # Reset timeout counter
    self.image_received = False  # Wait for fresh image after reset
    
    # Wait for state to update and get fresh camera image
    timeout_counter = 0
    max_wait = 50  # 5 seconds max wait
    while not self.image_received and timeout_counter < max_wait:
        rospy.sleep(0.1)
        timeout_counter += 1
    
    return self.current_state.copy()
```

5. **Improved collision handling** (line ~257):
```python
# TERMINATION CONDITION 1: COLLISION PENALTY
if self.collision_detected:
    reward = -500.0  # Massive penalty for collision
    done = True
    info['collision'] = True
    info['termination_reason'] = 'collision'
    rospy.logwarn(f"COLLISION DETECTED! Episode ending with penalty: {reward}")
    # Don't reset flag here - let reset() handle it
    return reward, done, info
```

### train_rl.py

**Updated progress logging** (line ~150):
```python
# Print progress
termination_reason = info.get('termination_reason', 'max_steps')
print(f"Episode {episode+1}/{self.num_episodes} | "
      f"Steps: {step+1} | "
      f"Reward: {episode_reward:.2f} | "
      f"Avg(100): {avg_reward:.2f} | "
      f"Epsilon: {self.agent.epsilon:.3f} | "
      f"Loss: {np.mean(episode_loss) if episode_loss else 0:.4f} | "
      f"End: {termination_reason}")
```

## Testing

Run the verification test script:
```bash
cd /home/fizzer/ENPH-353-COMPETITION/src/robbie/scripts
source ~/ENPH-353-COMPETITION/devel/setup.bash
python3 test_env_updates.py
```

This will verify:
1. ✓ Camera images are being received and updating
2. ✓ Collision detection properly ends episode and resets
3. ✓ Timeout termination works correctly

## Expected Behavior

After these fixes:
- **Camera updates**: Each step will wait for a fresh camera image before proceeding
- **Collision handling**: When robot collides, episode ends with -500 penalty and next episode starts fresh from reset position
- **Line timeout**: If robot loses line for 30+ frames, episode ends with -500 penalty
- **Training logs**: Show termination reason (collision, line_lost, or max_steps)

## Impact on Training

These fixes ensure:
1. Agent always trains on current, up-to-date camera data
2. Collision penalties are properly learned (episode truly restarts after collision)
3. Episodes cleanly separate with all counters properly reset
4. More reliable and consistent training behavior
